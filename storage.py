from __future__ import annotations

import json
import math
import os
import struct
import threading
import time
import unicodedata
from dataclasses import dataclass
from hashlib import sha256
from typing import Optional, Tuple, Union


ByteLike = Union[bytes, bytearray]


# ---------- NB value quantization and path mapping (8 decimal places) ----------


def quantize8(x: float) -> Tuple[int, str, bool]:
    neg = x < 0
    x = abs(x)
    q = round(float(x), 8)
    int_part = int(math.floor(q))
    frac = q - int_part
    dec8 = f"{frac:.8f}".split(".")[1]  # always 8 digits
    return int_part, dec8, neg


def nb_folder_name(int_part: int, dec8: str, neg: bool) -> str:
    return f"{'NB_NEG' if neg else 'NB'}_{int_part}_{dec8}"


def nb_bucket_path(root: str, x: float) -> str:
    int_part, dec8, neg = quantize8(x)
    d1, d2, d3 = dec8[:2], dec8[2:5], dec8[5:8]
    parts = [root]
    if neg:
        parts.append("neg")
    parts += [str(int_part), d1, d2, d3]
    return os.path.join(*parts, nb_folder_name(int_part, dec8, neg))


def nb_key_from_value(x: float) -> str:
    int_part, dec8, neg = quantize8(x)
    return nb_folder_name(int_part, dec8, neg)


# ---------- Simple segment file storage with manifest + WAL ----------


SEGMENT_FILE = "data.seg"
MANIFEST_FILE = "manifest.json"  # { key: {"offset": int, "length": int, ...} }
WAL_FILE = "wal.log"  # text lines: PUT/DEL key offset length checksum ts
BF_FILE = "manifest.bf"  # bloom filter dump


@dataclass
class Pointer:
    offset: int
    length: int
    checksum: str
    created_at: float
    deleted: bool = False


def _now() -> float:
    return time.time()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _checksum(data: ByteLike) -> str:
    return sha256(bytes(data)).hexdigest()[:16]


class SegmentStore:
    """Append-only segment with fixed record framing: [u64 length][bytes]."""

    def __init__(self, folder: str) -> None:
        self.folder = folder
        _ensure_dir(folder)
        self.seg_path = os.path.join(folder, SEGMENT_FILE)
        self.manifest_path = os.path.join(folder, MANIFEST_FILE)
        self.wal_path = os.path.join(folder, WAL_FILE)
        self._lock = threading.Lock()
        # load manifest if exists
        self._manifest = self._load_manifest()
        # bloom filter
        try:
            from .bloom import BloomFilter  # type: ignore
        except Exception:
            from bloom import BloomFilter  # type: ignore
        self._bf = BloomFilter(os.path.join(folder, BF_FILE))
        for k in self._manifest.keys():
            if not self._manifest[k].get("deleted"):
                self._bf.add(k)

    def _load_manifest(self) -> dict:
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_manifest(self) -> None:
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        os.replace(tmp, self.manifest_path)

    def _append_wal(self, line: str) -> None:
        with open(self.wal_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def put(self, key: str, value: Union[str, ByteLike, dict]) -> Pointer:
        if isinstance(value, dict):
            data = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        elif isinstance(value, str):
            data = value.encode("utf-8")
        else:
            data = bytes(value)
        chk = _checksum(data)
        ts = _now()
        with self._lock:
            # append record
            with open(self.seg_path, "ab") as seg:
                offset = seg.tell()
                seg.write(struct.pack(">Q", len(data)))
                seg.write(data)
            ptr = {
                "offset": offset,
                "length": len(data),
                "checksum": chk,
                "created_at": ts,
                "deleted": False,
            }
            self._manifest[key] = ptr
            self._save_manifest()
            self._bf.add(key)
            self._bf.flush()
            self._append_wal(f"PUT {key} {offset} {len(data)} {chk} {int(ts)}")
            return Pointer(**ptr)

    def get(self, key: str) -> Optional[bytes]:
        # quick reject using bloom
        if not self._bf.contains(key):
            return None
        meta = self._manifest.get(key)
        if not meta or meta.get("deleted"):
            return None
        offset = int(meta["offset"])
        length = int(meta["length"])
        with open(self.seg_path, "rb") as seg:
            seg.seek(offset)
            stored_len = struct.unpack(">Q", seg.read(8))[0]
            if stored_len != length:
                return None
            data = seg.read(length)
        return data

    def delete(self, key: str) -> bool:
        if key not in self._manifest:
            return False
        with self._lock:
            self._manifest[key]["deleted"] = True
            self._save_manifest()
            # bloom filter는 false positive만 보장하므로 재구축 대상. 간단히 즉시 flush만 수행
            self._bf.flush()
            self._append_wal(f"DEL {key} {int(_now())}")
        return True


class NbStorage:
    """High-level API mapping NB float value to folder and storing payloads."""

    def __init__(self, root: str) -> None:
        self.root = root

    def _store_for_value(self, nb_value: float) -> Tuple[str, SegmentStore, str]:
        folder = nb_bucket_path(self.root, nb_value)
        store = SegmentStore(folder)
        key = nb_key_from_value(nb_value)
        return folder, store, key

    def put(self, nb_value: float, payload: Union[str, ByteLike, dict]) -> Pointer:
        _, store, key = self._store_for_value(nb_value)
        return store.put(key, payload)

    def get(self, nb_value: float) -> Optional[bytes]:
        _, store, key = self._store_for_value(nb_value)
        return store.get(key)

    def delete(self, nb_value: float) -> bool:
        _, store, key = self._store_for_value(nb_value)
        return store.delete(key)

    def get_or_put(
        self,
        nb_value: float,
        supplier,
    ) -> Tuple[Optional[bytes], bool]:
        """
        Read-through cache.
        - If present, return (bytes, True).
        - If absent, compute via supplier() -> dict|str|bytes, store, and return (bytes, False).
        """
        blob = self.get(nb_value)
        if blob is not None:
            return blob, True
        value = supplier()
        # Reuse SegmentStore conversion logic by calling put, then get again
        self.put(nb_value, value)
        return self.get(nb_value), False

    # ---------- Extended API: allow custom record keys within NB bucket ----------
    def put_with_key(self, nb_value: float, key: str, payload: Union[str, ByteLike, dict]) -> Pointer:
        folder = nb_bucket_path(self.root, nb_value)
        store = SegmentStore(folder)
        return store.put(key, payload)

    def get_with_key(self, nb_value: float, key: str) -> Optional[bytes]:
        folder = nb_bucket_path(self.root, nb_value)
        store = SegmentStore(folder)
        return store.get(key)

    def delete_with_key(self, nb_value: float, key: str) -> bool:
        folder = nb_bucket_path(self.root, nb_value)
        store = SegmentStore(folder)
        return store.delete(key)

    def get_or_put_with_key(self, nb_value: float, key: str, supplier) -> Tuple[Optional[bytes], bool]:
        blob = self.get_with_key(nb_value, key)
        if blob is not None:
            return blob, True
        value = supplier()
        self.put_with_key(nb_value, key, value)
        return self.get_with_key(nb_value, key), False


