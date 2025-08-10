from __future__ import annotations

from typing import Dict, List, Tuple, Iterable
import math
import hashlib
import re


# Global state analogous to JS SUPER_BIT
SUPER_BIT: float = 0.0


def initializeArrays(length: int) -> Dict[str, List[float]]:
    return {
        "BIT_START_A50": [0.0] * length,
        "BIT_START_A100": [0.0] * length,
        "BIT_START_B50": [0.0] * length,
        "BIT_START_B100": [0.0] * length,
        "BIT_START_NBA100": [0.0] * length,
    }


def calculateBit(nb: List[float], bit: float = 5.5, reverse: bool = False) -> float:
    if len(nb) < 2:
        return bit / 100.0

    BIT_NB = bit
    max_v = max(nb)
    min_v = min(nb)
    COUNT = 50
    # CONT is unused in original JS; keeping for parity
    # CONT = 20

    negativeRange = abs(min_v) if min_v < 0 else 0.0
    positiveRange = max_v if max_v > 0 else 0.0

    denom = (COUNT * len(nb) - 1)
    if denom <= 0:
        return bit / 100.0
    negativeIncrement = negativeRange / denom
    positiveIncrement = positiveRange / denom

    arrays = initializeArrays(COUNT * len(nb))
    count = 0
    totalSum = 0.0

    for value in nb:
        for _ in range(COUNT):
            BIT_END = 1
            if value < 0:
                A50 = min_v + negativeIncrement * (count + 1)
            else:
                A50 = min_v + positiveIncrement * (count + 1)

            A100 = (count + 1) * BIT_NB / (COUNT * len(nb))

            if value < 0:
                B50 = A50 - negativeIncrement * 2
                B100 = A50 + negativeIncrement
            else:
                B50 = A50 - positiveIncrement * 2
                B100 = A50 + positiveIncrement

            # Prevent division by zero
            denom2 = max(1, (len(nb) - BIT_END))
            NBA100 = A100 / denom2

            arrays["BIT_START_A50"][count] = float(A50)
            arrays["BIT_START_A100"][count] = float(A100)
            arrays["BIT_START_B50"][count] = float(B50)
            arrays["BIT_START_B100"][count] = float(B100)
            arrays["BIT_START_NBA100"][count] = float(NBA100)
            count += 1
        totalSum += float(value)

    if reverse:
        arrays["BIT_START_NBA100"].reverse()

    NB50 = 0.0
    for value in nb:
        for a in range(len(arrays["BIT_START_NBA100"])):
            if arrays["BIT_START_B50"][a] <= value <= arrays["BIT_START_B100"][a]:
                idx = min(a, len(arrays["BIT_START_NBA100"]) - 1)
                NB50 += arrays["BIT_START_NBA100"][idx]
                break

    if len(nb) == 2:
        return bit - NB50

    return NB50


def updateSuperBit(newValue: float) -> None:
    global SUPER_BIT
    SUPER_BIT = float(newValue)


def BIT_MAX_NB(nb: List[float], bit: float = 5.5) -> float:
    result = float(calculateBit(nb, bit, False))
    if not math.isfinite(result) or result > 100 or result < -100:
        return SUPER_BIT
    updateSuperBit(result)
    return result


def BIT_MIN_NB(nb: List[float], bit: float = 5.5) -> float:
    result = float(calculateBit(nb, bit, True))
    if not math.isfinite(result) or result > 100 or result < -100:
        return SUPER_BIT
    updateSuperBit(result)
    return result


def calculateArrayOrderAndDuplicate(nb1: List[float], nb2: List[float]) -> Dict[str, float]:
    length1 = len(nb1)
    length2 = len(nb2)

    elementCount1: Dict[float, int] = {}
    elementCount2: Dict[float, int] = {}
    for v in nb1:
        elementCount1[v] = elementCount1.get(v, 0) + 1
    for v in nb2:
        elementCount2[v] = elementCount2.get(v, 0) + 1

    duplicateMatch = 0
    for k, c1 in elementCount1.items():
        c2 = elementCount2.get(k, 0)
        if c1 >= 1 and c2 >= 1:
            duplicateMatch += min(c1, c2)

    maxOrderMatch = 0
    for i in range(length1):
        for j in range(length2):
            if nb1[i] == nb2[j]:
                tempMatch = 0
                x, y = i, j
                while x < length1 and y < length2 and nb1[x] == nb2[y]:
                    tempMatch += 1
                    x += 1
                    y += 1
                if tempMatch > maxOrderMatch:
                    maxOrderMatch = tempMatch

    orderMatchRatio = (maxOrderMatch / max(1, min(length1, length2))) * 100.0
    duplicateMatchRatioLeft = (duplicateMatch / max(1, length1)) * 100.0
    duplicateMatchRatioRight = (duplicateMatch / max(1, length2)) * 100.0
    duplicateMatchRatio = (duplicateMatchRatioLeft + duplicateMatchRatioRight) / 2.0

    if length2 < length1:
        lengthDifference = (length2 / length1) * 100.0
    else:
        lengthDifference = (length1 / length2) * 100.0

    return {
        "orderMatchRatio": orderMatchRatio,
        "duplicateMatchRatio": duplicateMatchRatio,
        "duplicateMatchRatioLeft": duplicateMatchRatioLeft,
        "duplicateMatchRatioRight": duplicateMatchRatioRight,
        "lengthDifference": lengthDifference,
    }


def calculateInclusionFromBase(sentence1: str, sentence2: str) -> Dict[str, object]:
    if not sentence1 or not sentence2:
        return {"matched": 0, "total": 0, "ratio": 0.0, "matchedWords": []}

    def clean(s: str) -> str:
        # Keep word chars and spaces
        s2 = re.sub(r"[^\w\s]", " ", s)
        s2 = re.sub(r"\s+", " ", s2).strip()
        return s2

    baseWords = clean(sentence1).split(" ") if sentence1 else []
    compareWords = set(clean(sentence2).split(" ")) if sentence2 else set()

    matchedWords: List[str] = []
    for w in baseWords:
        if w in compareWords:
            matchedWords.append(w)

    matchCount = len(matchedWords)
    ratio = (matchCount / max(1, len(baseWords))) * 100.0
    return {"matched": matchCount, "total": len(baseWords), "ratio": float(f"{ratio:.5f}"), "matchedWords": matchedWords}


def levenshtein(a: str, b: str) -> int:
    # Classic DP implementation
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    dp = [[0] * (la + 1) for _ in range(lb + 1)]
    for i in range(lb + 1):
        dp[i][0] = i
    for j in range(la + 1):
        dp[0][j] = j
    for i in range(1, lb + 1):
        for j in range(1, la + 1):
            cost = 0 if b[i - 1] == a[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[lb][la]


def calculateLevenshteinSimilarity(nb1: List[str], nb2: List[str]) -> float:
    if not nb1 or not nb2:
        return 0.0
    totalSimilarity = 0.0
    for i in range(len(nb1)):
        best = math.inf
        for j in range(len(nb2)):
            d = levenshtein(nb1[i], nb2[j])
            if d < best:
                best = d
        max_len = max(len(nb1[i]), len(nb2[0]) if nb2 else 1)
        max_len = max(1, max_len)
        similarity = ((max_len - best) / max_len) * 100.0
        totalSimilarity += similarity
    return totalSimilarity / max(1, len(nb1))


def soundex(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    if not s:
        return "0000"
    first = s[0]
    mapping = {
        ("b", "f", "p", "v"): "1",
        ("c", "g", "j", "k", "q", "s", "x", "z"): "2",
        ("d", "t"): "3",
        ("l",): "4",
        ("m", "n"): "5",
        ("r",): "6",
    }

    def map_char(c: str) -> str:
        for keys, val in mapping.items():
            if c in keys:
                return val
        return ""

    coded: List[str] = []
    prev = None
    for c in s[1:]:
        code = map_char(c)
        if code and code != prev:
            coded.append(code)
        prev = code
    res = (first + "".join(coded) + "000")[:4].upper()
    return res


def calculateSoundexMatch(nb1: List[str], nb2: List[str]) -> float:
    if not nb1 or not nb2:
        return 0.0
    soundexMatch = 0
    nb2_sx = [soundex(x) for x in nb2]
    for v in nb1:
        sx = soundex(v)
        for sx2 in nb2_sx:
            if sx == sx2:
                soundexMatch += 1
    denom = max(1, min(len(nb1), len(nb2)))
    return (soundexMatch / denom) * 100.0


def calculateBitArrayOrderAndDuplicate(nb1: List[float], nb2: List[float], bit: float = 5.5) -> Dict[str, float]:
    return calculateArrayOrderAndDuplicate(nb1, nb2)


def wordSim(nbMax: float = 100, nbMin: float = 50, max_v: float = 100, min_v: float = 50) -> float:
    simMax = (nbMax / max_v * 100.0) if nbMax <= max_v and max_v != 0 else (max_v / max(nbMax, 1e-9)) * 100.0
    simMax = 100 - abs(simMax) if abs(simMax) > 100 else simMax
    if nbMax == max_v:
        simMax = 99.99

    simMin = (nbMin / min_v * 100.0) if nbMin <= min_v and min_v != 0 else (min_v / max(nbMin, 1e-9)) * 100.0
    simMin = 100 - abs(simMin) if abs(simMin) > 100 else simMin
    if nbMin == min_v:
        simMin = 99.99

    similarity = (simMax + simMin) / 2.0
    return abs(similarity)


def wordSim2(nbMax: float = 100, max_v: float = 100) -> float:
    simMax = (nbMax / max_v * 100.0) if nbMax <= max_v and max_v != 0 else (max_v / max(nbMax, 1e-9)) * 100.0
    if nbMax == max_v:
        simMax = 99.99
    return abs(simMax)


def calculateArraySimilarity(array1: List[float], array2: List[float]) -> float:
    set2 = set(array2)
    intersection = [v for v in array1 if v in set2]
    union = set(array1)
    union.update(array2)
    jaccardSimilarity = (len(intersection) / len(union) * 100.0) if len(union) > 0 else 0.0

    orderedMatches = [v for i, v in enumerate(array1) if i < len(array2) and v == array2[i]]
    orderedSimilarity = (len(orderedMatches) / len(array1) * 100.0) if (len(array1) > 0 and len(array1) == len(array2)) else 0.0

    return jaccardSimilarity * 0.5 + orderedSimilarity * 0.5


def areLanguagesSame(str1: str, str2: str) -> bool:
    return identifyLanguage(str1) == identifyLanguage(str2)


def wordNbUnicodeFormat(domain: str) -> List[int]:
    defaultPrefix = "DEFAULT:"
    if not domain:
        domain = defaultPrefix
    else:
        domain = f"{defaultPrefix}:{domain}"

    chars = list(domain)
    langRanges = [
        {"range": (0xAC00, 0xD7AF), "prefix": 1_000_000},  # Korean
        {"range": (0x3040, 0x309F), "prefix": 2_000_000},  # Japanese Hiragana
        {"range": (0x30A0, 0x30FF), "prefix": 3_000_000},  # Japanese Katakana
        {"range": (0x4E00, 0x9FFF), "prefix": 4_000_000},  # Chinese
        {"range": (0x0410, 0x044F), "prefix": 5_000_000},  # Russian
        {"range": (0x0041, 0x007A), "prefix": 6_000_000},  # English (basic Latin)
        {"range": (0x0590, 0x05FF), "prefix": 7_000_000},  # Hebrew
        {"range": (0x00C0, 0x00FD), "prefix": 8_000_000},  # Vietnamese (approx)
        {"range": (0x0E00, 0x0E7F), "prefix": 9_000_000},  # Thai
    ]

    out: List[int] = []
    for ch in chars:
        code = ord(ch)
        prefix = 0
        for lr in langRanges:
            a, b = lr["range"]
            if a <= code <= b:
                prefix = lr["prefix"]
                break
        out.append(prefix + code)
    return out


def calculateSimilarity(word1: str, word2: str) -> float:
    stageLevel = 1.0
    arrs1 = wordNbUnicodeFormat(word1)
    nbMax = BIT_MAX_NB(arrs1)
    nbMin = BIT_MIN_NB(arrs1)

    arrs2 = wordNbUnicodeFormat(word2)
    max_v = BIT_MAX_NB(arrs2)
    min_v = BIT_MIN_NB(arrs2)

    similarity1 = wordSim(nbMax, nbMin, max_v, min_v)
    similarity2 = calculateArraySimilarity(arrs1, arrs2)

    if areLanguagesSame(word1, word2):
        return max(similarity1, similarity2) * stageLevel
    else:
        return min(similarity1, similarity2) / stageLevel


def calculateSimilarity2(maxValue: float, minValue: float, firstWord: str, secondWord: str) -> Dict[str, float]:
    stageLevel = 1.0

    unicodeArray1 = wordNbUnicodeFormat(firstWord)
    unicodeArray2 = wordNbUnicodeFormat(secondWord)

    maxBitValue = BIT_MAX_NB(unicodeArray2)
    minBitValue = BIT_MIN_NB(unicodeArray2)

    similarityBasedOnValues = wordSim(maxValue, minValue, maxBitValue, minBitValue)
    similarityBasedOnArrays = calculateArraySimilarity(unicodeArray1, unicodeArray2)

    if areLanguagesSame(firstWord, secondWord):
        finalSimilarity = max(similarityBasedOnValues, similarityBasedOnArrays) * stageLevel
    else:
        finalSimilarity = min(similarityBasedOnValues, similarityBasedOnArrays) / stageLevel

    return {
        "finalSimilarity": finalSimilarity,
        "maxValue": maxValue,
        "minValue": minValue,
        "maxBitValue": maxBitValue,
        "minBitValue": minBitValue,
    }


def identifyLanguage(s: str) -> str:
    unicodeArray = list(s)
    languageCounts = {
        "Japanese": 0.0,
        "Korean": 0.0,
        "English": 0.0,
        "Russian": 0.0,
        "Chinese": 0.0,
        "Hebrew": 0.0,
        "Vietnamese": 0.0,
        "Thai": 0.0,
        "Portuguese": 0.0,
        "Others": 0.0,
    }

    portugueseChars = {
        0x00C0, 0x00C1, 0x00C2, 0x00C3, 0x00C7, 0x00C8, 0x00C9, 0x00CA, 0x00CB, 0x00CC, 0x00CD, 0x00CE,
        0x00CF, 0x00D2, 0x00D3, 0x00D4, 0x00D5, 0x00D9, 0x00DA, 0x00DB, 0x00DC, 0x00DD, 0x00E0, 0x00E1,
        0x00E2, 0x00E3, 0x00E7, 0x00E8, 0x00E9, 0x00EA, 0x00EB, 0x00EC, 0x00ED, 0x00EE, 0x00EF, 0x00F2,
        0x00F3, 0x00F4, 0x00F5, 0x00F9, 0x00FA, 0x00FB, 0x00FC, 0x00FD, 0x0107, 0x0113, 0x012B, 0x014C,
        0x016B, 0x1ECD, 0x1ECF, 0x1ED1, 0x1ED3, 0x1ED5, 0x1ED7, 0x1ED9, 0x1EDB, 0x1EDD, 0x1EDF, 0x1EE1,
        0x1EE3, 0x1EE5, 0x1EE7, 0x1EE9, 0x1EEB, 0x1EED, 0x1EEF, 0x1EF1,
    }

    for ch in unicodeArray:
        code = ord(ch)
        if code in portugueseChars:
            languageCounts["Portuguese"] += 1
            languageCounts["Portuguese"] *= 10
        elif 0xAC00 <= code <= 0xD7AF:
            languageCounts["Korean"] += 1
            languageCounts["Korean"] *= 100
        elif (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF):
            languageCounts["Japanese"] += 1
            languageCounts["Japanese"] *= 10
        elif 0x4E00 <= code <= 0x9FFF:
            languageCounts["Chinese"] += 1
        elif (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A):
            languageCounts["English"] += 1
        elif (0x00C0 <= code <= 0x00FF) or (0x0102 <= code <= 0x01B0):
            languageCounts["Vietnamese"] += 1
            languageCounts["Vietnamese"] *= 10
        elif 0x0410 <= code <= 0x044F:
            languageCounts["Russian"] += 1
            languageCounts["Russian"] *= 10
        elif 0x0590 <= code <= 0x05FF:
            languageCounts["Hebrew"] += 1
            languageCounts["Hebrew"] *= 10
        elif 0x0E00 <= code <= 0x0E7F:
            languageCounts["Thai"] += 1
            languageCounts["Thai"] *= 10
        else:
            languageCounts["Others"] += 1

    totalCharacters = sum(languageCounts.values())
    if totalCharacters <= 0:
        return "None"
    languageRatios = {k: (v / totalCharacters) for k, v in languageCounts.items()}
    sortedLanguages = sorted(languageRatios.items(), key=lambda x: x[1], reverse=True)
    identifiedLanguage, maxRatio = sortedLanguages[0]
    if identifiedLanguage == "Others" or maxRatio == 0:
        if len(sortedLanguages) > 1:
            secondLanguage, secondRatio = sortedLanguages[1]
            return "None" if secondRatio == 0 else secondLanguage
        return "None"
    return identifiedLanguage


def calculateSentenceBits(sentence: str) -> Dict[str, float]:
    unicodeArray = wordNbUnicodeFormat(sentence)
    bitMax = BIT_MAX_NB(unicodeArray)
    bitMin = BIT_MIN_NB(unicodeArray)
    return {"bitMax": bitMax, "bitMin": bitMin}


def removeSpecialCharsAndSpaces(input_str: str | None) -> str:
    if input_str is None:
        return ""
    normalizedSpaces = re.sub(r"\s+", " ", input_str)
    # Keep alnum, Hangul, spaces, brackets, '#'
    return re.sub(r"[^0-9A-Za-z\uAC00-\uD7A3\s\[\]#]", "", normalizedSpaces).strip()


def cosineSimilarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


# Convenience: convert complex CPU output to a normalized NB key in [0,1]
def toNBKeyFromCPU(output: Iterable[float] | float, decimals: int = 8) -> float:
    if isinstance(output, (int, float)):
        mean_cpu = float(output)
        nb = max(0.0, min(1.0, mean_cpu / 100.0))
        return float(f"{nb:.{decimals}f}")
    xs = list(output)
    if not xs:
        return 0.0
    # Default: mean-based
    mean_cpu = sum(float(x) for x in xs) / len(xs)
    nb = max(0.0, min(1.0, mean_cpu / 100.0))
    return float(f"{nb:.{decimals}f}")


# Vector NB pipeline per redefined spec: series -> NB vector -> quantize -> key
def compute_nb_from_series(
    series: List[float],
    vec_len: int = 16,
    quant_decimals: int = 5,
    ns: str = "nbvec",
    ver: int = 1,
    eps: float = 1e-9,
    norm_mode: str = "bit",  # "bit" | "minmax"
) -> Tuple[List[float], List[float], str, float]:
    """
    Returns (nb, nb_q, key, bucket_float)
    - nb: normalized vector in [0,1]^N
    - nb_q: quantized vector (round to quant_decimals)
    - key: NB:{ns}:{ver}:{hash16}
    - bucket_float: mean(nb) rounded to 8 decimals (for folder bucketing)
    """
    if not series or len(series) < vec_len:
        raise ValueError(f"need at least {vec_len} samples, got {len(series) if series else 0}")
    x = [float(v) for v in series[-vec_len:]]
    # Normalization range selection
    nb: List[float] = []
    if norm_mode == "bit":
        try:
            # Use BIT-based dynamic range in 0..100
            bit_max = BIT_MAX_NB(x)  # 0..100
            bit_min = BIT_MIN_NB(x)  # 0..100
            mn = float(bit_min)
            mx = float(bit_max)
        except Exception:
            mn = min(x)
            mx = max(x)
    else:
        mn = min(x)
        mx = max(x)

    rng = mx - mn
    if rng <= eps:
        nb = [0.5 for _ in x]
    else:
        for xi in x:
            v = (xi - mn) / (rng + eps)
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            nb.append(v)
    nb_q = [float(f"{v:.{quant_decimals}f}") for v in nb]
    import json as _json
    payload = {"ns": ns, "ver": int(ver), "nb": nb_q}
    cano = _json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    h = hashlib.sha256(cano.encode("utf-8")).hexdigest()[:16]
    key = f"NB:{ns}:{ver}:{h}"
    bucket_float = float(f"{(sum(nb) / len(nb)):.8f}")
    return nb, nb_q, key, bucket_float


