nb-cache-tuner

A lightweight real-time dashboard to visualize NB_MAX and NB_MIN scores over a rolling window using BIT helpers. It samples system activity (CPU by default) and shows:
- NB_MAX/NB_MIN time series (Matplotlib)
- Realtime Top 10 windows of NB ranks
- System specs (CPU/GPU) and Top Processes (Windows)

Features
- Dual Qt backend: PyQt5 preferred, falls back to PySide6 automatically
- Matplotlib embedding for smooth plotting
- psutil optional (CPU sampling). Falls back to a synthetic signal if unavailable
- Windows only: Top Processes section via PowerShell (best-effort)

Requirements
- Python 3.9+
- One of: PyQt5 OR PySide6
- matplotlib
- Optional: psutil (CPU sampling), torch (GPU name)

Quick start
- Install dependencies
  pip install PyQt5 matplotlib psutil
  # or
  pip install PySide6 matplotlib psutil
- Run
  python ui_main.py

Notes
- BIT helpers live in nb_bit.py and are imported as BIT_MAX_NB / BIT_MIN_NB
- On non-Windows systems, process list may be unavailable
- The app is defensive; most optional features fail gracefully

License
- TBD

## 예시 출력
아래 이미지는 예시 데이터로 실행한 화면입니다.

![예시 출력](docs/example_output.png)

### 이 결과로 알 수 있는 것
- Top 10 NB_MAX (high): 관측 구간별 최고점의 인덱스와 값  피크 타이밍 비교에 사용
- Top 10 NB_MIN (low): 관측 구간별 최저점의 인덱스와 값  바닥 타이밍 비교에 사용
- Elapsed(s): 세션 누적 시간  동일 조건에서 측정 길이 확인
- AVG NB_MAX / AVG NB_MIN: 창 구간별 NB 지표 평균  장비/세션 간 성능 비교 지표
- Top Processes (CPU): Windows에서 CPU 사용량 상위 10개 프로세스(가능할 때만)
- CPU / GPU: 측정 환경 스펙(예: CPU 모델명, GPU 모델명)

### 샘플 해석(첨부 이미지 기준)
- NB_MAX 최고값 ≈ 1.98 (@ idx 143)
- NB_MIN 최저값 ≈ 3.61 (@ idx 143)
- 평균값: AVG NB_MAX ≈ 1.44, AVG NB_MIN ≈ 4.15
- [Finalized: auto END] 표시는 Top 10 목록이 채워져 자동 종료되었음을 의미

> 참고: 실제 값은 측정 환경과 기간에 따라 달라지며, NB_MAX/NB_MIN의 상대적 크기와 변동성을 통해 부하 패턴을 비교할 수 있습니다.
