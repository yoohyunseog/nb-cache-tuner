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

## 예시 데이터로 사용법
1) Interval(s), Window(s) 값을 설정합니다. (예: 1.0s / 60)
2) START를 눌러 샘플링을 시작합니다. 좌측 그래프는 NB_MAX/NB_MIN 시계열을, 우측 패널은 창 구간별 Top 10 NB_MAX/NB_MIN을 실시간 갱신합니다.
3) Rank/Spec 버튼을 누르면 CPU/GPU 스펙과(Windows의 경우) Top Processes(CPU) 목록을 함께 출력합니다.
4) Top 10 목록이 모두 채워지면 자동으로 [Finalized: auto END] 상태가 표시되며, END로 중지할 수도 있습니다.
5) Screenshot 버튼을 누르면 현재 대시보드 화면이 클립보드에 복사됩니다. 클립보드에 이미지가 있으면 docs/example_output.png로 저장해 README에 표시됩니다.

### 예시 출력 이미지
아래 이미지는 예시 데이터로 실행한 결과입니다. (이미지를 교체하려면 앱에서 Screenshot 클릭 후 다시 저장하면 됩니다.)

![예시 출력](docs/example_output.png)

### 결과 해설(예시 기준)
- Top 10 NB_MAX (high): 관측 창들 중 최고점이 큰 순서. 피크 타이밍(예: idx 143  1.98)을 빠르게 식별
- Top 10 NB_MIN (low): 관측 창들 중 최저점이 작은 순서. 바닥 타이밍 및 안정 구간 파악
- Elapsed(s): 측정 누적 시간. 동일 환경 비교 시 수집 길이 통제에 활용
- AVG NB_MAX / AVG NB_MIN: 창 구간별 NB 지표 평균. 장비/세션/설정 간 상대 비교 지표
- CPU/GPU/Top Processes: 측정 환경(하드웨어 스펙) 및 현재 부하 원인에 대한 보조 정보

> 팁: NB_MAX/NB_MIN의 상대적 크기와 변동폭을 비교하면 부하 패턴(스파이크/안정 구간)을 직관적으로 파악할 수 있습니다.
