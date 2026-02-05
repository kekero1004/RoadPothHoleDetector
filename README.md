# Road Pothole Detector

## AI-based Road Pothole/Crack Detection & Maintenance Route Optimization Plugin

QGIS 플러그인으로, 드론/차량 영상을 기반으로 도로 포트홀 및 균열을 자동 탐지하고, 유지보수 최적 경로를 생성합니다.

---

## 주요 기능

### 1. AI 기반 자동 탐지 (Detection)
- 드론/차량 촬영 영상 또는 정사영상 분석
- YOLOv7 기반 객체 탐지 (시뮬레이션 모드)
- 포트홀, 거북등 균열, 선형 균열 자동 식별
- 탐지된 결손 위치를 QGIS 포인트 레이어로 자동 변환

### 2. 위험도 평가 (Risk Assessment)
- 5단계 심각도 등급 (A~E)
  - Grade A: 경미
  - Grade B: 주의
  - Grade C: 보통
  - Grade D: 심각
  - Grade E: 위험
- 결손 유형별/심각도별 통계
- 테이블 기반 상세 조회 및 필터링

### 3. 유지보수 경로 최적화 (Route Optimization)
- TSP(외판원 문제) 알고리즘 기반 최적 순회 경로 생성
- 우선순위 기반 경로 (D, E 등급 우선 방문)
- 최근접 이웃(Nearest Neighbor) 휴리스틱
- GPX 및 Shapefile 내보내기 지원

### 4. LCC 분석 (Life Cycle Cost Analysis)
- 즉시 보수 vs 지연 보수 시나리오 비용 비교
- 인플레이션율/할인율 적용 NPV 계산
- 최적 개입 시점 제안

### 5. 작업 지시서 생성 (Work Order Report)
- 결손 목록, 위치, 예상 비용 포함
- 최적 경로 순서대로 작업 지시
- 텍스트/HTML 형식 내보내기

---

## 시스템 요구사항

- **QGIS**: 3.36 이상
- **Python**: 3.9+ (QGIS 내장 Python 사용)
- **OS**: Windows, Linux
- **추가 라이브러리**: 없음 (PyQt5만 사용)

---

## 설치 방법

### ZIP 파일로 설치

1. `RoadPotholeDetector` 폴더를 ZIP 파일로 압축합니다.
   - 포함 파일: `__init__.py`, `main.py`, `metadata.txt`, `icon.png`, `README.md`

2. QGIS를 실행합니다.

3. 메뉴에서 **플러그인 > 플러그인 관리 및 설치...** 선택

4. **ZIP에서 설치** 탭 선택

5. **...** 버튼을 클릭하여 압축한 ZIP 파일 선택

6. **플러그인 설치** 버튼 클릭

7. 설치 완료 후 **설치됨** 탭에서 "Road Pothole Detector" 체크하여 활성화

8. 툴바에 플러그인 아이콘이 표시됩니다.

---

## 사용 방법

### 1단계: 탐지 실행

1. 툴바의 **Road Pothole Detector** 아이콘 클릭
2. **Detection** 탭에서 설정:
   - 중심 좌표 (경도/위도) 입력
   - 탐지 반경 설정
   - 시뮬레이션 결손 개수 지정
   - 신뢰도 임계값 설정
3. **Run Detection** 버튼 클릭

### 2단계: 위험도 평가

1. **Risk Assessment** 탭으로 이동
2. 탐지된 결손 통계 확인
3. 테이블에서 결손 상세 정보 조회
4. 필터로 심각도/유형별 필터링
5. 선택한 결손으로 지도 줌

### 3단계: 경로 최적화

1. **Route Optimization** 탭으로 이동
2. 알고리즘 및 옵션 설정
3. **Generate Optimal Route** 클릭
4. 결과 확인 및 GPX/Shapefile 내보내기

### 4단계: LCC 분석

1. **LCC Analysis** 탭으로 이동
2. 분석 기간, 인플레이션율, 할인율 설정
3. **Run LCC Analysis** 클릭
4. 즉시 보수 vs 지연 보수 비용 비교 결과 확인

### 5단계: 보고서 생성

1. **Report** 탭으로 이동
2. 포함할 항목 선택
3. 저장 경로 지정
4. **Generate Work Order Report** 클릭

---

## 파일 구조

```
RoadPotholeDetector/
├── __init__.py      # 플러그인 초기화
├── main.py          # 메인 로직 및 GUI
├── metadata.txt     # 플러그인 메타데이터
├── icon.png         # 툴바 아이콘
└── README.md        # 문서
```

---

## 기술 스택

- **PyQGIS**: QGIS Python API
- **PyQt5**: GUI 프레임워크
- **TSP Algorithm**: 최근접 이웃 휴리스틱
- **LCC Analysis**: NPV 기반 생애주기비용 분석

---

## 확장 계획

- YOLOv7/v8 실제 모델 통합
- LSTM/XGBoost 열화도 예측
- NetworkX/OR-Tools 기반 고급 경로 최적화
- 변화 탐지(Change Detection) 모니터링
- PDF 보고서 생성

---

## 라이선스

MIT License

---

## 문의

- **이슈 리포트**: [GitHub Issues](https://github.com/example/road-pothole-detector/issues)
- **이메일**: developer@example.com

---

## 버전 이력

### v1.0.0 (2024)
- 초기 릴리즈
- AI 탐지 시뮬레이션
- TSP 기반 경로 최적화
- LCC 분석
- 작업 지시서 생성
