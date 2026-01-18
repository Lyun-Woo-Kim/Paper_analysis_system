# 논문 분석 도구
## 현재 진행 사항
PDF 논문에서 이미지 요소(Table, Figure, Picture, Chart)와 수식을 자동 추출 과정까지 진행행.

## 주요 기능

- **이미지 요소 추출**: Surya OCR을 사용하여 PDF에서 Table, Figure, Picture, Chart를 자동으로 감지하고 추출
- **캡션 매칭**: 추출된 이미지 요소와 가장 가까운 캡션을 자동으로 매칭
- **이미지 분석**: Vision-Language 모델을 사용하여 추출된 이미지의 내용을 상세 분석
- **수식 추출**: PDF 페이지별로 수식을 LaTeX 형식으로 추출

## 실행 환경경

- Python 3.10.12
- CUDA 12.4
- NVIDIA GPU (권장)

## 설치 방법

### 1. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

**참고**: CUDA 12.4용 PyTorch는 별도로 설치해야 할 수 있습니다.
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. 설정 파일 확인

`model_config.yaml` 파일을 열어서 모델 이름과 포트 설정을 확인하세요.

```yaml
VL_MODEL_NAME: "Qwen/Qwen3-VL-8B-Instruct"
VL_HOST: "localhost"
VL_PORT: 8000
VL_CUDA_DEVICE: 0
```

필요에 따라 모델 이름, 호스트, 포트, GPU 디바이스를 변경할 수 있습니다.

## 사용 방법

### 1단계: Vision-Language 모델 서버 실행

먼저 `vl_app.py`를 실행하여 Vision-Language 모델 서버를 시작해야 합니다.

```bash
cd llm_load
python vl_app.py
```

서버가 정상적으로 시작되면 다음과 같은 메시지가 표시됩니다:
```
Server starting on port 8000...
```

**참고**: 
- 모델이 처음 로드될 때는 시간이 걸릴 수 있습니다
- 서버가 실행 중인 상태로 유지되어야 합니다 (별도 터미널에서 실행하거나 백그라운드로 실행)

### 2단계: PDF 파일 준비

분석할 PDF 파일을 `sample_paper.pdf`로 저장하거나, `example_paper_load.py`를 사용하여 예제 논문을 다운로드할 수 있습니다.

```bash
python example_paper_load.py
```

### 3단계: 이미지 및 수식 추출 실행

`main.py`를 실행하여 PDF에서 이미지 요소와 수식을 추출합니다.

```bash
python main.py
```

실행 과정:
1. **이미지 요소 추출**: Surya OCR을 사용하여 PDF에서 Table, Figure, Picture, Chart를 감지하고 추출
2. **이미지 분석**: 추출된 각 이미지를 Vision-Language 모델로 분석
3. **수식 추출**: PDF의 각 페이지에서 수식을 LaTeX 형식으로 추출
4. **모델 언로드**: 작업 완료 후 GPU 메모리를 해제

## 출력 결과

### 이미지 요소 추출 결과

- **위치**: `output_img/` 디렉토리
- **파일 형식**: 
  - `p{페이지번호}_{레이블}_{인덱스}.png`: 추출된 이미지
  - `p{페이지번호}_{레이블}_{인덱스}.jsonl`: 이미지 메타데이터 (페이지 번호, 레이블, 캡션, 바운딩 박스)

### 이미지 분석 결과

- **위치**: `output/img_analysis.jsonl`
- **내용**: 각 이미지에 대한 상세 분석 결과 (요약, 주요 포인트, 숫자 정보 등)

### 수식 추출 결과

- **위치**: `output_formula/results.json`
- **내용**: 페이지별로 추출된 수식 정보
  - LaTeX 형식의 수식
  - 수식 태그 (있는 경우)
  - 바운딩 박스 좌표
  - 신뢰도 점수

## 프로젝트 구조

```
paper_analysis/
├── main.py                 # 메인 실행 파일
├── model_config.yaml       # 모델 설정 파일
├── requirements.txt        # 의존성 패키지 목록
├── example_paper_load.py   # 예제 논문 다운로드 스크립트
├── llm_load/
│   ├── vl_app.py          # Vision-Language 모델 Flask 서버
│   └── vl_engine.py       # Vision-Language 모델 엔진
├── utils/
│   ├── visual_extractor.py # 이미지 요소 추출 모듈
│   ├── math_extractor.py   # 수식 추출 모듈
│   └── utils.py            # 유틸리티 함수
├── output/                 # 이미지 분석 결과
├── output_img/             # 추출된 이미지 파일
└── output_formula/         # 수식 추출 결과
```

## 문제 해결

### 모델 서버 연결 오류

- `vl_app.py`가 실행 중인지 확인하세요
- `model_config.yaml`의 `VL_HOST`와 `VL_PORT` 설정이 올바른지 확인하세요
- 방화벽이나 포트 충돌이 없는지 확인하세요

### GPU 메모리 부족

- `model_config.yaml`에서 `VL_CUDA_DEVICE`를 다른 GPU로 변경하세요
- 다른 GPU 프로세스를 종료하세요
- 모델 크기를 줄이거나 양자화된 모델을 사용하세요

### 모델 다운로드 오류

- Hugging Face 토큰이 필요한 경우 환경 변수에 설정하세요:
  ```bash
  export HF_TOKEN=your_token_here
  ```
- 인터넷 연결을 확인하세요

## 참고사항

- 첫 실행 시 모델이 자동으로 다운로드되므로 시간이 걸릴 수 있습니다
- 대용량 PDF 파일의 경우 처리 시간이 오래 걸릴 수 있습니다
- GPU 메모리가 부족한 경우 CPU 모드로 실행할 수 있지만 속도가 느려집니다

