# 논문 분석 도구

논문 PDF 파일을 분석하고 질문에 답변할 수 있는 RAG(Retrieval-Augmented Generation) 기반 도구입니다.

## 환경 요구사항

- **Python**: 3.10.12
- **CUDA**: 12.4
- **NVIDIA GPU** (권장)

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

### 1단계: 초기 VectorDB 구축

논문 분석을 시작하기 전에, 먼저 논문 PDF에서 데이터를 추출하고 VectorDB를 구축해야 합니다.

`build_vectorDB_main.py`를 실행하여 다음 작업을 수행합니다:
- PDF에서 이미지 요소(Table, Figure, Picture, Chart) 추출
- 이미지 분석
- 수식 추출 및 분석
- 텍스트 청킹
- Milvus VectorDB에 데이터 저장

```bash
python build_vectorDB_main.py
```

실행 과정:
1. **이미지 요소 추출**: Surya OCR을 사용하여 PDF에서 Table, Figure, Picture, Chart를 감지하고 추출
2. **이미지 분석**: 추출된 각 이미지를 Vision-Language 모델로 분석
3. **수식 추출 및 분석**: PDF의 각 페이지에서 수식을 LaTeX 형식으로 추출하고 분석
4. **텍스트 청킹**: PDF 텍스트를 청크로 분할
5. **VectorDB 저장**: 추출된 모든 데이터를 Milvus VectorDB에 저장

**참고**: 
- 이 과정은 시간이 오래 걸릴 수 있습니다 (논문 크기에 따라 다름)
- 모델이 처음 로드될 때는 추가 시간이 필요합니다
- VectorDB 구축은 논문당 한 번만 수행하면 됩니다

### 2단계: 논문 질문하기

VectorDB 구축이 완료되면, `main.py`를 실행하여 논문에 대해 질문할 수 있습니다.

```bash
python main.py
```

실행하면 다음과 같이 질문을 입력할 수 있습니다:

```
Enter your question: [여기에 질문을 입력하세요]
```

예시 질문:
- "What is the main idea of the paper?"
- "What methods are used in this paper?"
- "Explain the experimental results"
- "What are the key contributions?"

질문을 입력하면 RAG 시스템이 VectorDB에서 관련 정보를 검색하고 답변을 생성합니다.

종료하려면 `exit`를 입력하세요.

## 출력 결과

### VectorDB 구축 결과

`build_vectorDB_main.py` 실행 후 생성되는 파일들:

- **이미지 요소**: `output_img/{논문이름}/` 디렉토리
  - `p{페이지번호}_{레이블}_{인덱스}.png`: 추출된 이미지
  - `p{페이지번호}_{레이블}_{인덱스}.json`: 이미지 메타데이터

- **이미지 분석 결과**: `output/{논문이름}/img_analysis.jsonl`
  - 각 이미지에 대한 상세 분석 결과

- **수식 추출 결과**: `output_formula/{논문이름}/formula_analysis.json`
  - 페이지별로 추출된 수식 정보 및 분석 결과

- **VectorDB**: Milvus Lite 데이터베이스
  - 텍스트, 수식, 이미지 정보가 벡터로 저장됨

## 프로젝트 구조

```
paper_analysis/
├── main.py                 # 논문 질문 실행 파일
├── build_vectorDB_main.py  # VectorDB 구축 실행 파일
├── model_config.yaml       # 모델 설정 파일
├── requirements.txt        # 의존성 패키지 목록
├── example_paper_load.py   # 예제 논문 다운로드 스크립트
├── llm_load/
│   ├── rag.py             # RAG 시스템
│   ├── app_manager.py     # 모델 서버 관리
│   └── ...
├── utils/
│   ├── visual_extractor.py # 이미지 요소 추출 모듈
│   ├── math_extractor.py   # 수식 추출 모듈
│   ├── pdf_text_chunker.py # 텍스트 청킹 모듈
│   ├── milvus_stores.py    # Milvus VectorDB 관리
│   └── ...
├── output/                 # 이미지 분석 결과
├── output_img/             # 추출된 이미지 파일
└── output_formula/         # 수식 추출 결과
```

## 문제 해결

### 모델 서버 연결 오류

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

### VectorDB 관련 오류

- `build_vectorDB_main.py`가 정상적으로 완료되었는지 확인하세요
- `model_config.yaml`의 `PAPER_NAME`이 올바른지 확인하세요
- Milvus 데이터베이스 경로를 확인하세요

## 참고사항

- 첫 실행 시 모델이 자동으로 다운로드되므로 시간이 걸릴 수 있습니다
- 대용량 PDF 파일의 경우 VectorDB 구축 시간이 오래 걸릴 수 있습니다
- GPU 메모리가 부족한 경우 CPU 모드로 실행할 수 있지만 속도가 느려집니다
- VectorDB는 논문별로 구축되므로, 새로운 논문을 분석하려면 다시 `build_vectorDB_main.py`를 실행해야 합니다
