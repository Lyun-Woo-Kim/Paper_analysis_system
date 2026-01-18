# llm_engine.py
import torch
import gc
import sys
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# 현재 파일이 있는 디렉토리의 상위 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

from utils.utils import load_config

class QwenLLMEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        # 현재 파일 기준으로 model_config.yaml 경로 설정
        config_path = root_dir / "model_config.yaml"
        self.model_id = load_config(str(config_path))["LLM_MODEL_NAME"]

    def is_loaded(self):
        """모델이 현재 메모리에 올라와 있는지 확인"""
        return self.model is not None

    def load_model(self):
        """모델과 토크나이저를 GPU에 로드"""
        if self.is_loaded():
            print("✅ Model is already loaded.")
            return

        print(f"🚀 Loading model: {self.model_id}...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            )
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise e

    def unload_model(self):
        """모델을 메모리에서 해제하고 GPU 캐시 정리 (핵심 기능)"""
        if not self.is_loaded():
            print("⚠️ Model is not loaded.")
            return

        print("♻️ Unloading model and clearing GPU memory...")
        
        # 객체 삭제
        del self.model
        del self.tokenizer
        
        # 참조 초기화
        self.model = None
        self.tokenizer = None
        
        # 가비지 컬렉션 및 CUDA 캐시 비우기
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        print("✅ GPU memory cleared!")

    def generate_response(self, messages):
        """텍스트 추론 수행"""
        if not self.is_loaded():
            # 편의를 위해 로드가 안 되어 있으면 자동으로 로드
            print("🔄 Model not loaded. Auto-loading...")
            self.load_model()

        # 1. 입력 전처리 (chat template 적용)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 2. 텐서 변환
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        # 3. 추론 (Inference)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024
            )

        # 4. 결과 디코딩
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text

# 싱글톤 인스턴스 생성
engine = QwenLLMEngine()

