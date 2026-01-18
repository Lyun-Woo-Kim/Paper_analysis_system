from .utils import load_config

VL_FORMULA_PROMPT = r"""
You are an expert in OCR and mathematical document analysis.
Your task is to identify and extract ALL mathematical formulas (equations) from the provided image.

[CRITICAL RULE]
- First, check if any mathematical formulas exist in the image.
- If NO formulas are present, return EXACTLY: {{"page_index": {page_index}, "formulas": []}}
- If formulas exist, extract them according to the following JSON schema:

{{
  "page_index": {page_index},
  "formulas": [
    {{
      "latex": "LaTeX string",         // Use \\frac, \\sum, \\begin{{aligned}}, etc.
      "tag": "normalized tag/null",    // e.g., "(3)" -> "3"
      "tag_raw": "raw tag/null",       // e.g., "Eq. (3.1)"
      "bbox": [x0, y0, x1, y1],        // Image pixel coordinates
      "confidence": 0.0 ~ 1.0
    }}
  ]
}}

[Instructions]
1) Extract ONLY equations. Do not include plain text or figure captions in the "latex" field.
2) Ensure all backslashes in LaTeX are properly escaped for JSON (e.g., "\\\\frac{{a}}{{b}}").
3) Multi-line equations should use "\\begin{{aligned}} ... \\end{{aligned}}".

Caption for Context: {caption}
"""

import fitz  # PyMuPDF
import io
import os
import base64
import json
import requests
from PIL import Image
from tqdm import tqdm

class MathExtractor:
    def __init__(self, out_dir="assets_visual", config_path = 'model_config.yaml'):
        self.configs = load_config(config_path)
        self.url = f"http://{self.configs['VL_HOST']}:{self.configs['VL_PORT']}/analyze"
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def _pdf_page_to_base64(self, page, scale=2.0):
        """페이지를 고해상도(2x2) 이미지로 변환 후 Base64 인코딩"""
        # 해상도를 scale배만큼 높임 (Matrix(2, 2)는 가로세로 2배씩 확대)
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat)
        
        # Pixmap을 PIL 이미지로 변환
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # 메모리 내에서 Base64로 변환 (디스크 저장 없이 바로 전송)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def extract_formulas(self, pdf_path):
        """PDF 전체를 순회하며 수식 추출 수행"""
        doc = fitz.open(pdf_path)
        print(f">> Processing PDF: {pdf_path} ({len(doc)} pages)")
        
        results = []

        for page_idx in tqdm(range(len(doc)), desc = f"Extracting formulas from {pdf_path}"):
            page = doc[page_idx]
            # 1. 고해상도 이미지 생성 (Base64)
            base64_img = self._pdf_page_to_base64(page, scale=2.0)

            # 2. 기존 분석 데이터(Caption 등)가 있다면 로드
            # (Surya Layout 결과물 등이 out_dir에 저장되어 있다고 가정)
            caption = self._load_caption(page_idx)

            # 3. 프롬프트 구성
            current_prompt = VL_FORMULA_PROMPT.format(
                page_index=page_idx, 
                caption=caption
            )

            # 4. Flask 서버에 요청 전송
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"data:image/png;base64,{base64_img}"},
                            {"type": "text", "text": current_prompt}
                        ]
                    }
                ]
            }

            try:
                response = requests.post(self.url, json=payload)
                response.raise_for_status()
                data = response.json()
                
                # 결과 파싱 (정규식 등으로 JSON만 추출하는 로직 권장)
                extracted_json = self._parse_json(data.get("response", ""))
                results.append(extracted_json)
                
            except Exception as e:
                print(f"      ! Error on page {page_idx}: {e}")
                results.append({"page_index": page_idx, "formulas": [], "error": str(e)})
        with open(os.path.join(self.out_dir, "formula_analysis.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def _load_caption(self, page_idx):
        caption_path = os.path.join(self.out_dir, f"p{page_idx}.txt")
        if os.path.exists(caption_path):
            with open(caption_path, "r", encoding="utf-8") as f:
                return f.read()
        return "No specific layout context available."

    def _parse_json(self, text):
        """모델 응답에서 JSON 데이터 추출"""
        try:
            # ```json ... ``` 블록 제거
            clean_text = text.split("```json")[-1].split("```")[0].strip()
            return json.loads(clean_text)
        except:
            return {"raw_text": text, "status": "parse_error"}