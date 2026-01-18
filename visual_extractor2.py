import fitz
import io
import os
import base64
import json
import requests
from PIL import Image

# 1. 시각 요소 추출을 위한 강화된 프롬프트
VL_VISUAL_PROMPT = r"""
Extract all Figures, Tables, Charts, and Pictures. 
[CRITICAL] If a caption exists, the 'bbox' MUST include BOTH the object and its caption.
Return JSON:
{{
  "page_index": {page_index},
  "visual_elements": [
    {{
      "type": "figure"|"table"|"chart"|"picture",
      "caption": "text",
      "bbox": [x0, y0, x1, y1],
      "description": "A detailed 1-2 sentence summary of this element"
    }}
  ]
}}
"""

class VisualElementExtractor:
    def __init__(self, out_dir, img_dir, config_path):
        # 사용자 설정 경로 반영
        from utils.utils import load_config
        self.configs = load_config(config_path)
        self.url = f"http://{self.configs['VL_HOST']}:{self.configs['VL_PORT']}/analyze"
        
        self.out_dir = out_dir
        self.img_dir = img_dir
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)

    def _pdf_page_to_base64(self, page):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _save_cropped_element(self, page, bbox, filename):
        """bbox 영역(캡션 포함)을 고해상도로 잘라 저장"""
        try:
            rect = fitz.Rect(bbox)
            # 3배 해상도로 캡처하여 저장
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=rect)
            path = os.path.join(self.img_dir, filename)
            pix.save(path)
            return path
        except:
            return None

    def process_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        print(f">> Processing: {pdf_path}")

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            print(f"   [Page {page_idx+1}] Analyzing...")

            # VLM 요청 전송
            base64_img = self._pdf_page_to_base64(page)
            payload = {
                "messages": [{"role": "user", "content": [
                    {"type": "image", "image": f"data:image/png;base64,{base64_img}"},
                    {"type": "text", "text": VL_VISUAL_PROMPT.format(page_index=page_idx)}
                ]}]
            }

            try:
                resp = requests.post(self.url, json=payload)
                data = resp.json()
                # JSON 파싱 (기존 _parse_json 로직 사용)
                content = self._parse_json(data.get("response", ""))
                
                # 시각 요소가 있다면 이미지 크롭 및 개별 저장
                if "visual_elements" in content:
                    for i, elem in enumerate(content["visual_elements"]):
                        fname = f"p{page_idx}_{elem['type']}_{i}.png"
                        saved_path = self._save_cropped_element(page, elem['bbox'], fname)
                        elem["local_path"] = saved_path # JSON에 경로 추가

                # *** 핵심: 분석 결과를 파일로 직접 저장 ***
                output_json_path = os.path.join(self.out_dir, f"res_p{page_idx}.json")
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(content, f, indent=2, ensure_ascii=False)
                
                print(f"      > Saved: {output_json_path}")

            except Exception as e:
                print(f"      ! Error on p{page_idx}: {e}")

    def _parse_json(self, text):
        try:
            clean_text = text.split("```json")[-1].split("```")[0].strip()
            return json.loads(clean_text)
        except:
            return {"visual_elements": []}