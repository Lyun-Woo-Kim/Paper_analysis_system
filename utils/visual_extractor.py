import fitz
from PIL import Image
import numpy as np
import math
import os, sys
from tqdm import tqdm

from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.settings import settings

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..')
sys.path.insert(0, root_dir)
from utils.utils import save_to_jsonl

class SuryaLayoutExtractor:
    def __init__(self, output_dir="assets_visual"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print(">> Loading Surya Layout Model...")
        # 모델 로딩 부분은 사용자분의 기존 코드를 따릅니다.
        foundation = FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
        self.layout_predictor = LayoutPredictor(foundation)

    def get_distance(self, box1, box2):
        """
        두 박스(x1, y1, x2, y2) 사이의 중심점 거리를 계산합니다.
        필요에 따라 '위/아래' 거리만 계산하도록 로직 수정 가능합니다.
        """
        c1_x, c1_y = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
        c2_x, c2_y = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
        return math.hypot(c1_x - c2_x, c1_y - c2_y)

    def is_overlapping(self, box1, box2, threshold=0.5):
        """
        두 박스가 겹치는지 확인합니다 (Intersection over Union 혹은 단순 교차).
        여기서는 단순 교차 여부만 봅니다.
        """
        x1_a, y1_a, x2_a, y2_a = box1
        x1_b, y1_b, x2_b, y2_b = box2

        # 겹치지 않는 경우
        if x2_a < x1_b or x2_b < x1_a or y2_a < y1_b or y2_b < y1_a:
            return False
        
        # 겹치는 영역 계산 (옵션)
        return True

    def extract_visual_elements(self, pdf_path):
        doc = fitz.open(pdf_path)
        print(f">> Processing {pdf_path}...")
        
        crop_padding = 5
        scale_factor = 2 # 이미지 크기에 따른 좌표 조정용 파라미터

        for page_idx, page in enumerate(doc):
            mat = fitz.Matrix(scale_factor, scale_factor)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            W, H = img.size

            # 1. 텍스트 블록(문장/문단) 추출 및 좌표 변환
            # page.get_text("blocks") returns: (x0, y0, x1, y1, "text", block_no, block_type)
            raw_blocks = page.get_text("blocks")
            text_blocks = []
            
            for b in raw_blocks:
                # block_type == 0 (Text), 1 (Image). 우리는 텍스트만 필요
                if b[6] == 0: 
                    scaled_box = (b[0]*scale_factor, b[1]*scale_factor, 
                                  b[2]*scale_factor, b[3]*scale_factor)
                    text_content = b[4].replace("\n", " ").strip()
                    if text_content: # 빈 텍스트 제외
                        text_blocks.append({"bbox": scaled_box, "text": text_content})

            # 2. Surya 레이아웃 예측
            pred = self.layout_predictor([img])[0]

            for i, item in enumerate(tqdm(pred.bboxes, desc = f"Extracting visual elements from {pdf_path}")):
                label = getattr(item, "label", None)
                if label not in {"Table", "Figure", "Picture", "Chart"}:
                    continue

                # BBox 추출 및 정제
                bbox = getattr(item, "bbox", None)
                if bbox is None:
                    # Polygon 처리 (기존 코드 유지)
                    poly = getattr(item, "polygon", None)
                    if poly:
                        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
                        bbox = (min(xs), min(ys), max(xs), max(ys))
                    else:
                        continue

                img_box = list(map(int, bbox)) # [x1, y1, x2, y2]

                # 3. 캡션 찾기 로직
                candidate_captions = []
                
                for block in text_blocks:
                    txt_box = block["bbox"]
                    
                    # (A) 이미지와 겹치는지 확인 -> 겹치면 캡션 아님 (표 안의 텍스트 등)
                    if self.is_overlapping(img_box, txt_box):
                        continue
                    
                    # (B) 거리 계산
                    dist = self.get_distance(img_box, txt_box)
                    candidate_captions.append((dist, block["text"], txt_box))

                # (C) 거리순 정렬 후 가장 가까운 텍스트 선택
                caption_text = ""
                if candidate_captions:
                    # 거리(dist) 기준 오름차순 정렬
                    candidate_captions.sort(key=lambda x: x[0])
                    
                    # 가장 가까운 텍스트 (보통 Figure는 아래, Table은 위에 캡션이 있음)
                    # 여기서는 단순히 가장 가까운 녀석을 가져옵니다.
                    closest = candidate_captions[0]
                    
                    # 거리가 너무 멀면(예: 300픽셀 이상) 캡션이 아닐 수 있음 -> threshold 설정 권장
                    if closest[0] < 300 * scale_factor: 
                        caption_text = closest[1]

                # 4. 이미지 크롭 및 저장 (기존 코드 개선)
                x1, y1, x2, y2 = img_box
                x1 = max(0, min(W, x1 - crop_padding))
                y1 = max(0, min(H, y1 - crop_padding))
                x2 = max(0, min(W, x2 + crop_padding))
                y2 = max(0, min(H, y2 + crop_padding))

                if x2 <= x1 or y2 <= y1: continue

                filename = f"p{page_idx}_{label}_{i}.png"
                crop_img = img.crop((x1, y1, x2, y2))
                crop_img.save(os.path.join(self.output_dir, filename))
                
                save_file = {
                    'page_index': page_idx, 
                    'label': label,
                    'caption': caption_text,
                    'bbox': img_box
                }
                
                save_to_jsonl(save_file, filename=f"{self.output_dir}/p{page_idx}_{label}_{i}.jsonl")
                