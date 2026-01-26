from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from utils.utils import load_from_jsonl, load_config
import fitz
import sys
import os
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..')
sys.path.append(root_dir)

@dataclass
class TextBlock:
    page_index: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1) in rendered-scale coords
    text: str

def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / max(1e-9, (area_a + area_b - inter))

def load_visual_bboxes_by_page(visual_jsonl_records: List[dict], scale_factor: float) -> Dict[int, List[Tuple[int,int,int,int]]]:
    """
    visual_jsonl_records: 사용자가 이미 만든 jsonl들을 합쳐 읽어온 list[dict]
      예: {"page_index":2,"label":"Figure","bbox":[395,142,828,792], ...}
    bbox가 이미 scale_factor가 적용된 좌표면 scale_factor=1로 두시면 됩니다.
    """
    out: Dict[int, List[Tuple[int,int,int,int]]] = {}
    for r in visual_jsonl_records:
        p = int(r["page_index"])
        b = r["bbox"]
        sb = (b[0]*scale_factor, b[1]*scale_factor, b[2]*scale_factor, b[3]*scale_factor)
        out.setdefault(p, []).append(sb)
    return out

def extract_text_blocks_excluding_visuals(
    pdf_path: str,
    visual_bboxes_by_page: Dict[int, List[Tuple[float,float,float,float]]],
    scale_factor: float = 2.0,
    iou_th: float = 0.10,
) -> List[TextBlock]:
    doc = fitz.open(pdf_path)
    blocks: List[TextBlock] = []

    for page_index, page in enumerate(doc):
        # get_text("blocks")는 PDF 좌표 기준이므로, 렌더 scale과 맞추기 위해 동일 scale_factor 적용
        raw_blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text, block_no, block_type)
        visuals = visual_bboxes_by_page.get(page_index, [])

        for b in raw_blocks:
            if b[6] != 0:
                continue  # text only
            text = re.sub(r"\s+", " ", (b[4] or "")).strip()
            if not text:
                continue
            bbox = (b[0]*scale_factor, b[1]*scale_factor, b[2]*scale_factor, b[3]*scale_factor)

            # 이미지 영역과 겹치면 제외 (표 내부 텍스트 등)
            if visuals:
                overlapped = any(_iou(bbox, vb) >= iou_th for vb in visuals)
                if overlapped:
                    continue

            blocks.append(TextBlock(page_index=page_index, bbox=bbox, text=text))

    # 읽기 순서 정렬 (대략 y -> x)
    blocks.sort(key=lambda x: (x.page_index, x.bbox[1], x.bbox[0]))
    return blocks

# Chunking 함수. chunk size와 overlap size를 조절 가능.
def merge_blocks_to_chunks(
    blocks: List[TextBlock],
    target_chars: int = 1000,
    overlap_chars: int = 150,
) -> List[dict]:
    chunks: List[dict] = []
    cur_text = []
    cur_bboxes = []
    cur_page: Optional[int] = None

    def flush():
        nonlocal cur_text, cur_bboxes, cur_page
        if not cur_text:
            return
        text = " ".join(cur_text).strip()
        x0 = min(b[0] for b in cur_bboxes)
        y0 = min(b[1] for b in cur_bboxes)
        x1 = max(b[2] for b in cur_bboxes)
        y1 = max(b[3] for b in cur_bboxes)
        chunks.append({"text": text, "page_index": cur_page, "bbox": [x0,y0,x1,y1]})
        # overlap 구현: 끝부분 overlap_chars 만큼만 유지
        if overlap_chars > 0 and len(text) > overlap_chars:
            tail = text[-overlap_chars:]
            cur_text = [tail]
            cur_bboxes = []  # bbox는 새로 시작(겹침은 텍스트 연결용)
        else:
            cur_text, cur_bboxes = [], []
        # cur_page는 유지 (같은 페이지에서 계속)

    for b in blocks:
        if cur_page is None:
            cur_page = b.page_index

        # 페이지가 바뀌면 flush (페이지 섞인 chunk는 bbox 의미가 애매해집니다)
        if b.page_index != cur_page:
            flush()
            cur_page = b.page_index
            cur_text, cur_bboxes = [], []

        cur_text.append(b.text)
        cur_bboxes.append(b.bbox)

        if sum(len(t) for t in cur_text) >= target_chars:
            flush()

    flush()
    return chunks