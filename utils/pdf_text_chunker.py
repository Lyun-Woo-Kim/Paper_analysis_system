from __future__ import annotations

from typing import List, Dict, Tuple
import os
import sys
import re
import fitz

from utils.utils import load_from_jsonl, load_config

# ---------------------------------------------------------------------
# Path 설정
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, "..")
sys.path.append(root_dir)

# ---------------------------------------------------------------------
# Geometry util
# ---------------------------------------------------------------------
def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / max(1e-9, (area_a + area_b - inter))

# ---------------------------------------------------------------------
# Visual bbox 로드 (이미지/표/그림 등)
# ---------------------------------------------------------------------
def load_visual_bboxes_by_page(
    visual_jsonl_records: List[dict],
    scale_factor: float,
) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """
    visual_jsonl_records: 사용자가 이미 만든 jsonl list[dict]
      예: {"page_index":2,"label":"Figure","bbox":[395,142,828,792], ...}

    - bbox가 이미 렌더(scale 적용) 좌표면 scale_factor=1.0
    - bbox가 PDF 원본 좌표면, 텍스트 bbox에 곱할 scale_factor와 동일 값 사용
    """
    out: Dict[int, List[Tuple[float, float, float, float]]] = {}
    for r in visual_jsonl_records:
        p = int(r["page_index"])
        b = r["bbox"]  # [x0,y0,x1,y1]
        sb = (b[0] * scale_factor, b[1] * scale_factor, b[2] * scale_factor, b[3] * scale_factor)
        out.setdefault(p, []).append(sb)
    return out

# 텍스트 블록만 추출하되, Visual bbox와 겹치는 블록은 제외
# (여기서는 bbox를 반환하지 않고 "정렬된 텍스트 블록 문자열"만 반환)
def extract_text_blocks_text_only_excluding_visuals(
    pdf_path: str,
    visual_bboxes_by_page: Dict[int, List[Tuple[float, float, float, float]]],
    scale_factor: float = 2.0,
    iou_threshold: float = 0.10,
) -> List[str]:
    """
    반환: visual과 겹치지 않는 텍스트 블록들의 리스트(읽기 순서로 정렬)
    - 내부적으로만 bbox를 사용해서 overlap 여부를 판단합니다.
    - 최종 결과는 text만 반환합니다.
    """
    doc = fitz.open(pdf_path)
    kept = []

    for page_index, page in enumerate(doc):
        raw_blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text, block_no, block_type)
        visuals = visual_bboxes_by_page.get(page_index, [])

        page_text_blocks = []
        for b in raw_blocks:
            if b[6] != 0:
                continue  # text only

            text = re.sub(r"\s+", " ", (b[4] or "")).strip()
            if not text:
                continue

            # visual bbox와 같은 좌표계로 맞추기 위해 scale 적용
            bbox = (b[0] * scale_factor, b[1] * scale_factor, b[2] * scale_factor, b[3] * scale_factor)

            # visual bbox와 겹치면 제외 (표 내부 텍스트 등)
            if visuals:
                if any(_iou(bbox, vb) >= iou_threshold for vb in visuals):
                    continue

            # 정렬을 위해 bbox의 y,x를 함께 들고 있다가 최종적으로 text만 반환
            page_text_blocks.append((bbox[1], bbox[0], text))

        # 페이지 내 읽기 순서 정렬
        page_text_blocks.sort(key=lambda t: (t[0], t[1]))
        # bbox 정보는 사용하지 않고, text만 반환
        kept.extend([t[2] for t in page_text_blocks])

    return kept

# 순수 텍스트 청킹 (chunk_size / overlap_size)
def chunk_text_blocks(
    text_blocks: List[str],
    chunk_size: int = 1000,
    overlap_size: int = 150,
) -> List[dict]:
    """
    입력: text_blocks (이미 visual 제외된 텍스트 블록 리스트)
    출력: [{"text": "..."} , ...] 형태의 청크 리스트

    - overlap_size는 "이전 청크의 끝부분 텍스트"를 다음 청크 앞에 붙입니다.
    - bbox는 사용/저장하지 않습니다.
    """
    chunks: List[dict] = []
    cur = ""

    def flush():
        nonlocal cur
        if not cur.strip():
            cur = ""
            return

        chunks.append({"text": cur.strip()})

        if overlap_size > 0 and len(cur) > overlap_size:
            cur = cur[-overlap_size:]  # tail 유지
        else:
            cur = ""

    for t in text_blocks:
        if not t:
            continue

        # 블록 간 공백 하나로 연결
        if cur:
            candidate = cur + " " + t
        else:
            candidate = t

        if len(candidate) >= chunk_size:
            cur = candidate
            flush()
        else:
            cur = candidate

    if cur.strip():
        chunks.append({"text": cur.strip()})

    return chunks