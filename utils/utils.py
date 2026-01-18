from pathlib import Path
import yaml
import json

def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Resolved absolute path: {config_path.resolve()}\n"
            f"Current working directory: {Path.cwd()}"
        )
    cfg_text = config_path.read_text(encoding="utf-8")
    return yaml.safe_load(cfg_text)

def save_to_jsonl(data, filename):
    with open(filename, 'a', encoding='utf-8') as f:
        # 딕셔너리를 한 줄의 문자열로 변환 후 줄바꿈(\n) 추가
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
def load_from_jsonl(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]
    
def normalize_label(label: str) -> str:
    if not label:
        return ""
    return str(label).strip().lower()
    
    
# 아래는 이미지 상세 분석을 위한 프롬프트 관련 함수수
def build_prompt_table(caption: str) -> str:
    return f"""
Use caption as a hint, but prioritize what is actually visible in the image.
If unclear, write "unknown". Do NOT guess.
Do NOT dump the whole table.

Return ONLY valid JSON:
{{
  "label": "Table",
  "caption": "{caption}",
  "summary": "1-2 sentences",
  "key_points": ["max 3"],
  "evidence": ["visible header/cell snippets, max 8"],
  "headers": {{"columns": ["max 8"], "rows": ["max 8"]}},
  "best": {{"what":"unknown", "where":"unknown", "value":"unknown"}},
  "numbers": [{{"name":"...", "value":"...", "unit":"unknown", "context":"..."}}]  // max 5
}}
""".strip()

def build_prompt_figure(caption: str) -> str:
    return f"""
Use caption as a hint, but prioritize what is actually visible in the image.
If unclear, write "unknown". Do NOT guess.

This is a scientific Figure (could be plot or diagram). Focus on:
- the main claim/message
- any visible labels/legend text
- up to 5 key numbers if visible

Return ONLY valid JSON:
{{
  "label": "Figure",
  "caption": "{caption}",
  "summary": "1-2 sentences",
  "key_points": ["max 3"],
  "evidence": ["visible text snippets (title/labels/legend), max 8"],
  "numbers": [{{"name":"...", "value":"...", "unit":"unknown", "context":"..."}}]  // max 5
}}
""".strip()

def build_prompt_chart(caption: str) -> str:
    return f"""
Use caption as a hint, but prioritize what is actually visible in the image.
If unclear, write "unknown". Do NOT guess.

This is a Chart. Focus on:
- axes labels and units
- legend/series names
- overall trend (increasing/decreasing/peak/flat)
- up to 5 key numbers if visible (best/worst/peak/improvement)

Return ONLY valid JSON:
{{
  "label": "Chart",
  "caption": "{caption}",
  "summary": "1-2 sentences",
  "key_points": ["max 3"],
  "evidence": ["visible text snippets (axes/legend/title), max 8"],
  "axes": {{"x": "unknown", "y": "unknown"}},
  "legend": ["max 6"],
  "trend": "increasing|decreasing|peak|flat|unknown",
  "numbers": [{{"name":"...", "value":"...", "unit":"unknown", "context":"..."}}]  // max 5
}}
""".strip()

def build_prompt_picture(caption: str) -> str:
    return f"""
Use caption as a hint, but prioritize what is actually visible in the image.
If unclear, write "unknown". Do NOT guess.

This is a Picture (photo/sample/qualitative result). Focus on:
- what objects/regions are shown
- what comparison is being illustrated (if any)
- any visible labels text
Avoid overly detailed descriptions.

Return ONLY valid JSON:
{{
  "label": "Picture",
  "caption": "{caption}",
  "summary": "1-2 sentences",
  "key_points": ["max 3"],
  "evidence": ["visible text snippets, max 8"],
  "numbers": [{{"name":"...", "value":"...", "unit":"unknown", "context":"..."}}]  // max 5
}}
""".strip()

def build_prompt_unknown(caption: str, label: str = "") -> str:
    # label이 비어있거나, Surya가 못 준 경우용 (가장 안전한 최소 스키마)
    safe_label = label if (label and str(label).strip()) else "unknown"
    return f"""
Use caption and label as hints, but prioritize what is actually visible in the image.
If unclear, write "unknown". Do NOT guess.

Return ONLY valid JSON:
{{
  "label": "{safe_label}",
  "caption": "{caption}",
  "summary": "1-2 sentences",
  "key_points": ["max 3"],
  "evidence": ["visible text snippets, max 8"],
  "numbers": [{{"name":"...", "value":"...", "unit":"unknown", "context":"..."}}]  // max 5
}}
""".strip()

def build_instruction(caption: str, label: str | None) -> str:
    l = (label or "").strip()
    if l == "Table":
        return build_prompt_table(caption)
    if l == "Figure":
        return build_prompt_figure(caption)
    if l == "Chart":
        return build_prompt_chart(caption)
    if l == "Picture":
        return build_prompt_picture(caption)
    return build_prompt_unknown(caption, label=l)