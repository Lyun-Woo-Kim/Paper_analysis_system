import sys
from pathlib import Path

# 현재 파일(main.py)이 있는 디렉토리를 Python 경로에 추가
base_dir = Path(__file__).parent
sys.path.insert(0, str(base_dir))

from utils.math_extractor import MathExtractor
from utils.visual_extractor import SuryaLayoutExtractor
from example_paper_load import download_example_paper, download_paper
from utils.utils import load_config, save_to_jsonl, load_from_jsonl, build_instruction
import requests
from tqdm import tqdm
import json
import os

def analyze_image(image_path, caption, label, page_index): 
    global configs
    vl_port = configs["VL_PORT"]
    vl_host = configs['VL_HOST']
    url = f"http://{vl_host}:{vl_port}/analyze"
    
    instruction = build_instruction(caption, label)
    messages = [{
        "role": "user", 
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": instruction},
        ]
    }]
    
    response = requests.post(url, json={"messages": messages})
    output = json.loads(response.json()['response'])
    output['page_index'] = page_index
    save_to_jsonl(output, filename=f"{base_dir}/output/img_analysis.jsonl")
    
def unload_vl_model(): 
    global configs
    vl_port = configs["VL_PORT"]
    vl_host = configs['VL_HOST']
    url = f"http://{vl_host}:{vl_port}/unload"
    response = requests.post(url)
    return response.json()

if __name__ == "__main__":     
    # 현재 파일(main.py)이 있는 디렉토리를 기준으로 경로 설정
    pdf_path = str(base_dir / "sample_paper.pdf")
    formula_out_dir = str(base_dir / "output_formula")
    img_out_dir = str(base_dir / "output_img")
    config_path = str(base_dir / "model_config.yaml")
    configs = load_config(config_path)
    # Table, Figure, Picture, Chart [Caption 포함] 추출 후 저장.
    visual_extractor = SuryaLayoutExtractor(img_out_dir)
    visual_extractor.extract_visual_elements(pdf_path)
    
    img_infos = list(set([img_info.split(".")[0] for img_info in os.listdir(img_out_dir)]))
    for img_info in tqdm(img_infos, desc = f"Analyzing images from {pdf_path}"): 
        img_path = os.path.join(img_out_dir, f"{img_info}.png")
        img_info_path = os.path.join(img_out_dir, f"{img_info}.jsonl")
        img_infos = load_from_jsonl(img_info_path)
        analyze_image(img_path, img_infos[0]['caption'], img_infos[0]['label'], img_infos[0]['page_index'])
    
    # 수식 추출 후 저장.
    math_extractor = MathExtractor(out_dir=formula_out_dir,
                                   config_path = config_path)
    math_extractor.extract_formulas(pdf_path)
    
    unload_vl_model()
    
