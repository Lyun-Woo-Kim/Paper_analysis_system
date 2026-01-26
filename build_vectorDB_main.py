import sys
from pathlib import Path

# 현재 파일(main.py)이 있는 디렉토리를 Python 경로에 추가
base_dir = Path(__file__).parent
sys.path.insert(0, str(base_dir))

from utils.math_extractor import MathExtractor
from utils.visual_extractor import SuryaLayoutExtractor
from example_paper_load import download_example_paper, download_paper
from utils.utils import load_config, save_to_jsonl, load_from_json, build_instruction, load_from_jsonl, parse_json_output, delete_think_tag
from utils.pdf_text_chunker import load_visual_bboxes_by_page, extract_text_blocks_text_only_excluding_visuals, chunk_text_blocks
# from utils.pdf_text_chunker_with_bbox import extract_text_blocks_excluding_visuals, merge_blocks_to_chunks
from utils.milvus_stores import PaperMilvusStoresLite
from llm_load.rag import PaperInfo_RAG
from llm_load.app_manager import AppManager
from utils.prompts import ANALYSIS_FORMULA_PROMPT, REFINE_JSON_PROMPT
import requests
from tqdm import tqdm
import json
import os
import re

def analyze_image(out_dir, image_path, caption, label, page_index, bbox): 
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
    output['bbox'] = bbox
    if os.path.exists(f"{out_dir}/img_analysis.jsonl"): 
        results = load_from_jsonl(f"{out_dir}/img_analysis.jsonl")
    else: 
        results = []
        
    if output not in results: 
        save_to_jsonl(output, filename=f"{out_dir}/img_analysis.jsonl")
    
def unload_vl_model(): 
    global configs
    vl_port = configs["VL_PORT"]
    vl_host = configs['VL_HOST']
    url = f"http://{vl_host}:{vl_port}/unload"
    response = requests.post(url)
    return response.json()

# 논문 분석을 위한 초기 데이터 추출 단계.
def extract_all_elements(pdf_path, formula_out_dir, img_out_dir, out_dir):
    global config_path
    if not os.path.exists(img_out_dir): 
        os.makedirs(img_out_dir)
    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)
    if not os.path.exists(formula_out_dir): 
        os.makedirs(formula_out_dir)
    
    # Table, Figure, Picture, Chart [Caption 포함] 추출 후 저장. [시간이 오래 걸리지 않음]    
    visual_extractor = SuryaLayoutExtractor(img_out_dir)
    visual_extractor.extract_visual_elements(pdf_path)
    
    # visual element 정보에 대한 분석 (시간이 오래 걸림)
    if not os.path.exists(out_dir + "/img_analysis.jsonl"): 
        img_infos = list(set([img_info.split(".")[0] for img_info in os.listdir(img_out_dir)]))
        for img_info in tqdm(img_infos, desc = f"Analyzing images from {pdf_path}"): 
            img_path = img_out_dir + f"/{img_info}.png"
            img_info_path = img_out_dir + f"/{img_info}.json"
            img_info_file = load_from_json(img_info_path)
            analyze_image(out_dir, img_path, img_info_file['caption'], 
                          img_info_file['label'], 
                          img_info_file['page_index'], 
                          img_info_file['bbox'])
            # 수식 추출 후 저장.
    if not os.path.exists(formula_out_dir + "/formula_analysis.json"): 
        math_extractor = MathExtractor(out_dir=formula_out_dir,
                                    config_path = config_path)
        math_extractor.extract_formulas(pdf_path)
    
def extract_chunk_text(img_analysis_data, pdf_path, chunk_size = 1000, overlap_size = 150):
    visual_bboxes = load_visual_bboxes_by_page(img_analysis_data, 1.0)
    text_blocks = extract_text_blocks_text_only_excluding_visuals(pdf_path, visual_bboxes, 2.0, 0.10)
    chunks = chunk_text_blocks(text_blocks, chunk_size, overlap_size)
    return chunks

if __name__ == "__main__":     
    #  현재 파일(main.py)이 있는 디렉토리를 기준으로 경로 설정
    app_manager = AppManager()
    app_manager.start_vl_server()
    config_path = str(base_dir / "model_config.yaml")
    configs = load_config(config_path)
    pdf_path = str(base_dir / configs["PDF_PATH"])
    paper_name = configs["PAPER_NAME"] # 논문 이름은 config에서 설정 (폴더 세분화를 위함.)
    formula_out_dir = str(base_dir / "output_formula" / paper_name)
    img_out_dir = str(base_dir / "output_img" / paper_name)
    out_dir = str(base_dir / "output" / paper_name)
    embedding_model_name = configs["EMBEDDING_MODEL_NAME"]
    paper_milvus_store = PaperMilvusStoresLite(paper_name, embedding_model_name)
    extract_all_elements(pdf_path, formula_out_dir, img_out_dir, out_dir)
    img_analysis_data = load_from_jsonl(os.path.join(out_dir, "img_analysis.jsonl"))
    chunks = extract_chunk_text(img_analysis_data, pdf_path, configs["CHUNK_SIZE"], configs["OVERLAP_SIZE"])
    app_manager.stop_vl_server()
    # Milvus DB에 데이터 저장
    print("\n" + "="*80)
    print("Milvus DB에 데이터 저장 시작...")
    print("="*80)
    
    # 문서 ID 설정 (논문 이름 사용)
    doc_id = paper_name
    
    paper_milvus_store.reset_database()
    
    # 1. 텍스트 청크 저장 (page_index와 bbox 정보 포함)
    print("\n[1/3] 텍스트 청크를 Milvus에 저장 중...")
    # page_index와 bbox를 포함한 텍스트 청크 생성
    visual_bboxes_for_chunking = load_visual_bboxes_by_page(img_analysis_data, scale_factor=1.0)
    text_blocks_with_bbox = extract_text_blocks_text_only_excluding_visuals(
        pdf_path=pdf_path,
        visual_bboxes_by_page=visual_bboxes_for_chunking,
        scale_factor=2.0,
        iou_threshold=0.10
    )
    chunks_with_bbox = chunk_text_blocks(
        text_blocks=text_blocks_with_bbox,
        chunk_size=configs["CHUNK_SIZE"],
        overlap_size=configs["OVERLAP_SIZE"]
    )
    
    # Milvus에 저장
    paper_milvus_store.add_text_chunks(chunks_with_bbox, doc_id=doc_id)
    print(f"텍스트 청크 {len(chunks_with_bbox)}개 저장 완료")
    
    retriever_k_dict = {
        "text": 10,
        "equation": 2, 
        "visual": 2
    }
    
    app_manager.start_llm_server()
    rag_app = PaperInfo_RAG(configs)
    chain = rag_app.build_rag_chain(
        prompt_template = ANALYSIS_FORMULA_PROMPT, 
        retriever_k_dict = retriever_k_dict, 
        collections = ["text"]
        )
    fix_llm = rag_app.load_fix_langchain_model()
    
    # 2. 수식(Equation) 저장
    print("\n[2/3] 수식 정보를 Milvus에 저장 중...")
    # formula_analysis.json 파일 로드
    formula_analysis_path = os.path.join(formula_out_dir, "formula_analysis.json")
    if not os.path.exists(formula_analysis_path):
        print(f"수식 분석 파일을 찾을 수 없습니다. {formula_analysis_path}")
    else:
        formula_data = load_from_json(formula_analysis_path)
        
        # 각 페이지의 수식들을 하나의 리스트로 변환
        equation_items = []
        for page_data in formula_data:
            if page_data.get("status") == "success" and "formulas" in page_data:
                page_index = page_data.get("page_index", 0)
                formulas = page_data.get("formulas", [])
                
                for formula in formulas:
                    # 각 수식에 page_index 추가 (이미 있을 수도 있지만 확실하게)
                    formula["page_index"] = page_index
                    try: 
                        formula_analysis = chain.invoke({"question": formula["latex"]})
                        formula_analysis = parse_json_output(formula_analysis)
                        formula["analysis"] = formula_analysis["analysis"]
                        formula["symbol"] = formula_analysis["symbol"]
                        equation_items.append(formula)
                    except Exception as e: 
                        response = fix_llm.invoke(REFINE_JSON_PROMPT.format(broken_text = delete_think_tag(formula_analysis), 
                                                                     error_msg=str(e)))
                        formula_analysis = parse_json_output(response.content)
                        formula["analysis"] = formula_analysis["analysis"]
                        formula["symbol"] = formula_analysis["symbol"]
                        equation_items.append(formula)
        # Milvus에 저장
        if equation_items:
            paper_milvus_store.add_equations(equation_items, doc_id=doc_id)
            print(f"수식 {len(equation_items)}개 저장 완료")
        else:
            print(f"저장할 수식이 없습니다.")
    
    app_manager.stop_llm_server()
    
    # 3. 그림(Visual) 저장
    print("\n[3/3] 그림 정보를 Milvus에 저장 중...")
    # img_analysis.jsonl은 이미 로드되어 있음
    if not img_analysis_data:
        print(f"그림 분석 데이터가 없습니다.")
    else:
        # Milvus에 저장
        paper_milvus_store.add_visuals(img_analysis_data, doc_id=doc_id)
        print(f"그림/표 {len(img_analysis_data)}개 저장 완료")
    
    # 저장 완료 요약
    print("\n" + "="*80)
    print("Milvus DB 저장 완료!")
    print("="*80)
    print(f"  - 논문 이름: {paper_name}")
    print(f"  - DB 경로: {paper_milvus_store.db_path}")
    print(f"  - 텍스트 컬렉션: {paper_milvus_store.text_collection}")
    print(f"  - 수식 컬렉션: {paper_milvus_store.eq_collection}")
    print(f"  - 그림 컬렉션: {paper_milvus_store.visual_collection}")
    print("="*80 + "\n")
    
    pass