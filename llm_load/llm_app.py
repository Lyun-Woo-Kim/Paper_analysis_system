# llm_app.py
import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from llm_engine import engine

# 현재 파일이 있는 디렉토리의 상위 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

from utils.utils import load_config

app = Flask(__name__)

# 현재 파일 기준으로 model_config.yaml 경로 설정
config_path = root_dir / "model_config.yaml"
config = load_config(str(config_path))

@app.route('/status', methods=['GET'])
def status():
    """현재 모델 로드 상태 확인"""
    return jsonify({
        "loaded": engine.is_loaded(),
        "model": config["LLM_MODEL_NAME"]
    })

@app.route('/load', methods=['POST'])
def load():
    """모델 명시적 로드"""
    try:
        engine.load_model()
        return jsonify({"status": "Model loaded", "loaded": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/unload', methods=['POST'])
def unload():
    """모델 메모리 해제 (GPU 비우기)"""
    try:
        engine.unload_model()
        return jsonify({"status": "Model unloaded", "loaded": False})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    """텍스트 생성 요청"""
    data = request.json
    messages = data.get('messages', [])

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    try:
        response_text = engine.generate_response(messages)
        return jsonify({
            "response": response_text,
            "status": "success"
        })
    except Exception as e:
        print(f"Error during generation: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"Server starting on port {config['LLM_PORT']}...")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["LLM_CUDA_DEVICE"])
    app.run(host=config.get("LLM_HOST", "localhost"), port=config["LLM_PORT"])

