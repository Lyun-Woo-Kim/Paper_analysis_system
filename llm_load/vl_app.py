# app.py
import os, sys
from flask import Flask, request, jsonify
from vl_engine import engine

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..')
sys.path.insert(0, root_dir)
from utils.utils import load_config

app = Flask(__name__)

config = load_config(f"{os.path.join(current_dir, '..')}/model_config.yaml")

@app.route('/status', methods=['GET'])
def status():
    """현재 모델 로드 상태 확인"""
    return jsonify({
        "loaded": engine.is_loaded(),
        "model": config["VL_MODEL_NAME"]
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

@app.route('/analyze', methods=['POST'])
def analyze():
    """논문/이미지 분석 요청"""
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
    print(f"Server starting on port {config['VL_PORT']}...")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["VL_CUDA_DEVICE"])
    app.run(host=config["VL_HOST"], port=config["VL_PORT"])