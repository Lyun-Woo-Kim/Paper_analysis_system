import os
import sys
import subprocess
import socket
import time
from pathlib import Path

# 현재 디렉토리와 루트 디렉토리 설정
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

from utils.utils import load_config

# LLM과 VL을 동시에 관리할 수 있는 클래스
class AppManager:
    
    def __init__(self):
        config_path = root_dir / "model_config.yaml"
        self.config = load_config(str(config_path))
        
        self.llm_process = None
        self.vl_process = None
        
        # LLM 서버 설정
        self.llm_model_path = self.config.get("LLM_MODEL_NAME", "Qwen/Qwen3-8B")
        self.llm_host = self.config.get("LLM_HOST", "localhost")
        self.llm_port = self.config.get("LLM_PORT", 8001)
        self.llm_gpu_device = self.config.get("LLM_CUDA_DEVICE", 0)
        self.llm_dtype = self.config.get("LLM_DTYPE", "FP16")
        self.llm_gpu_memory_utilization = self.config.get("LLM_GPU_MEMORY_UTILIZATION", 0.80)
        self.llm_max_model_len = self.config.get("LLM_MAX_MODEL_LEN", 8000)
        # VL 서버 설정
        self.vl_host = self.config.get("VL_HOST", "localhost")
        self.vl_port = self.config.get("VL_PORT", 8000)
        self.vl_gpu_device = self.config.get("VL_CUDA_DEVICE", 0)
        
        self.timeout = 1200 # 서버가 start될 때까지 최대 20분 대기 (모델 설치 시간 고려)
        self.interval = 0.2 # 서버가 start되었는 지 확인 간격 0.2초
    
    def check_model_start(self, host, port): 
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            # [추가됨] 프로세스가 중간에 죽었는지 확인
            if self.llm_process is not None and self.llm_process.poll() is not None:
                print(f"\n 서버 프로세스가 종료되었습니다. (Exit Code: {self.llm_process.returncode})")
                # 로그가 PIPE로 잡혀있지 않다면 화면에 에러가 이미 떴을 것입니다.
                return False

            try:
                with socket.create_connection((host, port), timeout=1.0):
                    return True
            except OSError:
                time.sleep(self.interval)
        return False
    
    def start_llm_server(self):
        """vLLM 서버 시작"""
        if self.llm_process is not None:
            print("LLM 서버가 이미 실행 중입니다.")
            return False
        
        # 환경 변수 설정
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.llm_gpu_device)
        
        # vLLM 서버 명령어
        command = [
            "vllm", "serve", self.llm_model_path,
            "--port", str(self.llm_port),
            "--max-model-len", str(self.llm_max_model_len),
            "--tokenizer", self.llm_model_path,
            "--dtype", self.llm_dtype,
            "--gpu-memory-utilization", str(self.llm_gpu_memory_utilization)
        ]
        
        print(f"LLM 서버 시작 중... (GPU {self.llm_gpu_device}, Port {self.llm_port}, Host {self.llm_host})")
        print(f"Command: {' '.join(map(str, command))}")
        
        try:
            # 백그라운드로 실행
            self.llm_process = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if not self.check_model_start(self.llm_host, self.llm_port):
                print("LLM 서버 시작 실패: 서버가 정상적으로 시작되지 않았습니다.")
                self.llm_process = None
                return False
            print(f"LLM 서버가 시작되었습니다. (PID: {self.llm_process.pid})")
            return True
        except Exception as e:
            print(f"LLM 서버 시작 실패: {e}")
            self.llm_process = None
            return False
    
    def stop_llm_server(self):
        """vLLM 서버 중지"""
        if self.llm_process is None:
            print("LLM 서버가 실행 중이 아닙니다.")
            return False
        
        try:
            print(f"LLM 서버 중지 중... (PID: {self.llm_process.pid})")
            self.llm_process.terminate()
            
            # 프로세스가 종료될 때까지 대기 (최대 10초)
            try:
                self.llm_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("정상 종료 실패, 강제 종료 중...")
                self.llm_process.kill()
                self.llm_process.wait()
            
            self.llm_process = None
            print("LLM 서버가 중지되었습니다.")
            return True
        except Exception as e:
            print(f"LLM 서버 중지 실패: {e}")
            self.llm_process = None
            return False
    
    def start_vl_server(self):
        """VL 앱 서버 시작"""
        if self.vl_process is not None:
            print("VL 서버가 이미 실행 중입니다.")
            return False
        
        # 환경 변수 설정
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.vl_gpu_device)
        
        # VL 앱 실행 명령어
        vl_app_path = current_dir / "vl_app.py"
        command = [
            sys.executable, str(vl_app_path)
        ]
        
        print(f"VL 서버 시작 중... (GPU {self.vl_gpu_device}, Port {self.vl_port}, Host {self.vl_host})")
        
        try:
            # 백그라운드로 실행
            self.vl_process = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(current_dir)
            )
            if not self.check_model_start(self.vl_host, self.vl_port):
                print("VL 서버 시작 실패: 서버가 정상적으로 시작되지 않았습니다.")
                self.vl_process = None
                return False
            print(f"VL 서버가 시작되었습니다. (PID: {self.vl_process.pid})")
            return True
        except Exception as e:
            print(f"VL 서버 시작 실패: {e}")
            self.vl_process = None
            return False
    
    def stop_vl_server(self):
        """VL 앱 서버 중지"""
        if self.vl_process is None:
            print("VL 서버가 실행 중이 아닙니다.")
            return False
        
        try:
            print(f"VL 서버 중지 중... (PID: {self.vl_process.pid})")
            self.vl_process.terminate()
            
            # 프로세스가 종료될 때까지 대기 (최대 10초)
            try:
                self.vl_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("정상 종료 실패, 강제 종료 중...")
                self.vl_process.kill()
                self.vl_process.wait()
            
            self.vl_process = None
            print("VL 서버가 중지되었습니다.")
            return True
        except Exception as e:
            print(f"VL 서버 중지 실패: {e}")
            self.vl_process = None
            return False
    
    def start_all(self):
        """모든 서버 시작"""
        print("=" * 60)
        print("모든 서버 시작 중...")
        print("=" * 60)
        
        llm_success = self.start_llm_server()
        time.sleep(2)  # 서버 시작 간격
        
        vl_success = self.start_vl_server()
        
        if llm_success and vl_success:
            print("\n모든 서버가 시작되었습니다!")
            print(f"   - LLM 서버: http://localhost:{self.llm_port}")
            print(f"   - VL 서버: http://{self.vl_host}:{self.vl_port}")
        else:
            print("\n일부 서버 시작에 실패했습니다.")
        
        return llm_success and vl_success
    
    def stop_all(self):
        """모든 서버 중지"""
        print("=" * 60)
        print("모든 서버 중지 중...")
        print("=" * 60)
        
        vl_success = self.stop_vl_server()
        time.sleep(1)
        
        llm_success = self.stop_llm_server()
        
        if llm_success and vl_success:
            print("\n모든 서버가 중지되었습니다!")
        else:
            print("\n일부 서버 중지에 실패했습니다.")
        
        return llm_success and vl_success
    
    def status(self):
        """서버 상태 확인"""
        print("=" * 60)
        print("서버 상태")
        print("=" * 60)
        
        llm_status = "실행 중" if (self.llm_process is not None and self.llm_process.poll() is None) else "중지됨"
        vl_status = "실행 중" if (self.vl_process is not None and self.vl_process.poll() is None) else "중지됨"
        
        print(f"LLM 서버: {llm_status}")
        if self.llm_process is not None:
            print(f"  - PID: {self.llm_process.pid}")
            print(f"  - Port: {self.llm_port}")
        
        print(f"VL 서버: {vl_status}")
        if self.vl_process is not None:
            print(f"  - PID: {self.vl_process.pid}")
            print(f"  - Port: {self.vl_port}")
        
        print("=" * 60)