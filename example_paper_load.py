import os
import requests

def download_example_paper():
    url = f"https://arxiv.org/pdf/1706.03762.pdf" # Attention is all you need 논문
    save_path = "./sample_paper.pdf"
    if not os.path.exists(save_path):
        response = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(response.content)

def download_paper(url, save_path):
    if not os.path.exists(save_path):
        response = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(response.content)