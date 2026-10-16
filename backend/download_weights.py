import os
from pathlib import Path
import urllib.request

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
WEIGHTS_URL = os.environ.get(
    "WEIGHTS_URL",
    "https://huggingface.co/MEETJAIN6091/deepfake-detector-weights/resolve/main/dual_paradigm_deepfake_model.weights.h5"
)

def download_weights():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = MODELS_DIR / "dual_paradigm_deepfake_model.weights.h5"

    if weights_path.exists() and weights_path.stat().st_size > 200_000_000:
        print("Weights file already exists locally.")
        return

    print("Pre-downloading weights from Hugging Face...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(WEIGHTS_URL, headers=headers)
    with urllib.request.urlopen(req) as response, open(weights_path, 'wb') as out_file:
        out_file.write(response.read())
    print("Weights downloaded successfully.")

if __name__ == "__main__":
    download_weights()