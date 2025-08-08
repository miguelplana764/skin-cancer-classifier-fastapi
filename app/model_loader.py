# app/model_loader.py
import os
import tensorflow as tf
from huggingface_hub import hf_hub_download

MODEL_PATH = "app/model/efficientnetv2s.h5"
REPO_ID = "Miguel764/efficientnetv2s-skin-cancer-classifier"
FILENAME = "efficientnetv2s.h5"

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found locally. Downloading from Hugging Face...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir="app/model"
        )
    else:
        print("Model already exists locally.")

    return tf.keras.models.load_model(MODEL_PATH)
