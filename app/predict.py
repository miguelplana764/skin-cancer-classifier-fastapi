# app/predict.py
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from app.model_loader import load_model

model = load_model()

# Pre-calculated optimal temperature value
TEMPERATURE = 2.77

class_names_mapping = {
    0: "AKIEC",
    1: "BCC",
    2: "BKL",
    3: "DF",
    4: "MEL",
    5: "NV",
    6: "VASC"
}

full_names = {
    "AKIEC": "Actinic Keratoses and Intraepithelial Carcinoma (AKIEC)",
    "BCC": "Basal Cell Carcinoma (BCC)",
    "BKL": "Benign Keratosis-like Lesions (BKL)",
    "DF": "Dermatofibroma (DF)",
    "MEL": "Melanoma (MEL)",
    "NV": "Melanocytic Nevi (NV)",
    "VASC": "Vascular Lesions (VASC)"
}

def preprocess_image(file):
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = (img_array - 0.5) * 2
    return np.expand_dims(img_array, axis=0)

def predict_image(file):
    processed = preprocess_image(file)

    # Obtaining logits by disabling the final softmax
    logits_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    logits = logits_model(processed)

    # Apply final layer without softmax (if the last layer is Dense with softmax)
    final_dense = model.layers[-1]
    logits = final_dense(logits)

    # Apply T-scaling: logits / T, then softmax
    scaled_logits = logits / TEMPERATURE
    scaled_probs = tf.nn.softmax(scaled_logits).numpy()[0]

    class_idx = int(np.argmax(scaled_probs))
    top_label = full_names[class_names_mapping[class_idx]]
    top_confidence = float(scaled_probs[class_idx])

    all_predictions = [
        {"label": class_names_mapping[i], "confidence": float(pred)}
        for i, pred in enumerate(scaled_probs)
    ]

    return top_label, top_confidence, all_predictions
