import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# Local cache directory for the model
MODEL_PATH = "happyface_emotion_model"
# Hugging Face model name
MODEL_NAME = "dima806/facial_emotions_image_detection"

# Ensure model is available (load local or download)
if os.path.exists(MODEL_PATH):
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
        model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
        print("Loaded model from local cache.")
    except Exception as e:
        print("Local model load failed, downloading a new copy:", e)
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        os.makedirs(MODEL_PATH, exist_ok=True)
        processor.save_pretrained(MODEL_PATH)
        model.save_pretrained(MODEL_PATH)
else:
    print("Downloading model from Hugging Face...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    os.makedirs(MODEL_PATH, exist_ok=True)
    processor.save_pretrained(MODEL_PATH)
    model.save_pretrained(MODEL_PATH)
    print("Model downloaded and cached locally.")

def predict_emotion(image_path):
    """Predict the emotion in a given image path."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    emotion = model.config.id2label[predicted_class_idx]
    return emotion
