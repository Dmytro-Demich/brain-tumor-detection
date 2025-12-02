import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

IMG_SIZE = 128
MODEL_PATH = "brain_tumor_cnn.pth"
METRICS_PATH = "metrics.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BrainTumorCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
        conv_out = self.conv(dummy)
        flatten_size = conv_out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class MRIValidator:
    @staticmethod
    def is_mri(image):
        img = np.array(image)

        diff = np.abs(img[:, :, 0] - img[:, :, 1]) + np.abs(img[:, :, 1] - img[:, :, 2])
        if diff.mean() > 15:
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=40,
            maxRadius=300
        )

        return circles is not None

class Preprocessor:
    @staticmethod
    def preprocess(image):
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = np.transpose(image, (2, 0, 1))
        return torch.tensor(image, dtype=torch.float32)

class DescriptionProvider:
    @staticmethod
    def get_description(label):
        if label == "glioma":
            return (
                "Гліома — це пухлина, що виникає з гліальних клітин, "
                "які виконують підтримувальні функції в мозку. "
                "Може бути як доброякісною, так і злоякісною. "
                "Симптоми часто включають головний біль, судоми та когнітивні порушення."
            )

        elif label == "meningioma":
            return (
                "Менінгіома — найчастіше доброякісна пухлина, що розвивається "
                "з оболонок головного мозку. Росте повільно, але може викликати "
                "здавлення мозкових структур. Можливі симптоми: головний біль, "
                "порушення зору, слабкість кінцівок."
            )

        elif label == "notumor":
            return (
                "Ознак внутрішньочерепних пухлин не виявлено. "
                "Структури мозку виглядають без патологічних утворень. "
                "Якщо симптоми зберігаються — потрібна консультація лікаря."
            )

        elif label == "pituitary":
            return (
                "Пухлина гіпофіза формується в ділянці гіпофіза та може впливати "
                "на гормональну систему. Часто спричиняє порушення зору, "
                "ендокринні зміни та головний біль."
            )

class ModelPredictor:
    def __init__(self, model_path):
        self.model = BrainTumorCNN().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()

        self.labels_ua = ["Гліома", "Менінгіома", "Без пухлини", "Пухлина гіпофіза"]
        self.labels_eng = ["glioma", "meningioma", "notumor", "pituitary"]

    def predict(self, img_tensor):
        pred = self.model(img_tensor.unsqueeze(0).to(DEVICE))
        idx = pred.argmax().item()
        return self.labels_ua[idx], self.labels_eng[idx]
    
class MetricsLoader:
    def __init__(self, path="metrics.txt"):
        self.path = path

    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            accuracy = float(lines[0].split(":")[1].strip())
            precision = float(lines[1].split(":")[1].strip())

            return accuracy, precision

        except:
            return None, None

predictor = ModelPredictor(MODEL_PATH)
metrics = MetricsLoader(METRICS_PATH)
val_accuracy, val_precision = metrics.load()


st.set_page_config(page_title="Виявлення пухлин мозку за МРТ-зображенням", layout="wide")

st.markdown("""
<style>
.sidebar-text { font-size: 18px; }
.result-title { font-size: 38px; font-weight: 900; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Опис")
    st.write("Веб-застосунок виявляє пухлини мозку з МРТ-зображенням")

    st.markdown("## Модель розпізнає:")
    st.write("- Гліома")
    st.write("- Менінгіома")
    st.write("- Пухлина гіпофіза")
    st.write("- Без пухлини")

    st.markdown("## Точність моделі:")
    if val_accuracy is not None:
        st.write(f"Валідаційна точність: **{val_accuracy}%**")
    else:
        st.write("Точність недоступна")


    st.markdown("## Як користуватися:")
    st.write("1. Завантажте МРТ-фото.")
    st.write("2. Модель зробить прогноз.")
    st.write("3. Дивіться результат праворуч.")

st.markdown("<h1 style='text-align:center;'>Виявлення пухлин мозку за МРТ-зображенням</h1>", unsafe_allow_html=True)

uploaded = st.file_uploader("Завантажте МРТ:", type=['jpg', 'jpeg', 'png'])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    if not MRIValidator.is_mri(image):
        st.error("Це не схоже на МРТ-зображення. Завантажте, будь ласка, медичний МРТ-скан.")
        st.image(image, caption="Завантажене зображення", width=450)
        st.stop()

    img_tensor = Preprocessor.preprocess(image)
    label_ua, label_eng = predictor.predict(img_tensor)

    left, right = st.columns([1.2, 1])

    with left:
        st.image(image, caption="Вибране МРТ", width=450)

    with right:
        st.markdown(f"<div class='result-title'> Результат: {label_ua}</div>", unsafe_allow_html=True)
        st.write(DescriptionProvider.get_description(label_eng))
