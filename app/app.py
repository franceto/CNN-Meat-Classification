import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from dotenv import load_dotenv
from groq import Groq

st.set_page_config(page_title="Meat Freshness Classifier", page_icon="🥩", layout="wide")

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_PATH = PROJECT_DIR / "artifacts" / "models" / "basic_cnn_final.pt"

load_dotenv(PROJECT_DIR / ".env")

class ConvBN(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class BasicCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBN(3, 32), nn.MaxPool2d(2),
            ConvBN(32, 64), nn.MaxPool2d(2),
            ConvBN(64, 128), nn.MaxPool2d(2),
            nn.Dropout2d(0.05),
            ConvBN(128, 192), nn.MaxPool2d(2),
            nn.Dropout2d(0.08),
            ConvBN(192, 256)
        )
        self.cam_layer = self.features[-1]
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.head(self.features(x))

@st.cache_resource
def load_model():
    package = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = BasicCNN(package["num_classes"])
    model.load_state_dict(package["model_state_dict"])
    model.eval()
    idx_to_class = {int(k): v for k, v in package["idx_to_class"].items()}
    return model, package, idx_to_class

class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.a = None
        self.g = None
        self.h1 = layer.register_forward_hook(self.f_hook)
        self.h2 = layer.register_full_backward_hook(self.b_hook)

    def f_hook(self, m, i, o):
        self.a = o

    def b_hook(self, m, gi, go):
        self.g = go[0]

    def __call__(self, x, target):
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        score = out[:, target].sum()
        score.backward()
        w = self.g.mean(dim=(2, 3), keepdim=True)
        cam = (w * self.a).sum(1).relu()
        cam = F.interpolate(cam[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def close(self):
        self.h1.remove()
        self.h2.remove()

def preprocess(img, package):
    tf = transforms.Compose([
        transforms.Resize((package["img_size"], package["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(package["mean"], package["std"])
    ])
    return tf(img).unsqueeze(0)

def overlay_gradcam(img, cam):
    img = img.resize((cam.shape[1], cam.shape[0]))
    img_np = np.array(img).astype(np.float32) / 255
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return np.clip(0.6 * img_np + 0.4 * heat, 0, 1)

def predict(model, img, package, idx_to_class):
    x = preprocess(img, package)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    pred_id = int(prob.argmax())
    pred_name = idx_to_class[pred_id]
    conf = float(prob[pred_id])
    cam_tool = GradCAM(model, model.cam_layer)
    cam = cam_tool(x, pred_id)
    cam_tool.close()
    overlay = overlay_gradcam(img, cam)
    return pred_id, pred_name, conf, prob, overlay

def ask_groq(question, context):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return "Bạn chưa cấu hình GROQ_API_KEY trong file .env nên chatbot AI chưa hoạt động."
    client = Groq(api_key=key)
    prompt = f"""
Bạn là trợ lý AI tư vấn an toàn thực phẩm.
Trả lời tiếng Việt, ngắn gọn, dễ hiểu.
Không khẳng định thay xét nghiệm phòng lab.
Nếu thịt có dấu hiệu hỏng, khuyên không nên sử dụng.

Ngữ cảnh dự đoán:
{context}

Câu hỏi người dùng:
{question}
"""
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Bạn là trợ lý AI giải thích kết quả phân loại thịt tươi/thịt hỏng và Grad-CAM."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return res.choices[0].message.content

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #7A1E1E;
    margin-bottom: 4px;
}
.sub-title {
    font-size: 18px;
    color: #555;
    margin-bottom: 24px;
}
.result-card {
    padding: 20px;
    border-radius: 18px;
    background: linear-gradient(135deg, #fff7f2, #ffffff);
    border: 1px solid #f0d8d0;
    box-shadow: 0 4px 18px rgba(0,0,0,0.06);
}
.metric-big {
    font-size: 30px;
    font-weight: 800;
}
.chat-box {
    border-radius: 16px;
    padding: 16px;
    background: #f8f8f8;
    border: 1px solid #e8e8e8;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🥩 Meat Freshness Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Phân loại thịt tươi / thịt hỏng bằng PyTorch CNN + Grad-CAM + Groq AI</div>', unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error(f"Không tìm thấy model: {MODEL_PATH}")
    st.stop()

model, package, idx_to_class = load_model()

left, right = st.columns([1, 1])

with left:
    uploaded = st.file_uploader("Upload ảnh thịt", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded:
    raw_img = Image.open(uploaded)
    fmt = raw_img.format
    img = raw_img.convert("RGB")

    with left:
        st.image(img, caption="Ảnh gốc", use_container_width=True)

        info_df = pd.DataFrame([
            ["Tên file", uploaded.name],
            ["Định dạng", fmt],
            ["Kích thước file", f"{uploaded.size / 1024:.2f} KB"],
            ["Độ phân giải", f"{img.width} × {img.height} px"],
            ["Chế độ màu", img.mode]
        ], columns=["Thông tin", "Giá trị"])

        st.dataframe(info_df, hide_index=True, use_container_width=True)

    predict_btn = st.button("🔍 Dự đoán", use_container_width=True)

    if predict_btn:
        pred_id, pred_name, conf, prob, overlay = predict(model, img, package, idx_to_class)

        st.session_state["pred_context"] = {
            "class": pred_name,
            "confidence": conf,
            "prob": {idx_to_class[i]: float(prob[i]) for i in range(len(prob))}
        }

        with right:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### Kết quả dự đoán")
            st.markdown(f'<div class="metric-big">{pred_name}</div>', unsafe_allow_html=True)
            st.write(f"Xác suất: **{conf * 100:.2f}%**")
            st.markdown("</div>", unsafe_allow_html=True)

            prob_df = pd.DataFrame({
                "Class": [idx_to_class[i] for i in range(len(prob))],
                "Probability": prob * 100
            })

            fig = px.bar(prob_df, x="Class", y="Probability", text="Probability", range_y=[0, 100])
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig.update_layout(height=340, title="Xác suất từng lớp")
            st.plotly_chart(fig, use_container_width=True)

        cam_col1, cam_col2 = st.columns(2)
        with cam_col1:
            st.image(img, caption="Ảnh gốc", use_container_width=True)
        with cam_col2:
            st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)

st.divider()
st.markdown("## 💬 Bong bóng chat AI")

context = st.session_state.get("pred_context", None)

if context:
    context_text = f"Class dự đoán: {context['class']}; confidence: {context['confidence']:.4f}; probabilities: {context['prob']}"
else:
    context_text = "Chưa có kết quả dự đoán."

with st.chat_message("assistant"):
    st.write("Bạn có thể hỏi AI về dấu hiệu thịt tươi/hỏng hoặc hỏi vì sao Grad-CAM tập trung vào vùng ảnh đó.")

question = st.chat_input("Nhập câu hỏi cho AI...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("AI đang trả lời..."):
            answer = ask_groq(question, context_text)
            st.write(answer)