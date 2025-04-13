import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import time
import os
import base64
from io import BytesIO

st.set_page_config(page_title="CIFAR-10 Classifier", layout="centered")

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

SAMPLE_IMAGES = [f"Assets/img_{i}.png" for i in range(1, 11)]

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 10)
    model_path = "Models/model.pth" 
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model.to('cpu')

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

st.title("CIFAR-10 Image Classifier")
st.caption("!!! *CIFAR-10 (only 32x32 image resolution). Accuracy for high resolution won't be accurate. !!!")
st.caption("For CIFAR-10 Testing and Fun Only")
st.caption("Updated later -ridwanenam")

# === SESSION STATE ===
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None

# === HELPER ===
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def run_model_prediction(image):
    st.image(image, caption="Selected Image", width=300)
    with st.spinner("Processing"):
        bar = st.empty()
        for i in range(9):
            bar.progress((i + 1) / 9)
            time.sleep(0.1)

    with torch.no_grad():
        input_tensor = transform(image).unsqueeze(0)
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, pred_idx = torch.max(probs, 0)

    st.success(f"Prediction: **{CLASS_NAMES[pred_idx]}** ({confidence * 100:.1f}%)")

    if confidence < 0.5:
        st.warning("Model isn't accurate enough or CIFAR-10 excluded")

    st.markdown("Prediction: ")
    top_probs, top_idxs = torch.topk(probs, 3)
    for i, (prob, idx) in enumerate(zip(top_probs, top_idxs)):
        st.write(f"{i + 1}. **{CLASS_NAMES[idx]}** — {prob * 100:.1f}%")

# === SAMPLE SELECTOR ===
st.markdown("### Choose Sample Images: ")
sample_cols = st.columns(5)

for i, img_path in enumerate(SAMPLE_IMAGES):
    with sample_cols[i % 5]:
        if os.path.exists(img_path):
            if st.button(f"Sample {i+1}", key=f"img_btn_{i}"):
                st.session_state.selected_sample = img_path
            img = Image.open(img_path).resize((112, 112))
            b64_img = image_to_base64(img)
            st.markdown(
                f"""
                <div style="border:1px solid #444; padding:2px; border-radius:10px;">
                    <img src="data:image/png;base64,{b64_img}" width="112" />
                </div>
                """,
                unsafe_allow_html=True
            )

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("Upload Images", type=["jpg", "png", "jpeg"])

# === BUTTONS ===
col1, col2 = st.columns(2)
with col1:
    if st.button("Sample Images Prediction") and st.session_state.selected_sample:
        image = Image.open(st.session_state.selected_sample).convert("RGB")
        run_model_prediction(image)
with col2:
    if st.button("Selected Images Prediction") and uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        run_model_prediction(image)

# === RESET ===
if st.button("Reset"):
    st.session_state.selected_sample = None
    st.rerun()
