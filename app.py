import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ====== Konfigurasi ======
st.set_page_config(page_title="Garbage Classification", page_icon="🗑️", layout="centered")

st.title("🗑️ Garbage Classification - Deep Learning (MobileNetV2)")
st.markdown("""
Model ini menggunakan **Transfer Learning (MobileNetV2)** untuk mengklasifikasikan gambar sampah ke dalam 6 kategori (Baterai, Kaca, Metal, Organik, Kertas, dan Plastik).           
Klik tombol "Browse File" dibawah untuk meng-upload gambar yang ingin diprediksi.
""")

# ====== Class Labels ======
class_names = ['Battery', 'Glass', 'Metal', 'Organic', 'Paper', 'Plastic']
model_path = "output/garbage_cnn_model.pth"

# ====== Load Model ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# ====== Transformasi ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====== Upload Gambar ======
uploaded_file = st.file_uploader("📤 Upload gambar sampah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_column_width=True)

    # Prediksi
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    # ====== Hasil Prediksi ======
    st.subheader("Hasil Prediksi Probabilitas Tiap Kelas")
    prob_table = {class_names[i]: f"{probs[i]*100:.2f}%" for i in range(len(class_names))}
    st.table(prob_table)

    # ====== Grafik Probabilitas ======
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(class_names, probs * 100, color=['#007bff', '#17a2b8', '#28a745', '#ffc107', '#fd7e14', '#dc3545'])
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Distribusi Probabilitas Kelas")
    plt.xticks(rotation=30)
    st.pyplot(fig)

    # ====== Kesimpulan ======
    max_idx = np.argmax(probs)
    predicted_class = class_names[max_idx]
    confidence = probs[max_idx] * 100

    st.markdown("---")
    st.markdown(f"###  **Prediksi Akhir:** `{predicted_class.upper()}`")
    st.markdown(f"####  Confidence: **{confidence:.2f}%**")
    st.success(f"Hasil prediksi menunjukkan bahwa gambar kemungkinan besar adalah **{predicted_class.upper()}** dengan persentase {confidence:.2f}%.")

else:
    st.info("Silakan upload gambar terlebih dahulu untuk melakukan prediksi.")

# ====== Footer ======
st.markdown("""
---
👨‍💻 **Developed by Finn Team**  
Powered by PyTorch + Streamlit  
""")
