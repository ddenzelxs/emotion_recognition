import streamlit as st
import numpy as np
import cv2
import joblib

# ==========================
# Load Model dan Tools
# ==========================
model = joblib.load("random_forest_model.pkl")
pca = joblib.load("pca_transform.pkl")
label_encoder = joblib.load("label_encoder.pkl")

IMG_SIZE = (64, 64)

def preprocess_image(uploaded_file):
    # Baca sebagai array numpy
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Gagal membaca gambar. Pastikan file berupa gambar.")

    img_resized = cv2.resize(img, IMG_SIZE)
    img_flattened = img_resized.flatten().reshape(1, -1)
    img_pca = pca.transform(img_flattened)
    return img, img_pca

# ==========================
# Streamlit App
# ==========================
st.title("Deteksi Ekspresi Wajah")
st.write("Unggah gambar wajah berukuran cukup jelas (grayscale atau RGB).")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        original_img, img_processed = preprocess_image(uploaded_file)
        prediction = model.predict(img_processed)
        label = label_encoder.inverse_transform(prediction)[0]

        st.image(original_img, caption="Gambar yang Diupload", channels="GRAY", width=250)
        st.success(f"Ekspresi terdeteksi: **{label.upper()}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
