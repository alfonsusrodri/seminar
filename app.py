import streamlit as st
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import torchvision
import time

# ==========================
#    PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Deteksi Penyakit Daun Tomat",
    layout="wide",
    page_icon="🍅",
)

# ==========================
#        CUSTOM CSS
# ==========================
page_bg = """
<style>
body {
    background-color: #f6faf5;
}

.sidebar .sidebar-content {
    background-color: #e8f5e9 !important;
}

h1, h2, h3 {
    color: #2e7d32;
    font-family: 'Arial Rounded MT Bold', sans-serif;
}

.card {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    transition: 0.2s;
}

.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.12);
}

.button-style button {
    background-color: #4caf50 !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 10px 20px !important;
}

.sidebar-img {
    border-radius: 50%;
    width: 130px;
    margin-left: auto;
    margin-right: auto;
    display: block;
    border: 4px solid #4caf50;
}
</style>
"""

# ==========================
# SESSION STATE INIT
# ==========================
if "camera_image" not in st.session_state:
    st.session_state.camera_image = None

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

st.markdown(page_bg, unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    checkpoint = torch.load("plant_diseases_modelfinal.pth", map_location="cpu")

    model = torchvision.models.mobilenet_v3_small(pretrained=False)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 11)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    class_names = checkpoint['class_names']
    return model, class_names

model, class_names = load_model()

# Transformasi
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================
#     SIDEBAR MENU
# ==========================
st.sidebar.title("🍅 Deteksi Penyakit Tomat")

try:
    img = Image.open("logo.jpg")
    st.sidebar.image(img, use_container_width=True, output_format="PNG", caption="Daun Tomat")
except:
    st.sidebar.warning("Gambar logo.jpg tidak ditemukan")

menu = st.sidebar.radio(
    "Navigasi",
    ["Beranda", "Upload Citra", "Jenis Penyakit"]
)

# ==========================
#         BERANDA
# ==========================
if menu == "Beranda":
    st.markdown("""
        <style>
        .main-container {
            background: linear-gradient(to bottom right, #e8f5e9, #ffffff);
            padding: 30px;
            border-radius: 14px;
        }
        .title {
            font-size: 42px;
            font-weight: bold;
            color: #1b5e20;
            text-align: center;
            margin-bottom: -5px;
        }
        .subtitle {
            font-size: 20px;
            text-align: center;
            color: #4b4b4b;
            margin-bottom: 30px;
        }
        .section-box {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 2px 10px rgba(0,0,0,0.15);
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>🍅 Sistem Deteksi Penyakit Daun Tomat</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Berbasis Convolutional Neural Network (CNN) & MobileNetV3</p>", unsafe_allow_html=True)

    st.markdown("""
        <div class='section-box'>
            <h3 style='color:#1b5e20;'>📌 Tentang Aplikasi</h3>
            <p style='text-align: justify; font-size:17px;'>
                Aplikasi ini dirancang untuk membantu mengidentifikasi berbagai penyakit pada daun tomat
                menggunakan teknologi <b>deep learning</b>, khususnya arsitektur <b>MobileNetV3</b>.
                Citra daun yang diunggah oleh pengguna akan diproses dan dianalisis untuk menentukan
                jenis penyakit secara otomatis dan cepat.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='section-box'>
            <h3 style='color:#1b5e20;'>🌿 Cara Menggunakan</h3>
            <p style='text-align: justify; font-size:17px;'>
                Buka menu <b>Upload Citra</b> di sidebar, unggah foto daun tomat,
                lalu klik tombol <b>Prediksi</b> untuk mengetahui jenis penyakit.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================
#       UPLOAD CITRA
# ==========================
elif menu == "Upload Citra":
    st.title("🔍 Deteksi Penyakit Daun Tomat")

    mode = st.radio(
        "📌 Pilih Metode Input Citra",
        ["Kamera", "Upload File"],
        horizontal=True
    )

    st.markdown("---")

    # ==========================
    # MODE KAMERA
    # ==========================
    if mode == "Kamera":
        st.session_state.uploaded_image = None
        st.subheader("📷 Kamera")
        st.caption("💡 Pastikan cahaya cukup. Gunakan kamera HP Anda.")

        camera_image = st.camera_input("Ambil gambar daun tomat")

        if camera_image is not None:
            st.session_state.camera_image = camera_image
            st.image(camera_image, caption="Citra dari Kamera", width=300)

    # ==========================
    # MODE UPLOAD FILE
    # ==========================
    elif mode == "Upload File":
        st.session_state.camera_image = None
        st.subheader("📁 Upload Citra Daun Tomat")

        uploaded_file = st.file_uploader(
            "Upload gambar daun tomat (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            try:
                test_img = Image.open(uploaded_file).convert("RGB")
                st.session_state.uploaded_image = uploaded_file
                st.image(uploaded_file, caption="Citra dari Upload File", width=300)
            except Exception as e:
                st.error(f"Gambar tidak valid: {e}")
                st.session_state.uploaded_image = None

    st.markdown("---")

    # ==========================
    # TOMBOL PREDIKSI
    # ==========================
    if st.button("🔮 Prediksi Penyakit"):
        # Validasi ada gambar
        if st.session_state.camera_image is None and st.session_state.uploaded_image is None:
            st.warning("Silakan ambil gambar atau upload citra terlebih dahulu.")
            st.stop()

        # Load gambar
        try:
            if st.session_state.camera_image is not None:
                image = Image.open(st.session_state.camera_image).convert("RGB")
            else:
                image = Image.open(st.session_state.uploaded_image).convert("RGB")
        except Exception as e:
            st.error(f"Gagal membaca gambar: {e}")
            st.stop()

        # Proses prediksi
        with st.spinner("🔄 Sedang memproses gambar..."):
            time.sleep(0.3)
            
            try:
                img_tensor = transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    confidence, predicted = torch.max(probabilities, 0)
                
                predicted_class = class_names[predicted.item()]
                confidence_percent = round(confidence.item() * 100, 2)
                
                # Hasil
                st.subheader("📌 Hasil Prediksi")
                
                if predicted_class.lower() == "healthy":
                    st.success(f"✅ Status: **{predicted_class}** (Daun Sehat)")
                else:
                    st.error(f"⚠️ Penyakit: **{predicted_class}**")
                
                st.info(f"📊 Tingkat Keyakinan: **{confidence_percent}%**")
                st.progress(confidence_percent / 100)
                
                # Top 3 prediksi
                st.write("### 📋 Detail Prediksi")
                top3_prob, top3_idx = torch.topk(probabilities, 3)
                for i in range(3):
                    prob_val = top3_prob[i].item() * 100
                    kelas = class_names[top3_idx[i].item()]
                    st.write(f"- **{kelas}**: {prob_val:.2f}%")
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")

# ==========================
#     JENIS PENYAKIT
# ==========================
elif menu == "Jenis Penyakit":
    st.title("🍅 Daftar Jenis Penyakit Daun Tomat")
    st.write("Berikut adalah daftar penyakit daun tomat yang dapat dideteksi oleh sistem:")

    penyakit_info = {
        "bacterial_spot": {
            "nama": "Bacterial Spot",
            "desc": "Disebabkan oleh bakteri Xanthomonas campestris. Gejala: bercak kecil berwarna coklat kehitaman."
        },
        "early_blight": {
            "nama": "Early Blight",
            "desc": "Disebabkan oleh jamur Alternaria solani. Gejala: bercak coklat berbentuk lingkaran."
        },
        "healthy": {
            "nama": "Healthy (Sehat)",
            "desc": "Daun tomat sehat, berwarna hijau tanpa bercak atau kerusakan."
        },
        "late_blight": {
            "nama": "Late Blight",
            "desc": "Disebabkan oleh Phytophthora infestans. Gejala: bercak gelap yang cepat menyebar."
        },
        "leaf_mold": {
            "nama": "Leaf Mold",
            "desc": "Jamur Passalora fulva menyebabkan bercak kuning dan lapisan jamur di bawah daun."
        },
        "mosaic_virus": {
            "nama": "Mosaic Virus",
            "desc": "Virus menyebabkan pola mosaik hijau-kuning pada daun."
        },
        "septoria_leaf_spot": {
            "nama": "Septoria Leaf Spot",
            "desc": "Bercak kecil abu-abu dengan tepi gelap."
        },
        "target_spot": {
            "nama": "Target Spot",
            "desc": "Bercak berbentuk lingkaran seperti target."
        },
        "twospotted_spider_mite": {
            "nama": "Twospotted Spider Mite",
            "desc": "Hama tungau menyebabkan bintik kuning dan jaring halus."
        },
        "yellow_leaf_curl_virus": {
            "nama": "Yellow Leaf Curl Virus",
            "desc": "Daun menguning, menggulung, dan pertumbuhan terhambat."
        },
        "powdery_mildew": {
            "nama": "Powdery Mildew",
            "desc": "Lapisan putih seperti tepung pada permukaan daun."
        }
    }

    for i, class_name in enumerate(class_names):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if class_name in penyakit_info:
                info = penyakit_info[class_name]
                st.subheader(f"🌿 {info['nama']}")
                st.write(info['desc'])
            else:
                st.subheader(f"🌿 {class_name.replace('_', ' ').title()}")
                st.write("Informasi sedang diperbarui")
        
        with col2:
            img_path = f"images/{class_name}.jpg"
            if os.path.exists(img_path):
                st.image(img_path, width=200)
        
        st.markdown("---")
