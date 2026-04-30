import streamlit as st
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import torchvision

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
if "kamera_aktif" not in st.session_state:
    st.session_state.kamera_aktif = False

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


# ✅ WAJIB ADA INI
model, class_names = load_model()

# ==========================
#     SIDEBAR MENU
# ==========================
st.sidebar.title("🍅 Deteksi Penyakit Tomat")

# Sidebar Image (Circle Frame)
try:
    img = Image.open("images/logo.jpg")
    st.sidebar.image(img, use_container_width=True, output_format="PNG", caption="Daun Tomat",)
except:
    st.sidebar.warning("Sidebar image not found: images/sidebar_leaf.jpg")

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

    # Tentang Aplikasi
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

    # Cara Menggunakan
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




elif menu == "Upload Citra":
    st.title("🔍 Deteksi Penyakit Daun Tomat")

    # ==========================
    # PILIH MODE INPUT
    # ==========================
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

        # Matikan upload file
        st.session_state.uploaded_image = None

        st.subheader("📷 Kamera")

        # Tombol aktifkan kamera
        if not st.session_state.kamera_aktif:
            if st.button("▶️ Aktifkan Kamera"):
                st.session_state.kamera_aktif = True

        # Kamera hanya muncul setelah tombol ditekan
        if st.session_state.kamera_aktif:
            camera_image = st.camera_input("Ambil gambar daun tomat")

            if camera_image is not None:
                st.session_state.camera_image = camera_image

                st.image(
                    camera_image,
                    caption="Citra dari Kamera",
                    width=300
                )

            if st.button("❌ Matikan Kamera"):
                st.session_state.kamera_aktif = False
                st.session_state.camera_image = None

    # ==========================
    # MODE UPLOAD FILE
    # ==========================
    elif mode == "Upload File":

        # Matikan kamera saat pindah menu
        st.session_state.kamera_aktif = False
        st.session_state.camera_image = None

        st.subheader("📁 Upload Citra Daun Tomat")

        uploaded_file = st.file_uploader(
            "Upload gambar daun tomat",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            st.session_state.uploaded_image = uploaded_file

            st.image(
                uploaded_file,
                caption="Citra dari Upload File",
                width=300
            )

    st.markdown("---")

    # ==========================
    # TOMBOL PREDIKSI
    # ==========================
    if st.button("🔮 Prediksi Penyakit"):

        if st.session_state.camera_image is not None:
            image = Image.open(st.session_state.camera_image).convert("RGB")

        elif st.session_state.uploaded_image is not None:
            image = Image.open(st.session_state.uploaded_image).convert("RGB")

        else:
            st.warning("Silakan ambil gambar atau upload citra terlebih dahulu.")
            st.stop()

    # ==========================
# TRANSFORM (PREPROCESSING)
# ==========================
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
            )
        ])

# ==========================
# PREPROCESSING IMAGE
# ==========================
        img = transform(image).unsqueeze(0)

# ==========================
# MODEL PREDICTION
# ==========================
        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)

# ==========================
# RESULT
# ==========================
        predicted_class = class_names[predicted.item()]
        confidence = round(confidence.item() * 100, 2)

        st.subheader("📌 Hasil Prediksi")
        st.success(f"Jenis Penyakit: {predicted_class}")
        st.info(f"Tingkat Keyakinan Model: {confidence}%")

        st.write("Index prediksi:", predicted.item())
        st.write("Class name:", class_names[predicted.item()])

        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)

# TAMPILKAN DI STREAMLIT
        st.write("Probabilities:", probabilities)
        st.write("Confidence:", confidence.item())
        st.write("Predicted class:", predicted.item())


# ==========================
#     JENIS PENYAKIT
# ==========================
elif menu == "Jenis Penyakit":
    st.title("Daftar Jenis Penyakit Daun Tomat")
    st.write("Berikut adalah daftar 11 jenis penyakit daun tomat lengkap dengan gambar contoh dan penjelasannya.")

    import os

    penyakit_info = {
        "bacterial_spot": {
            "desc": "Penyakit ini disebabkan oleh bakteri Xanthomonas campestris, ditandai dengan bercak kecil berwarna coklat kehitaman."
        },
        "early_blight": {
            "desc": "Disebabkan oleh jamur Alternaria solani dengan bercak coklat berbentuk lingkaran."
        },
        "healthy": {
            "desc": "Daun tomat sehat berwarna hijau tanpa bercak atau kerusakan."
        },
        "late_blight": {
            "desc": "Disebabkan oleh Phytophthora infestans dengan bercak gelap yang cepat menyebar."
        },
        "leaf_mold": {
            "desc": "Jamur Passalora fulva menyebabkan bercak kuning dan lapisan jamur di bawah daun."
        },
        "tomato_mosaic_virus": {
            "desc": "Virus menyebabkan pola mosaik hijau-kuning pada daun."
        },
        "powdery_mildew": {
            "desc": "Lapisan putih seperti tepung pada permukaan daun."
        },
        "septoria_leaf_spot": {
            "desc": "Bercak kecil abu-abu dengan tepi gelap."
        },
        "target_spot": {
            "desc": "Bercak berbentuk lingkaran seperti target."
        },
        "spider_mites_two_spotted_spider_mite": {
            "desc": "Hama tungau menyebabkan bintik kuning dan jaring halus."
        },
        "tomato_yellow_teaf_turl_virus": {
            "desc": "Daun menguning, menggulung, dan pertumbuhan tanaman terhambat."
        }
    }

    # GRID 2 KOLOM (biar tidak panjang ke bawah)
    outer_cols = st.columns(2)

    for i, (nama, info) in enumerate(penyakit_info.items()):

        # Ambil gambar langsung dari folder images
        img_path = f"images/{nama}.jpg"

        with outer_cols[i % 2]:

            col1, col2 = st.columns([1, 2])

            with col1:
                if os.path.exists(img_path):
                    st.image(img_path, width=120)
                else:
                    st.warning(f"Gambar tidak ditemukan: {nama}.jpg")

            with col2:
                st.subheader(nama)
                st.write(info["desc"])

            st.markdown("---")

        if (i + 1) % 2 == 0:
            outer_cols = st.columns(2)