import streamlit as st
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import os
import tempfile
import shutil

# Import YOLO

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

st.set_page_config(page_title="Plant Leaf Disease Detection System")

# Periksa apakah library YOLO tersedia
def cek_library():
    if not YOLO_AVAILABLE:
        st.error("Ultralytics tidak terpasang. Silakan instal dengan perintah berikut:")
        st.code("pip install ultralytics")
        return False
    return True

st.markdown("""
<div style="background-color:#17202a; padding: 20px; text-align: center;">
    <h1 style="color: white;"> Plant Leaf Disease Detection System </h1>
    <h5 style="color: white;">Deteksi Kesehatan Daun</h5>
    <p style="color: white; margin-top: 10px;">
        Dibuat oleh <strong>Syaiful Muflichin Purnama</strong> – 
        <a href="https://www.linkedin.com/in/syaiful-muflichin-purnama-3bb89699/" style="color: #1abc9c;" target="_blank">LinkID</a>
    </p>
</div>
""", unsafe_allow_html=True)

# Pastikan library sudah terpasang sebelum melanjutkan
if cek_library():
# upload gambar
     uploaded_file = st.file_uploader("upload gambar", type=['jpg', 'jpeg', 'png'])

     if uploaded_file:
        # Simpan sementara
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, "gambar.jpg")
        image = Image.open(uploaded_file)

        #Ubah Ukuran Gambar
        image = image.resize((300,300))
        image.save(temp_file)

        #Tampilkan gamabar
        st.markdown("<div style='text-align: center;'>",unsafe_allow_html=True)
        st.image(image, caption="Gambar yang diupload")
        st.markdown("</div>", unsafe_allow_html=True)
        
        #Deteksi Gambar
        if st.button("Deteksi Gambar"):
          with st.spinner("Sedang diproses"):
              try:
                  model = YOLO('best.pt')
                  hasil = model(temp_file)
                  
                  #Ambil Hasil Prediksi
                  nama_objek = hasil[0].names
                  nilai_prediksi= hasil[0].probs.data.numpy().tolist()
                  objek_terdeteksi = nama_objek[np.argmax(nilai_prediksi)]
                  
                  #buat grafik
                  grafik = go.Figure([go.Bar(x=list(nama_objek.values()), y=nilai_prediksi)])
                  grafik.update_layout(title='Tingkat Keyakinan Prediksi', xaxis_title='Kesehatan Daun',
                  yaxis_title='Keyakinan')
                  
                  #Tampilkan hasil
                  st.write(f"penyakit terdeteksi:{objek_terdeteksi}")
                  st.plotly_chart(grafik)
                
              except Exception as e :
                  st.error("Gambar tidak dapat terdeteksi")
                  st.error(f"Error:{e}")
                  
              #Hapus file sementara
              shutil.rmtree(temp_dir,ignore_errors=True)

st.markdown(
"<div style='text-align: center;' class='footer'>Program Deteksi Kesehatan Daun @2025</div>",
unsafe_allow_html=True
)


