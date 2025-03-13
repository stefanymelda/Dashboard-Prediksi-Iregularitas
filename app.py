import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load model dan encoder
model = joblib.load('decision_tree_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Fungsi untuk encoding input data
def encode_input(data, encoders):
    encoded_data = data.copy()
    for col, le in encoders.items():
        if col in data.columns:
            encoded_data[col] = le.transform(data[col])
    return encoded_data

# Fungsi untuk decoding hasil prediksi
def decode_prediction(prediction, target_encoder):
    return target_encoder.inverse_transform(prediction)

# Fungsi untuk menyimpan data prediksi secara kumulatif
def save_prediction(data, filepath='predicted_data.csv'):
    try:
        # Jika file sudah ada, baca file lama
        existing_data = pd.read_csv(filepath)

        # Gabungkan data baru dengan data lama tanpa duplikasi
        updated_data = pd.concat([existing_data, data]).drop_duplicates(ignore_index=True)
    except FileNotFoundError:
        # Jika file belum ada, gunakan data baru
        updated_data = data

    # Simpan data gabungan ke file CSV
    updated_data.to_csv(filepath, index=False)

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Dashboard Prediksi Iregularitas", layout="wide")
st.title("Dashboard Prediksi Iregularitas")

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
sidebar_options = ["Beranda", "Form Prediksi Data", "Visualizations"]
selection = st.sidebar.radio("Pilih Menu", sidebar_options)

# Menambahkan custom CSS untuk styling form
st.markdown("""
    <style>
        .css-18e3th9 {max-width: 700px; margin-left: auto; margin-right: auto;} /* Membatasi lebar form*/
        .css-1x8b5iz {height: 35px;} /* Menyesuaikan ukuran input form */
        .css-1v0mbd3 {padding: 6px;} /* Menambah ruang antara elemen form */
        .css-16hu4h5 {font-size: 14px;} /* Menyesuaikan ukuran font pada form */
    </style>
""", unsafe_allow_html=True)

# Menangani pilihan menu dari sidebar
if selection == "Beranda":
    st.markdown("### Selamat datang di Dashboard Iregularitas")
    st.markdown("Gunakan menu di sebelah kiri untuk melakukan klasifikasi tindakan korektif atau melihat visualisasi.")
    
    st.markdown("""
        **Dashboard ini membantu memprediksi tindakan korektif untuk penanganan iregularitas dengan cepat dan mudah.**
        - **Form Prediksi Data**: Masukkan data untuk memprediksi tindakan korektif.
        - **Visualizations**: Lihat data prediksi yang sudah tersimpan dalam bentuk visualisasi.
    """)

    # Menampilkan gambar dari file lokal
    st.image("logistik.jpg", use_container_width=True, width=200)
    
elif selection == "Form Prediksi Data":
    st.markdown("### Masukkan data baru untuk memprediksi corrective action.")

    # Menggunakan kolom untuk membuat form lebih ringkas
    col1, col2 = st.columns(2)

    with col1:
        reg_asal = st.selectbox("Regional Asal", ["Pilih Regional Asal", "Regional I Medan", "Regional II Jakarta", "Regional III Bandung", "Regonal IV Semarang", "Regional V Surabaya", "Regional VI Makassar"])
        kode_reg_asal = st.selectbox("Kode Regional Asal", ["Pilih Kode Regional Asal", "20004", "10004", "40004", "50004", "60004", "90004"])
        reg_tujuan = st.selectbox("Regional Tujuan", ["Pilih Regional Tujuan", "Regional I Medan", "Regional II Jakarta", "Regional III Bandung", "Regonal IV Semarang", "Regional V Surabaya", "Regional VI Makassar"])
        kode_reg_tujuan = st.selectbox("Kode Regional Tujuan", ["Pilih Kode Regional Tujuan", "20004", "10004", "40004", "50004", "60004", "90004"])

    with col2:
        dnln = st.selectbox("DNLN", ["DOMESTIK", "LN"])
        deskripsi_iregularitas = st.selectbox("Deskripsi Iregularitas", ["Pilih Deskripsi", "Geotagging Tidak Sesuai", "Foto Antaran Tidak Sesuai", "Gagal X-Ray", "Isi Kiriman Tidak Sesuai", "Kantung Basah", "Kiriman Rusak", "Salah Salur Kiriman", "Salah Tempel Resi", "Selisih Kurang Berat Kiriman", "Lain-Lain"])
        bulan_ba = st.date_input("Bulan BA", value=datetime.today())
        referensi_root_cause = st.selectbox("Referensi Root Cause", ["Pilih Referensi Root Cause", "Pengirim Salah Memberikan Data Penerima", "Pengirim Tidak Melakukan Packing Dengan Baik", "Pengirim Tidak Menginfokan Isi Kiriman Yang Benar", "Petugas Tidak Melaksanakan SOP", "Petugas Tidak Memeriksa Packing", "Petugas Tidak Teliti", "Potensi Kerusakan Pada Saat Perjalanan"])

    # Tombol submit
    submit_button = st.button("Prediksi")

    if submit_button:
        # Konversi input menjadi DataFrame
        input_data = pd.DataFrame({
            "Reg_Asal": [reg_asal],
            "Kode_Reg_Asal": [kode_reg_asal],
            "Reg_Tujuan": [reg_tujuan],
            "Kode_Reg_Tujuan": [kode_reg_tujuan],
            "DNLN": [dnln],
            "Deskripsi_Iregularitas": [deskripsi_iregularitas],
            "Bulan_BA": [bulan_ba.strftime('%Y-%m')],
            "referensi_root_cause": [referensi_root_cause]
        })

        # Encode input data
        try:
            encoded_data = encode_input(input_data, label_encoders)
            encoded_data["Bulan_BA"] = pd.to_datetime(encoded_data["Bulan_BA"]).astype(np.int64) // 10**9

            # Prediksi menggunakan model
            prediction = model.predict(encoded_data)

            # Decode hasil prediksi
            corrective_action = decode_prediction(prediction, label_encoders["corrective_action"])

            st.success(f"Prediksi Corrective Action: {corrective_action[0]}")

            # Simpan data yang dimasukkan dan hasil prediksi ke file CSV
            result_data = input_data.copy()
            result_data['Prediksi_Corrective_Action'] = corrective_action[0]

            # Simpan data secara kumulatif
            save_prediction(result_data)

        except Exception as e:
            st.error(f"Error dalam memproses data: {e}")

elif selection == "Visualizations":
    st.markdown("### Visualisasi Data Prediksi")
    try:
        # Baca file CSV yang menyimpan data kumulatif
        pred_data = pd.read_csv('predicted_data.csv')
        st.write(pred_data)  # Tampilkan data dalam bentuk tabel
        
        # Visualisasi distribusi prediksi
        st.bar_chart(pred_data['Prediksi_Corrective_Action'].value_counts())
    except FileNotFoundError:
        st.error("Data prediksi tidak ditemukan. Silakan input data baru untuk prediksi.")

# Footer
st.markdown("---")
st.markdown("**Dashboard ini menggunakan model Decision Tree CART untuk prediksi corrective action.**")
