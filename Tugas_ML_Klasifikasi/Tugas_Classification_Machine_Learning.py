
# =============================================================================
# TUGAS MACHINE LEARNING - IDENTIFIKASI DATASET CLASSIFICATION
# Nama       : Rafi Fadhil Amanullah 
# NIM        : 241011400893
# Kelas      : 05TPLP023
# Semester   : 5
# Program    : Teknik Informatika
# =============================================================================

import pandas as pd
import os

def tampilkan_teori():
    """Menampilkan penjelasan teori ketiga jenis klasifikasi."""
    print("="*80)
    print("1. KONSEP KLASIFIKASI DALAM MACHINE LEARNING".center(80))
    print("="*80)
    teori = """
A. Binary Classification
   - Jumlah Kelas: Tepat 2 kelas yang saling eksklusif.
   - Jumlah Label per Data: 1 label.
   - Karakteristik Target: Ya/Tidak, Benar/Salah, 0/1.
   - Contoh: Memprediksi apakah email adalah Spam atau Bukan Spam.

B. Multiclass Classification
   - Jumlah Kelas: Lebih dari 2 kelas yang saling eksklusif.
   - Jumlah Label per Data: 1 label (satu data hanya bisa masuk ke satu kategori).
   - Karakteristik Target: Kategori diskrit ganda (misal: A, B, C).
   - Contoh: Memprediksi jenis penyakit dari gambar rontgen (Normal, Pneumonia, COVID-19).

C. Multilabel Classification
   - Jumlah Kelas: Lebih dari 2 kelas, TIDAK saling eksklusif.
   - Jumlah Label per Data: Bisa lebih dari 1 label (satu data bisa memiliki banyak kategori sekaligus).
   - Karakteristik Target: Representasi multiple kolom biner (one-hot) atau list kategori.
   - Contoh: Menentukan genre film (satu film bisa bergenre Action DAN Comedy).
"""
    print(teori)

def analisis_dataset(nama, path_file, tipe_klasifikasi, kolom_target, narasi):
    """
    Fungsi untuk membaca dan mengidentifikasi karakteristik dataset menggunakan Pandas.
    Dilengkapi dengan penanganan error jika file belum diunduh.
    """
    print("\n" + "="*80)
    print(f"IDENTIFIKASI DATASET: {nama}".center(80))
    print("="*80)
    print(f"Jenis Klasifikasi : {tipe_klasifikasi}")
    print(f"Target/Label      : {kolom_target}\n")
    print("--- NARASI DATASET ---")
    print(narasi + "\n")
    
    if not os.path.exists(path_file):
        print(f"[!] PERINGATAN: File dataset tidak ditemukan di path '{path_file}'.")
        print("    Penjelasan: Pandas tidak bisa menampilkan output identifikasi karena file .csv fisiknya")
        print("    belum ada di komputer Anda. Silakan unduh dari link sumber lalu letakkan di folder yang sesuai.")
        return

    try:
        # Membaca dataset
        df = pd.read_csv(path_file)
        
        print("--- ANALISIS PANDAS ---")
        print(f"1. Dimensi Data (Baris, Kolom): {df.shape}")
        
        print("\n2. Lima Data Pertama (head):")
        print(df.head())
        
        print("\n3. Informasi Tipe Data dan Missing Values (info):")
        import io
        buffer = io.StringIO()
        df.info(buf=buffer)
        print(buffer.getvalue())
        
        print(f"4. Jumlah Data Duplikat: {df.duplicated().sum()}")
        
        print("\n5. Statistik Deskriptif (describe):")
        print(df.describe(include='all').to_string())
        
        print("\n6. Distribusi Label/Target:")
        if tipe_klasifikasi in ['Binary Classification', 'Multiclass Classification']:
            if isinstance(kolom_target, str) and kolom_target in df.columns:
                distribusi = df[kolom_target].value_counts()
                print(distribusi)
            else:
                print(f"Kolom target '{kolom_target}' tidak ditemukan di dataset.")
                
        elif tipe_klasifikasi == 'Multilabel Classification':
            if isinstance(kolom_target, list):
                valid_cols = [col for col in kolom_target if col in df.columns]
                if valid_cols:
                    print("Jumlah positif untuk setiap label:")
                    print(df[valid_cols].sum())
                else:
                    print("Kolom target multilabel tidak ditemukan di dataset.")
            elif isinstance(kolom_target, str) and kolom_target in df.columns:
                print("Label disimpan dalam satu kolom (kemungkinan dipisahkan koma). 5 variasi terbanyak:")
                print(df[kolom_target].value_counts().head())
                
    except Exception as e:
        print(f"Terjadi kesalahan saat membaca atau menganalisis file: {e}")


def main():
    tampilkan_teori()
    
    daftar_dataset = [
        {
            "nama": "Titanic: Machine Learning from Disaster",
            "path": "./datasets/titanic.csv",
            "tipe": "Binary Classification",
            "target": "Survived",
            "narasi": "Dataset ini digunakan untuk memprediksi keselamatan penumpang kapal Titanic. Data memuat fitur seperti umur, jenis kelamin, dan kelas tiket. Targetnya adalah kolom 'Survived' yang bernilai 0 (tidak selamat) atau 1 (selamat). Karena hanya ada dua kemungkinan kelas target, dataset ini adalah Binary Classification murni."
        },
        {
            "nama": "Breast Cancer Wisconsin (Diagnostic)",
            "path": "./datasets/breast_cancer.csv",
            "tipe": "Binary Classification",
            "target": "diagnosis",
            "narasi": "Dataset medis untuk mengklasifikasikan tumor payudara berdasarkan fitur citra sel (radius, tekstur, dll). Targetnya adalah 'diagnosis' (M = Ganas, B = Jinak). Karakteristik target yang bersifat Yes/No (2 kelas) menjadikannya Binary Classification."
        },
        {
            "nama": "Bank Customer Churn Prediction",
            "path": "./datasets/bank_churn.csv",
            "tipe": "Binary Classification",
            "target": "Exited",
            "narasi": "Digunakan untuk memprediksi apakah nasabah akan meninggalkan bank (churn) atau tidak. Fiturnya meliputi skor kredit, saldo, dan gaji. Target kolom 'Exited' hanya bernilai 1 (keluar) dan 0 (bertahan)."
        },
        {
            "nama": "Iris Species",
            "path": "./datasets/iris.csv",
            "tipe": "Multiclass Classification",
            "target": "Species",
            "narasi": "Dataset klasik ML untuk memprediksi spesies bunga Iris berdasarkan panjang dan lebar kelopak. Target 'Species' memiliki 3 kelas (Setosa, Versicolor, Virginica). Karena lebih dari 2 kelas dan saling eksklusif, ini adalah Multiclass Classification."
        },
        {
            "nama": "Palmer Penguins",
            "path": "./datasets/penguins.csv",
            "tipe": "Multiclass Classification",
            "target": "species",
            "narasi": "Memprediksi spesies pinguin berdasarkan ukuran paruh dan sayap. Target 'species' terdiri dari 3 kelas (Adelie, Chinstrap, Gentoo). Sama seperti Iris, satu pinguin hanya punya 1 spesies pasti."
        },
        {
            "nama": "Mobile Price Classification",
            "path": "./datasets/mobile_price.csv",
            "tipe": "Multiclass Classification",
            "target": "price_range",
            "narasi": "Dataset ini memprediksi rentang harga handphone (HP) berdasarkan spesifikasi hardware seperti RAM, kapasitas baterai, dan resolusi kamera. Targetnya adalah kolom 'price_range' yang memiliki 4 kelas kategori harga (0, 1, 2, dan 3). Karena jumlah kelas lebih dari dua dan setiap HP hanya masuk ke satu rentang harga, ini adalah Multiclass Classification."
        },
        {
            "nama": "Jigsaw Toxic Comment Classification",
            "path": "./datasets/toxic_comments.csv",
            "tipe": "Multilabel Classification",
            "target": ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
            "narasi": "Dataset teks dari Wikipedia untuk mengklasifikasikan tingkat racun komentar. Satu komentar bisa masuk ke beberapa kategori sekaligus (misal: toxic DAN insult). Target dipisah menjadi 6 kolom biner, menjadikannya Multilabel Classification."
        },
        {
            "nama": "Movies Dataset (IMDB Genres)",
            "path": "./datasets/movies.csv",
            "tipe": "Multilabel Classification",
            "target": "genres",
            "narasi": "Dataset metadata film yang mencakup budget, popularitas, dan genre. Target 'genres' seringkali berisi nilai gabungan (misal: 'Action, Thriller'). Karena satu film dapat memiliki lebih dari satu genre, ini termasuk Multilabel Classification."
        },
        {
            "nama": "StackSample: 10% Stack Overflow Q&A",
            "path": "./datasets/stackoverflow_tags.csv",
            "tipe": "Multilabel Classification",
            "target": "Tags",
            "narasi": "Dataset teks berisi isi pertanyaan pemrograman di Stack Overflow. Targetnya adalah 'Tags'. Karena satu pertanyaan bisa di-tag dengan 'Python' DAN 'Pandas' sekaligus, dataset ini sangat cocok untuk Multilabel Classification."
        }
    ]

    for data in daftar_dataset:
        analisis_dataset(
            nama=data["nama"],
            path_file=data["path"],
            tipe_klasifikasi=data["tipe"],
            kolom_target=data["target"],
            narasi=data["narasi"]
        )
        
    print("\n" + "="*80)
    print("KESIMPULAN".center(80))
    print("="*80)
    print("Penggunaan Pandas sangat krusial dalam tahap Exploratory Data Analysis (EDA).")
    print("Dengan method seperti info(), describe(), dan value_counts(), kita dapat mengenali")
    print("struktur data, mendeteksi missing values, serta memahami distribusi label target")
    print("untuk menentukan jenis metrik evaluasi yang tepat nantinya.")

if __name__ == "__main__":
    main()