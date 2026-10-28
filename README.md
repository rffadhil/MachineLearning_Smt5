# Praktikum Machine Learning - Semester 5

Selamat datang di repository GitHub untuk mata kuliah Machine Learning. Repository ini memuat sekumpulan latihan praktikum mingguan dan proyek tugas akhir pembuatan model klasifikasi.

**Identitas Mahasiswa:**
* **Nama:** Rafi Fadhil Amanullah
* **NIM:** 241011400893
* **Program Studi:** Teknik Informatika - Universitas Pamulang

---

## Petunjuk 
Untuk pemeriksaan tugas pembuatan model *Machine Learning* secara keseluruhan, Bapak dapat langsung menuju folder `Tugas_Machine_Learning` atau mengklik tautan pintasan (shortcut) ke file Jupyter Notebook utama di bawah ini:

**[Buka File Tugas Utama: mlbb_match_prediction.ipynb](./Tugas_Machine_Learning/mlbb_match_prediction.ipynb)**

---

## Struktur Direktori

### 1. `/Tugas_Machine_Learning` (Folder Tugas Utama)
Folder ini memuat proyek final: **Prediksi Hasil Pertandingan Mobile Legends Berdasarkan Kombinasi Draft Pick**. 
* **Tujuan:** Memprediksi status pertandingan (Victory/Defeat) murni dari susunan 5 hero tim tanpa melihat mekanik *in-game*.
* **Dataset:** Riwayat *match* pemain *Mythic/Top Global* dari Kaggle (Patch 1.7.68) dengan total >5.000 baris data bersih.
* **Metode:** Melakukan *Data Cleaning* (menghapus atribut *post-match* untuk mencegah *data leakage*), *One-Hot-Encoding* (mengubah nama hero menjadi matriks biner), dan membelah data (*Data Splitting* 80:20).
* **Algoritma:** Model dilatih menggunakan **Random Forest Classifier** dan dievaluasi hingga sukses mencapai **Akurasi 63.88%** (mengungguli *baseline* akurasi pembuat dataset asli di Kaggle).

### 2. `/Latihan` (Folder Praktikum Harian)
Folder ini digunakan sebagai arsip file-file Jupyter Notebook dan *dataset* ringan yang dipakai untuk latihan rutin, eksperimen kode, atau pemahaman sintaks mingguan selama kelas berlangsung.
