# 🚨 Sistem Deteksi Anomali Presensi Pegawai (Early Warning System - EWS)

Aplikasi web berbasis **Flask** dan **Machine Learning** yang dirancang untuk mendeteksi kecurangan (*fraud*) atau anomali presensi pegawai (seperti *fake GPS*, lompatan titik lokasi, dan ketidakcocokan spatio-temporal) menggunakan kombinasi **PyOD (Isolation Forest, LOF, ECOD)**, **Ensemble Majority Voting**, **Geospatial Mapping (Leaflet.js)**, dan **Explainable AI (SHAP)**.

Proyek ini dibangun sebagai prototipe sistem analitis cerdas untuk instansi pemerintah maupun perusahaan swasta yang membutuhkan pengawasan presensi pegawai berbasis data nyata.

---

## 🌟 Fitur Utama Aplikasi

- 📊 **Upload Data Fleksibel**: Mendukung berkas `.xlsx`, `.xls`, dan `.csv`.
- ⚙️ **Preprocessing Otomatis**: Validasi koordinat GPS Indonesia, penghitungan jarak Haversine ke centroid kantor resmi per unit kerja (SKPD), serta pembersihan duplikasi presensi.
- 🤖 **Ensemble Machine Learning**: Menggabungkan 3 algoritma deteksi outlier PyOD:
  - **Isolation Forest**: Isolasi titik data lokasi yang aneh.
  - **Local Outlier Factor (LOF)**: Pengukuran kepadatan lokal antar tetangga lokasi.
  - **ECOD**: Deteksi statistik berbasis fungsi distribusi kumulatif empiris.
- 🧠 **Explainable AI (SHAP)**: Transparansi model AI yang merinci alasan utama (*primary feature contribution*) dan alasan sekunder di balik penetapan status anomali tiap pegawai.
- 🗺️ **Peta Geospasial Interaktif (Leaflet.js)**: Memodelkan sebaran titik presensi, klustering DBSCAN, dan perbandingan spatio-temporal ST-DBSCAN.
- 🎛️ **Hyperparameter Tuning & Grid Search**: Modul eksperimentasi parameter model dengan fitur *checkpoint resume* (`grid_search_checkpoint.pkl`).
- 📥 **Ekspor Laporan Excel**: Mengunduh laporan hasil analisis anomali lengkap ke format `.xlsx` dalam sekali klik.

---

## 📋 Format Data Presensi (Input Required)

Agar aplikasi dapat memproses data presensi dengan akurat, pastikan file yang diunggah memiliki kolom-kolom berikut:

| Kolom | Status | Keterangan |
|---|---|---|
| `karyawan_id` | **Wajib** | ID / NIP Unik Pegawai |
| `id_skpd` | **Wajib** | Kode Unit Kerja / SKPD |
| `tanggal_kirim` | **Wajib** | Timestamp Presensi (`YYYY-MM-DD HH:MM:SS`) |
| `lat` | **Wajib** | Latitude titik presensi |
| `long` | **Wajib** | Longitude titik presensi |
| `jenis` | **Wajib** | Jenis presensi (`M` = Masuk, `P` = Pulang) |
| `approver_status` | Opsional | Status persetujuan atasan (`TERIMA`, `TOLAK`, `Pending`) untuk perbandingan metrik |

---

## 🛠️ Panduan Instalasi & Penggunaan (Untuk Orang Awam)

Ikuti langkah-langkah mudah di bawah ini untuk menjalankan aplikasi di komputer lokal Anda:

### Langkah 1: Pastikan Python Sudah Terinstal
Buka **Command Prompt (CMD)** atau **PowerShell**, lalu ketik:
```bash
python --version
```
*(Direkomendasikan menggunakan Python versi 3.10, 3.11, atau 3.12).*

---

### Langkah 2: Unduh / Clone Proyek Ini
Jika menggunakan Git, jalankan:
```bash
git clone https://github.com/riyosan/siset.git
cd siset
```
Atau ekstrak file ZIP proyek ke dalam satu folder, lalu buka terminal di folder tersebut.

---

### Langkah 3: Buat Lingkungan Virtual (Virtual Environment)
Menyiapkan lingkungan terisolasi agar dependensi tidak mengganggu sistem komputer Anda:

- **Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **Linux / macOS**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
*(Jika berhasil, akan muncul tanda `(.venv)` di awal prompt terminal).*

---

### Langkah 4: Install Dependensi / Library Python
Jalankan perintah berikut untuk menginstal semua kebutuhan library (Flask, PyOD, SHAP, Scikit-Learn, Pandas, dll):
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

### Langkah 5: Jalankan Aplikasi Web
Setelah proses install selesai, ketik:
```bash
python app.py
```

Jika berhasil, terminal akan menampilkan pesan server berjalan:
```text
 * Running on http://127.0.0.1:5000
```

---

### Langkah 6: Buka Aplikasi di Browser
Buka browser favorit Anda (Chrome, Edge, Firefox), lalu akses alamat:
👉 **`http://127.0.0.1:5000`**

1. Masuk ke menu **Upload & Laporan**.
2. Pilih file presensi Anda (`.xlsx` atau `.csv`).
3. Klik tombol **Jalankan Preprocessing & Prediksi**.
4. Jelajahi dashboard **EDA**, **Evaluasi Kinerja**, **SHAP Interpretability**, dan **Peta Geospasial**!

---

## ☁️ Panduan Deployment Gratis ke Render.com

Jika Anda ingin men-deploy aplikasi ini ke cloud agar dapat diakses publik:

1. Buka **[render.com](https://render.com)** dan buat akun gratis.
2. Buat **New Web Service** dan hubungkan ke repositori GitHub `riyosan/siset`.
3. Atur konfigurasi berikut:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
4. Klik **Create Web Service** dan tunggu hingga selesai.

---

## 📚 Referensi Metode & Algoritma

- **Isolation Forest**: *Liu, Ting, Zhou (2008)*.
- **Local Outlier Factor (LOF)**: *Breunig et al. (2000)*.
- **ECOD**: *Li et al. (2022)*.
- **SHAP (SHapley Additive exPlanations)**: *Lundberg & Lee (2017)*.

---
*Dikembangkan oleh Ruth Elvin Harianja — Master of Data Science & Artificial Intelligence (S2 DSAI)*
