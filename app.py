"""
╔══════════════════════════════════════════════════════════════════╗
║  Sistem Deteksi Anomali Presensi Pegawai                        ║
║  Flask Application                                               ║
║                                                                  ║
║  URUTAN PIPELINE PREPROCESSING:                                  ║
║                                                                  ║
║  [ST]  = diadopsi dari web Streamlit analisis absensi            ║
║  [NB]  = dari notebook tesisv2_fixed.ipynb                       ║
║                                                                  ║
║  TAHAP 0A │ Fix Format Desimal               [ST]                ║
║  TAHAP 0B │ Deduplikasi Relasional           [ST]                ║
║  TAHAP 1  │ Mapping & Normalisasi Status     [ST]                ║
║  TAHAP 2  │ Data Cleaning GPS & Missing      [NB Cell 5]         ║
║  TAHAP 3  │ Type Conversion & Label Pseudo   [NB Cell 6]         ║
║  TAHAP 4  │ Feature Engineering Waktu        [ST]                ║
║  TAHAP 5  │ Coordinate Transformation        [NB Cell 7A]        ║
║  TAHAP 6  │ ST-DBSCAN -> Centroid Kantor      [NB Cell 7B]        ║
║  TAHAP 7  │ Feature Geospasial Lanjutan      [ST + NB]           ║
║  TAHAP 8  │ Feature Deviasi Waktu            [ST]                ║
║  TAHAP 9  │ Feature Agregat per Karyawan     [ST]                ║
║  TAHAP 10 │ OHE shift_id                     [NB Cell 7C]        ║
║  TAHAP 11 │ OHE jenis & status_presensi      [NB Cell 8]         ║
║  TAHAP 12 │ Klasifikasi Catatan + OHE        [NB Cell 8.0]       ║
║  TAHAP 13 │ Seleksi Fitur Model              [NB]                ║
║  TAHAP 14 │ Impute NaN + RobustScaler        [NB Cell 9A]        ║
║  TAHAP 15 │ Isolation Forest                 [NB Cell 9B]        ║
║  TAHAP 16 │ Local Outlier Factor             [NB Cell 9C]        ║
║  TAHAP 17 │ ECOD                             [NB Cell 9D]        ║
║  TAHAP 18 │ Ensemble Majority Voting         [NB Cell 9E]        ║
║  TAHAP 19 │ Evaluasi Metrik                  [NB Cell 9F]        ║
║  TAHAP 20 │ SHAP Interpretability            [NB Cell 10]        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, io, re, pickle, warnings, sys, time, threading
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
try:
    import plotly.express as px
    import plotly.utils
except Exception:
    px = None
import json
from werkzeug.utils import secure_filename

# Fix for charmap encoding error on Windows console when printing emojis/arrows
try:
    if hasattr(sys.stdout, 'reconfigure') and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════
#  KONFIGURASI APLIKASI
# ══════════════════════════════════════════════════════════════════
NAMA_INSTANSI  = "Pemerintah Kota / Kabupaten"   # ← ubah sesuai instansi
is_vercel = bool(os.environ.get("VERCEL") or os.environ.get("VERCEL_ENV"))
MODEL_DIR      = "/tmp/models" if is_vercel else "models"
UPLOAD_FOLDER  = "/tmp/uploads" if is_vercel else "uploads"
ALLOWED_EXT    = {"xlsx", "xls", "csv"}
EDA_CACHE_PKL  = os.path.join(MODEL_DIR, "eda_summary.pkl")
EDA_CACHE_JSON = os.path.join(MODEL_DIR, "eda_summary.json")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.secret_key = "anomali-tesis-secret-2025"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB

os.makedirs(MODEL_DIR,     exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Simpan path file pending di variabel server — lebih reliable dari session cookie
# (session cookie bisa hilang jika server restart di tengah proses)
PENDING_FILE = {"path": None}


# ══════════════════════════════════════════════════════════════════
#  KONSTANTA STATUS PRESENSI  [ST]
#  Mapping kode mesin absen -> label deskriptif lengkap
# ══════════════════════════════════════════════════════════════════

# Kode pendek dari mesin absen -> nama status lengkap
STATUS_CODE_MAP = {
    "T1" : "TELAT_MASUK_RINGAN",
    "T2" : "TELAT_MASUK_SEDANG",
    "T3" : "TELAT_MASUK_BERAT",
    "T4" : "TELAT_MASUK_SANGAT_BERAT",
    "TWM": "TEPAT_WAKTU_MASUK",
    "TWP": "TEPAT_WAKTU_PULANG",
    "PC1": "PULANG_CEPAT",
    "PC2": "PULANG_CEPAT_RINGAN",
    "PC3": "PULANG_CEPAT_SEDANG",
    "PC4": "PULANG_CEPAT_BERAT",
}

# Label lama (sistem sebelumnya) -> label baru yang sudah distandarisasi
STATUS_LEGACY_MAP = {
    "PULANG_NORMAL": "TEPAT_WAKTU_PULANG",
    "HADIR"        : "TEPAT_WAKTU_MASUK",
}

# Kumpulan label yang sudah valid (tidak perlu dikonversi lagi)
STATUS_VALID = {
    "TELAT_MASUK_RINGAN", "TELAT_MASUK_SEDANG",
    "TELAT_MASUK_BERAT",  "TELAT_MASUK_SANGAT_BERAT",
    "TEPAT_WAKTU_MASUK",  "TEPAT_WAKTU_PULANG",
    "PULANG_CEPAT",       "PULANG_CEPAT_RINGAN",
    "PULANG_CEPAT_SEDANG","PULANG_CEPAT_BERAT",
}

# Label ambigu — butuh konteks kolom lain untuk di-resolve
STATUS_AMBIGUOUS = {"TELAT", "PULANG"}

# Kategori yang dianggap indisipliner (berat & sedang)
STATUS_BERMASALAH = {
    "TELAT_MASUK_SANGAT_BERAT", "TELAT_MASUK_BERAT", "TELAT_MASUK_SEDANG",
    "PULANG_CEPAT_BERAT",       "PULANG_CEPAT_SEDANG",
}

STATUS_ORDER = [
    'TELAT_MASUK_SANGAT_BERAT', 'PULANG_CEPAT_BERAT',
    'TELAT_MASUK_BERAT', 'PULANG_CEPAT_SEDANG',
    'TELAT_MASUK_SEDANG', 'PULANG_CEPAT_RINGAN', 'PULANG_CEPAT',
    'TELAT_MASUK_RINGAN',
    'TEPAT_WAKTU_MASUK', 'TEPAT_WAKTU_PULANG',
]

STATUS_COLORS = {
    'TELAT_MASUK_SANGAT_BERAT':  '#6e0d0d',
    'TELAT_MASUK_BERAT':         '#c0392b',
    'TELAT_MASUK_SEDANG':        '#e67e22',
    'TELAT_MASUK_RINGAN':        '#d4ac0d',
    'TEPAT_WAKTU_MASUK':         '#27ae60',
    'TEPAT_WAKTU_PULANG':        '#2ecc71',
    'PULANG_CEPAT':              '#f39c12',
    'PULANG_CEPAT_RINGAN':       '#d4ac0d',
    'PULANG_CEPAT_SEDANG':       '#e67e22',
    'PULANG_CEPAT_BERAT':        '#c0392b',
    'UNKNOWN':                   '#95a5a6',
}

STATUS_EMOJI = {
    'TELAT_MASUK_SANGAT_BERAT':  '⛔',
    'TELAT_MASUK_BERAT':         '🔴',
    'TELAT_MASUK_SEDANG':        '🟠',
    'TELAT_MASUK_RINGAN':        '🟡',
    'TEPAT_WAKTU_MASUK':         '🟢',
    'TEPAT_WAKTU_PULANG':        '🟢',
    'PULANG_CEPAT':              '🟡',
    'PULANG_CEPAT_RINGAN':       '🟡',
    'PULANG_CEPAT_SEDANG':       '🟠',
    'PULANG_CEPAT_BERAT':        '🔴',
    'UNKNOWN':                   '⚪',
}

# Jam batas ketepatan waktu
JAM_MASUK_BATAS  = 8.25    # 08:15 -> batas tepat waktu masuk
JAM_PULANG_BATAS = 16.0    # 16:00 -> batas tepat waktu pulang


# ══════════════════════════════════════════════════════════════════
#  KAMUS KLASIFIKASI CATATAN  [NB]
#  Rule-based text classification untuk kolom 'catatan'
# ══════════════════════════════════════════════════════════════════
KAMUS_CATATAN = {
    "Dinas": [
        "dinas","perjalanan dinas","surat perintah tugas","spt","tugas kedinasan",
        "prokopim","melaksanakan tugas","rapat","koordinasi","kordinasi","rakor",
        "musyawarah","pembahasan","sosialisasi","bimtek","rab","hps","kak","apel",
        "diklat","pelatihan","pkp","pka","pim","bpsdm","pantau wilayah",
        "pemantauan wilayah","pengawasan","kunjungan","survei","gotong royong",
        "gladi","upacara","piket","jaga malam","kantor camat","kantor lurah",
        "kantor kecamatan","ktr camat","kntr camat","gedung baru","lapangan merdeka",
        "pkk","absen pagi","absen sore","absen","bekerja dari rumah",
    ],
    "Kendala_Teknis": [
        "error","eror","aplikasi presensi","aplikasi baru","aplikasi lama",
        "presensi baru","presensi lama","tidak bisa absen","gagal absen","gps",
        "map eror","lokasi tidak valid","lokasi invalid","tidak valid","koordinat",
        "sinyal","jaringan","internet","wifi","hp","handphone","ponsel","baterai habis",
    ],
    "Alasan_Pribadi": [
        "sakit","demam","berobat","rs adam malik","adam malik","rumah sakit",
        "puskesmas","dokter","izin","ijin","cuti","dispensasi","macet","terlambat",
        "telat","jalanan macet","terjebak","hujan","banjir","ban bocor","kendaraan",
        "kecelakaan","keluarga sakit","bawa keluarga","orang tua","orgtua","lg berobat",
    ],
}
NILAI_KOSONG_CATATAN = {"tidak_ada", ".", "..", ".....", ",", "-", ""}


# ══════════════════════════════════════════════════════════════════
#  NARASI SHAP — nama fitur teknis -> kalimat Bahasa Indonesia
# ══════════════════════════════════════════════════════════════════
NARASI_FITUR = {
    "jarak_ke_kantor"           : "Lokasi presensi jauh dari kantor",
    "fe_dist_km"                : "Lokasi presensi jauh dari kantor",
    "is_noise"                  : "Lokasi di luar pola umum (noise spasial)",
    "cluster_id"                : "Pola lokasi tidak konsisten",
    "lat_radian"                : "Koordinat lintang tidak wajar",
    "long_radian"               : "Koordinat bujur tidak wajar",
    "lat_kantor_radian"         : "Lokasi kantor referensi tidak sesuai",
    "long_kantor_radian"        : "Lokasi kantor referensi tidak sesuai",
    "jam"                       : "Jam presensi di luar waktu normal",
    "fe_jam_desimal"            : "Jam presensi di luar waktu normal",
    "hari_ke"                   : "Presensi pada hari tidak lazim",
    "fe_weekday_num"            : "Presensi pada hari tidak lazim",
    "fe_is_weekend"             : "Presensi dilakukan di hari weekend",
    "time_seconds"              : "Pola waktu presensi tidak biasa",
    "fe_menit_telat"            : "Keterlambatan masuk yang signifikan",
    "fe_menit_pulang_cepat"     : "Pulang terlalu cepat dari jam kerja",
    "fe_total_indiscipline"     : "Riwayat indisipliner pegawai tinggi",
    "fe_pct_indiscipline"       : "Persentase indisipliner pegawai tinggi",
    "fe_outside_100m"           : "Lokasi presensi di luar radius 100m kantor",
    "fe_very_far_5km"           : "Lokasi presensi sangat jauh dari kantor",
    "fe_dbscan_only"            : "Lokasi sering dipakai absen tapi di hari berbeda (pola mencurigakan)",
    "fe_both_noise"             : "Lokasi tidak biasa — tidak masuk cluster manapun",
    "is_noise_dbscan"           : "Lokasi tidak pernah membentuk cluster spasial (noise DBSCAN)",
    "fe_jarak_masuk_pulang"     : "Jarak antara lokasi masuk dan pulang terlalu jauh (absen lompat)",
    "fe_absen_lompat"           : "Absen masuk dan pulang di lokasi berbeda (>1km)",
    "fe_dist_personal"          : "Lokasi absen jauh dari kebiasaan pegawai ini",
    "fe_drift_lokasi"           : "Perpindahan lokasi absen dari hari sebelumnya (pergeseran mencurigakan)",
    "fe_konsistensi_lokasi"     : "Pola lokasi pegawai tidak konsisten (sering berpindah-pindah)",
    "is_absen_lengkap"          : "Absen tidak lengkap (hanya masuk tanpa pulang, atau sebaliknya)",
    "catatan_Tidak_Ada_Catatan" : "Tidak ada keterangan dari pegawai",
    "catatan_Kendala_Teknis"    : "Pegawai melaporkan kendala teknis GPS/aplikasi",
    "catatan_Alasan_Pribadi"    : "Pegawai menyertakan alasan pribadi",
    "catatan_Dinas"             : "Pegawai mengklaim sedang perjalanan dinas",
    "catatan_Lainnya"           : "Keterangan pegawai tidak terkategori",
}


# ══════════════════════════════════════════════════════════════════
#  FUNGSI HELPER UMUM
# ══════════════════════════════════════════════════════════════════

def get_narasi(fitur):
    """Konversi nama fitur teknis -> narasi bahasa Indonesia untuk UI."""
    if not isinstance(fitur, str):
        return "Tidak diketahui"
    for key, narasi in NARASI_FITUR.items():
        if key in fitur:
            return narasi
    return f"Fitur: {fitur}"




def haversine_scalar(lat1, lon1, lat2, lon2):
    """
    Hitung jarak dua titik GPS — formula Haversine.
    Input: derajat decimal. Output: meter.
    Dipakai per-baris (apply) untuk jarak_ke_kantor.
    """
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Versi vektorisasi NumPy dari Haversine — jauh lebih cepat untuk data besar.
    Input: array/Series derajat. Output: array dalam KM.
    """
    rlat1 = np.radians(np.array(lat1, dtype=float))
    rlat2 = np.radians(np.array(lat2, dtype=float))
    rlon1 = np.radians(np.array(lon1, dtype=float))
    rlon2 = np.radians(np.array(lon2, dtype=float))
    a = (np.sin((rlat2-rlat1)/2)**2
         + np.cos(rlat1)*np.cos(rlat2)*np.sin((rlon2-rlon1)/2)**2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bersihkan_teks(teks):
    """Normalisasi teks: lowercase + strip whitespace berlebih."""
    if not isinstance(teks, str): return ""
    return re.sub(r"\s+", " ", teks.strip().lower())


def klasifikasi_catatan(teks):
    """
    Rule-based klasifikasi kolom 'catatan' berdasarkan kamus keyword.
    Urutan prioritas: Dinas -> Kendala_Teknis -> Alasan_Pribadi -> Lainnya
    Jika kosong/tidak bermakna -> Tidak_Ada_Catatan
    """
    t = bersihkan_teks(teks)
    if not t or t in NILAI_KOSONG_CATATAN or len(t) <= 1:
        return "Tidak_Ada_Catatan"
    for kategori, kata_list in KAMUS_CATATAN.items():
        for kata in kata_list:
            if kata in t:
                return kategori
    return "Lainnya"


# ══════════════════════════════════════════════════════════════════
#  FUNGSI NORMALISASI STATUS PRESENSI  [ST]
# ══════════════════════════════════════════════════════════════════

def map_status_value(val):
    """
    [TAHAP 1 - ST] Normalisasi satu nilai status_presensi:
      T1/T2/T3/T4/TWM/TWP/PC1-4 -> label deskriptif
      Label lama (HADIR, PULANG_NORMAL) -> label baru
      Label valid -> dikembalikan apa adanya
      Tidak dikenali / kosong -> 'UNKNOWN'
    """
    if pd.isna(val) or str(val).strip() == "":
        return "UNKNOWN"
    v = str(val).strip().upper()
    if v in STATUS_CODE_MAP:   return STATUS_CODE_MAP[v]
    if v in STATUS_VALID:      return v
    if v in STATUS_LEGACY_MAP: return STATUS_LEGACY_MAP[v]
    if v in STATUS_AMBIGUOUS:  return v   # resolve di fungsi berikutnya
    return "UNKNOWN"


def resolve_ambiguous_status(df):
    """
    [TAHAP 1 - ST] Resolve status ambigu 'TELAT' dan 'PULANG':

    'TELAT':
      -> Gunakan kolom 'jenis' (M/P) untuk membedakan masuk vs pulang
      -> Jika tidak ada kolom jenis -> default TEPAT_WAKTU_MASUK

    'PULANG':
      -> Gunakan jam_desimal: < 14 -> PULANG_CEPAT_BERAT, ≥ 14 -> PULANG_CEPAT_SEDANG
      -> Jika tidak ada jam_desimal -> default PULANG_CEPAT_SEDANG

    Return: (df_updated, list_log_message)
    """
    logs = []

    # ── Resolve 'TELAT' ──────────────────────────────────────
    mask_telat = df["status_presensi"] == "TELAT"
    if mask_telat.any():
        n = mask_telat.sum()
        if "jenis" in df.columns:
            df.loc[mask_telat & (df["jenis"] == "M"), "status_presensi"] = "TEPAT_WAKTU_MASUK"
            df.loc[mask_telat & (df["jenis"] == "P"), "status_presensi"] = "TEPAT_WAKTU_PULANG"
            # Sisa yang belum ter-resolve (jenis bukan M/P) -> default masuk
            df.loc[df["status_presensi"] == "TELAT", "status_presensi"] = "TEPAT_WAKTU_MASUK"
        else:
            df.loc[mask_telat, "status_presensi"] = "TEPAT_WAKTU_MASUK"
        logs.append(f"Resolve 'TELAT' ({n} baris) -> TEPAT_WAKTU via kolom jenis")

    # ── Resolve 'PULANG' ─────────────────────────────────────
    mask_pulang = df["status_presensi"] == "PULANG"
    if mask_pulang.any():
        n = mask_pulang.sum()
        if "jam_desimal" in df.columns:
            df.loc[mask_pulang & (df["jam_desimal"] >= 14), "status_presensi"] = "PULANG_CEPAT_SEDANG"
            df.loc[mask_pulang & (df["jam_desimal"] <  14), "status_presensi"] = "PULANG_CEPAT_BERAT"
            df.loc[df["status_presensi"] == "PULANG", "status_presensi"]       = "PULANG_CEPAT_SEDANG"
        else:
            df.loc[mask_pulang, "status_presensi"] = "PULANG_CEPAT_SEDANG"
        logs.append(f"Resolve 'PULANG' ({n} baris) -> PULANG_CEPAT via jam_desimal")

    return df, logs


def derive_status_dari_jam(jam_desimal, jenis):
    """
    [TAHAP 1 - ST] Fallback: derive status dari jam_desimal + jenis
    dipakai ketika status_presensi = UNKNOWN atau kolom tidak ada.

    Aturan MASUK (jenis='M'):
      ≤ 08:15 -> TEPAT_WAKTU_MASUK
      08:16–08:45 -> TELAT_MASUK_RINGAN
      08:46–09:15 -> TELAT_MASUK_SEDANG
      09:16–09:45 -> TELAT_MASUK_BERAT
      > 09:45     -> TELAT_MASUK_SANGAT_BERAT

    Aturan PULANG (jenis='P'):
      ≥ 16:00 -> TEPAT_WAKTU_PULANG
      15:30–15:59 -> PULANG_CEPAT
      15:00–15:29 -> PULANG_CEPAT_RINGAN
      14:00–14:59 -> PULANG_CEPAT_SEDANG
      < 14:00     -> PULANG_CEPAT_BERAT
    """
    if jenis == "M":
        if jam_desimal <= 8.25:   return "TEPAT_WAKTU_MASUK"
        elif jam_desimal <= 8.75: return "TELAT_MASUK_RINGAN"
        elif jam_desimal <= 9.25: return "TELAT_MASUK_SEDANG"
        elif jam_desimal <= 9.75: return "TELAT_MASUK_BERAT"
        else:                     return "TELAT_MASUK_SANGAT_BERAT"
    else:
        if jam_desimal >= 16.0:   return "TEPAT_WAKTU_PULANG"
        elif jam_desimal >= 15.5: return "PULANG_CEPAT"
        elif jam_desimal >= 15.0: return "PULANG_CEPAT_RINGAN"
        elif jam_desimal >= 14.0: return "PULANG_CEPAT_SEDANG"
        else:                     return "PULANG_CEPAT_BERAT"


def fix_decimal_columns(df):
    """
    [TAHAP 0A - ST] Deteksi dan perbaiki kolom yang menggunakan koma
    sebagai pemisah desimal (format lokal Indonesia -> format float Python).
    Contoh: '3,14' -> 3.14, '-6,89' -> -6.89

    Strategi:
    1. Kolom dengan nama hint numerik -> langsung konversi
    2. Kolom lain -> sampling: jika >70% berbentuk angka,koma,angka -> konversi
    """
    numeric_hints = [
        "lat", "long", "lat_rad", "long_rad", "office_lat", "office_long",
        "dist_km", "jarak", "jam_desimal", "jam", "menit", "weekday",
        "outside_100m", "very_far", "status_lokasi", "timestamp_num",
    ]
    fixed = []
    for col in df.columns:
        if df[col].dtype != object:
            continue
        if col in numeric_hints:
            try:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", ".", regex=False).str.strip(),
                    errors="coerce")
                fixed.append(col)
            except Exception:
                pass
        else:
            # Auto-detect: cek apakah mayoritas nilai berbentuk digit,koma,digit
            sample = df[col].dropna().head(20).astype(str)
            if sample.str.match(r"^-?\d+,\d+$").mean() > 0.7:
                try:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(",", ".", regex=False),
                        errors="coerce")
                    fixed.append(col)
                except Exception:
                    pass
    return df, fixed


# ══════════════════════════════════════════════════════════════════
#  CEK & LOAD MODEL
# ══════════════════════════════════════════════════════════════════

def models_exist():
    """Cek apakah semua file .pkl model sudah tersimpan di disk."""
    required = [
        "if_model.pkl", "lof_model.pkl", "ecod_model.pkl",
        "scaler.pkl", "shap_values.npy", "expected_val.pkl",
        "feature_names.pkl", "df_hasil.pkl", "X_scaled.npy",
        "results.pkl", "office_centroids.pkl",
    ]
    return all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in required)


def get_last_processed():
    """Ambil timestamp terakhir data diproses dari mtime file df_hasil.pkl."""
    f = os.path.join(MODEL_DIR, "df_hasil.pkl")
    if not os.path.exists(f):
        return None
    import datetime
    ts = os.path.getmtime(f)
    return datetime.datetime.fromtimestamp(ts).strftime("%d %b %Y, %H:%M")


def load_models():
    """
    Load semua model dan data hasil dari disk.
    Dipanggil setiap kali endpoint API membutuhkan data.
    """
    d = MODEL_DIR
    with open(f"{d}/if_model.pkl",         "rb") as f: IF_MODEL         = pickle.load(f)
    with open(f"{d}/lof_model.pkl",        "rb") as f: LOF_MODEL        = pickle.load(f)
    with open(f"{d}/ecod_model.pkl",       "rb") as f: ECOD_MODEL       = pickle.load(f)
    with open(f"{d}/scaler.pkl",           "rb") as f: scaler           = pickle.load(f)
    with open(f"{d}/expected_val.pkl",     "rb") as f: expected_val     = pickle.load(f)
    with open(f"{d}/feature_names.pkl",    "rb") as f: feature_names    = pickle.load(f)
    with open(f"{d}/results.pkl",          "rb") as f: results          = pickle.load(f)
    with open(f"{d}/office_centroids.pkl", "rb") as f: office_centroids = pickle.load(f)
    shap_values = np.load(f"{d}/shap_values.npy", allow_pickle=True)
    X_scaled    = np.load(f"{d}/X_scaled.npy",    allow_pickle=True)
    df          = pd.read_pickle(f"{d}/df_hasil.pkl")
    return (IF_MODEL, LOF_MODEL, ECOD_MODEL, scaler,
            shap_values, expected_val, feature_names,
            results, office_centroids, X_scaled, df)


def load_raw_df():
    """Load df_raw.pkl dari disk."""
    raw_path = os.path.join(MODEL_DIR, "df_raw.pkl")
    if os.path.exists(raw_path):
        try:
            return pd.read_pickle(raw_path)
        except Exception:
            pass
    return None


# ══════════════════════════════════════════════════════════════════
#  PIPELINE UTAMA: PREPROCESSING + PREDIKSI
# ══════════════════════════════════════════════════════════════════

def run_preprocessing_and_prediction(df_raw):
    """
    Menjalankan seluruh pipeline dari data mentah -> model tersimpan.
    Setiap tahap diberi komentar tracking: [TAHAP X] di terminal.

    Return:
        df              — DataFrame hasil lengkap dengan semua kolom
        shap_values     — array SHAP (n_samples × n_features)
        expected_val    — baseline SHAP (mean expected value IF)
        feature_names   — list nama fitur yang masuk model
        results         — dict metrik evaluasi (IF, LOF, ECOD, Ensemble)
        office_centroids— DataFrame centroid kantor per SKPD
        X_scaled        — array fitur setelah RobustScaler
    """
    from sklearn.preprocessing import RobustScaler
    from sklearn.cluster import DBSCAN
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                  average_precision_score, confusion_matrix)
    from pyod.models.ecod import ECOD
    try:
        try:
            import shap
        except Exception:
            shap = None
    except Exception:
        shap = None

    n_raw = len(df_raw)
    n_duplicate_deleted = 0
    n_gps_deleted = 0
    n_missing_gps = 0
    n_out_of_range = 0
    n_null_island = 0

    df = df_raw.copy()   # jangan modifikasi data asli
    if "status_presensi" in df.columns:
        df.drop(columns=["status_presensi"], inplace=True)


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 0A — FIX FORMAT DESIMAL  [ST]                     ║
    # ╚══════════════════════════════════════════════════════════╝
    # Masalah: beberapa export Excel Indonesia menggunakan KOMA
    # sebagai pemisah desimal, bukan titik.
    # Contoh masalah: lat = "0,118" (string) bukan 0.118 (float)
    # Jika tidak diperbaiki -> lat/long tidak bisa dipakai untuk
    # kalkulasi Haversine -> error atau hasil NaN
    df, kolom_fixed = fix_decimal_columns(df)
    if kolom_fixed:
        print(f"[TAHAP 0A] Fix desimal koma->titik pada kolom: {kolom_fixed}")
    else:
        print("[TAHAP 0A] Tidak ada kolom dengan format desimal koma, skip")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 0B — DEDUPLIKASI RELASIONAL  [ST]                 ║
    # ╚══════════════════════════════════════════════════════════╝
    # Masalah umum data mesin absen:
    #   1. Pegawai tap kartu beberapa kali -> double/triple absen
    #   2. Ada record PULANG tapi tidak ada record MASUK di hari sama
    #
    # Strategi penanganan (dengan prioritas approver_status):
    #   • Jika ada 2 absen masuk/pulang di hari yang sama:
    #     - Jika salah satu TERIMA dan lainnya TOLAK -> ambil yang TERIMA
    #     - Jika keduanya sama status -> MASUK ambil paling awal, PULANG ambil paling akhir
    #   • Absen PULANG tanpa pasangan MASUK -> tetap dipertahankan (ditandai is_absen_lengkap=0)
    #   • Absen MASUK tanpa pasangan PULANG -> tetap dipertahankan (ditandai is_absen_lengkap=0)
    #
    # Fitur baru: is_absen_lengkap = 1 jika pegawai punya MASUK dan PULANG di hari yang sama
    if "jenis" in df.columns and "tanggal_kirim" in df.columns:
        # Fix: konversi Excel serial number ke datetime
        # Beberapa export Excel menyimpan tanggal sebagai angka (misal 45903.688 = 2025-09-16 16:31)
        # Deteksi: jika nilai numerik > 40000 dan < 50000, kemungkinan Excel serial
        from datetime import datetime, timedelta
        def fix_excel_date(val):
            if pd.isna(val):
                return val
            # Jika sudah datetime, kembalikan apa adanya
            if isinstance(val, (pd.Timestamp, datetime)):
                return val
            # Jika angka (Excel serial number)
            try:
                num = float(val)
                if 40000 < num < 55000:
                    return datetime(1899, 12, 30) + timedelta(days=num)
            except (ValueError, TypeError):
                pass
            return val

        df["tanggal_kirim"] = df["tanggal_kirim"].apply(fix_excel_date)
        df["tanggal_kirim"] = pd.to_datetime(df["tanggal_kirim"], errors="coerce")
        df = df.dropna(subset=["tanggal_kirim", "karyawan_id", "jenis"])
        df["jenis"] = df["jenis"].astype(str).str.strip().str.upper()

        # Kolom bantu: tanggal saja (tanpa waktu) untuk groupby harian
        df["_tgl"] = df["tanggal_kirim"].dt.date
        n_sebelum  = len(df)

        # Normalisasi approver_status untuk sorting
        has_approver = "approver_status" in df.columns
        if has_approver:
            # Prioritas: TERIMA > Pending > TOLAK (TERIMA diambil duluan)
            apv_priority = {"TERIMA": 0, "Pending": 1, "TOLAK": 2}
            df["_apv_sort"] = (df["approver_status"].astype(str).str.strip()
                               .map(apv_priority).fillna(1).astype(int))
        else:
            df["_apv_sort"] = 0

        # MASUK: prioritas TERIMA, lalu ambil paling awal
        df_masuk = (
            df[df["jenis"] == "M"]
            .sort_values(["_apv_sort", "tanggal_kirim"])
            .drop_duplicates(subset=["karyawan_id", "_tgl"], keep="first")
        )

        # PULANG: prioritas TERIMA, lalu ambil paling akhir
        df_pulang = (
            df[df["jenis"] == "P"]
            .sort_values(["_apv_sort", "tanggal_kirim"], ascending=[True, False])
            .drop_duplicates(subset=["karyawan_id", "_tgl"], keep="first")
        )

        # Gabungkan kembali (TIDAK hapus pulang tanpa masuk atau masuk tanpa pulang)
        df = (pd.concat([df_masuk, df_pulang])
              .sort_values(["karyawan_id", "tanggal_kirim"])
              .reset_index(drop=True))

        # ── Fitur is_absen_lengkap ──
        # 1 jika pegawai punya MASUK dan PULANG di hari yang sama
        masuk_keys = set(zip(df_masuk["karyawan_id"], df_masuk["_tgl"]))
        pulang_keys = set(zip(df_pulang["karyawan_id"], df_pulang["_tgl"]))
        lengkap_keys = masuk_keys & pulang_keys  # intersection

        df["_key_check"] = list(zip(df["karyawan_id"], df["_tgl"]))
        df["is_absen_lengkap"] = df["_key_check"].isin(lengkap_keys).astype(int)

        df.drop(columns=["_tgl", "_apv_sort", "_key_check"], errors="ignore", inplace=True)

        n_dihapus = n_sebelum - len(df)
        n_duplicate_deleted += n_dihapus
        n_tidak_lengkap = (df["is_absen_lengkap"] == 0).sum()
        print(f"[TAHAP 0B] Deduplikasi relasional: {n_sebelum} -> {len(df)} baris "
              f"({n_dihapus} duplikat dihapus, prioritas approver_status=TERIMA)")
        print(f"[TAHAP 0B] Absen tidak lengkap (masuk tanpa pulang / pulang tanpa masuk): "
              f"{n_tidak_lengkap} rekaman ({n_tidak_lengkap/len(df)*100:.1f}%)")
    else:
        df["is_absen_lengkap"] = 1
        print("[TAHAP 0B] Kolom 'jenis' atau 'tanggal_kirim' tidak ada, skip deduplikasi")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 1 — MAPPING & NORMALISASI STATUS PRESENSI  [ST]   ║
    # ╚══════════════════════════════════════════════════════════╝
    print("[TAHAP 1] status_presensi ditakeout sesuai request penelitian.")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 2 — DATA CLEANING GPS & MISSING VALUES  [NB C5]   ║
    # ╚══════════════════════════════════════════════════════════╝
    # Hapus baris dengan koordinat GPS tidak valid:
    #   • NaN lat/long -> tidak bisa dihitung jarak
    #   • Di luar range WGS84 -> koordinat tidak mungkin ada di bumi
    #   • Koordinat (0,0) -> error teknis GPS (Null Island)
    n_sebelum = len(df)
    
    # 1. NaN GPS
    df_no_nan = df.dropna(subset=["lat", "long"])
    n_missing_gps = n_sebelum - len(df_no_nan)
    
    # 2. Out of Range WGS84
    df_in_range = df_no_nan[
        (df_no_nan["lat"]  >= -90)  & (df_no_nan["lat"]  <= 90) &
        (df_no_nan["long"] >= -180) & (df_no_nan["long"] <= 180)
    ]
    n_out_of_range = len(df_no_nan) - len(df_in_range)
    
    # 3. Null Island (0,0)
    df_clean_gps = df_in_range[(df_in_range["lat"] != 0) & (df_in_range["long"] != 0)]
    n_null_island = len(df_in_range) - len(df_clean_gps)
    
    df = df_clean_gps
    n_gps_deleted = n_missing_gps + n_out_of_range + n_null_island
    print(f"[TAHAP 2]  Cleaning GPS: {n_sebelum} -> {len(df)} baris "
          f"({n_gps_deleted} baris koordinat tidak valid dihapus: "
          f"NaN={n_missing_gps}, OutOfRange={n_out_of_range}, NullIsland={n_null_island})")

    # Fill missing pada kolom teks agar tidak error di tahap berikutnya
    for col in ["catatan", "catatan_penolakan", "message"]:
        if col in df.columns:
            df[col] = df[col].fillna("tidak_ada")

    # Fill missing approver_status -> 'Pending' (belum diputuskan atasan)
    if "approver_status" in df.columns:
        df["approver_status"] = df["approver_status"].fillna("Pending")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 3 — TYPE CONVERSION & LABEL PSEUDO  [NB C6]       ║
    # ╚══════════════════════════════════════════════════════════╝
    # Pastikan setiap kolom bertipe data yang tepat sebelum feature engineering.
    # Tipe yang salah (misal lat sebagai string) akan menyebabkan error di NumPy.

    # Timestamp -> datetime64
    # Deteksi dan konversi Excel serial date number (misal 45903.688 → 2025-09-16 16:31)
    # Excel serial date: angka desimal dimana integer = hari sejak 1900-01-01
    if "tanggal_kirim" in df.columns:
        col = df["tanggal_kirim"]
        # Cek apakah ada nilai numerik besar (>40000 = tahun 2009+) yang merupakan Excel serial
        numeric_mask = pd.to_numeric(col, errors="coerce")
        excel_serial_mask = (numeric_mask > 40000) & (numeric_mask < 60000)
        if excel_serial_mask.any():
            n_serial = excel_serial_mask.sum()
            # Konversi Excel serial → datetime
            # Excel epoch: 1899-12-30 (karena bug Excel yang menganggap 1900 leap year)
            excel_epoch = pd.Timestamp("1899-12-30")
            df.loc[excel_serial_mask, "tanggal_kirim"] = (
                excel_epoch + pd.to_timedelta(numeric_mask[excel_serial_mask], unit="D")
            )
            print(f"[TAHAP 3]  Konversi {n_serial} baris Excel serial date → datetime")

    df["tanggal_kirim"] = pd.to_datetime(df["tanggal_kirim"], errors="coerce")
    if "approver_at" in df.columns:
        df["approver_at"] = pd.to_datetime(df["approver_at"], errors="coerce")

    # Koordinat -> float64 (wajib untuk kalkulasi trigonometri)
    df["lat"]  = df["lat"].astype(float)
    df["long"] = df["long"].astype(float)

    # ID-ID -> string (hindari model menganggap ID sebagai angka kontinu)
    for col in ["karyawan_id", "instansi_id", "id_skpd"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "shift_id" in df.columns:
        df["shift_id"] = df["shift_id"].astype(str)

    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 3.5 — DEDUPLIKASI RELASIONAL (RESOLUSI ERROR)      ║
    # ╚══════════════════════════════════════════════════════════╝
    awal_len = len(df)
    df["_tanggal_tmp"] = df["tanggal_kirim"].dt.date

    # 1. Masuk paling AWAL per pegawai per hari
    df_masuk = (df[df["jenis"] == "M"]
                .sort_values("tanggal_kirim")
                .drop_duplicates(subset=["karyawan_id", "_tanggal_tmp"], keep="first"))

    # 2. Pulang paling AKHIR per pegawai per hari
    df_pulang = (df[df["jenis"] == "P"]
                 .sort_values("tanggal_kirim")
                 .drop_duplicates(subset=["karyawan_id", "_tanggal_tmp"], keep="last"))

    # 3. Hapus pulang yang tidak ada masuknya di hari sama
    valid_pairs = df_masuk[["karyawan_id", "_tanggal_tmp"]].drop_duplicates()
    df_pulang   = pd.merge(df_pulang, valid_pairs, on=["karyawan_id", "_tanggal_tmp"], how="inner")

    df = (pd.concat([df_masuk, df_pulang])
              .sort_values(["karyawan_id", "tanggal_kirim"])
              .reset_index(drop=True))

    df = df.drop(columns=["_tanggal_tmp"], errors="ignore")
    akhir_len = len(df)
    n_duplicate_deleted += (awal_len - akhir_len)
    print(f"[TAHAP 3.5] Deduplikasi: {awal_len} -> {akhir_len} baris ({awal_len - akhir_len} duplikat dihapus)")

    # Label pseudo untuk EVALUASI model (bukan fitur input):
    #   TOLAK = anomali (1), TERIMA = normal (0)
    # ⚠️ JANGAN masukkan approver_status ke fitur model -> data leakage!
    # Label lama (approver) tetap disimpan sebagai referensi
    if "approver_status" in df.columns:
        df["label_approver"] = (
            df["approver_status"].map({"TOLAK": 1, "TERIMA": 0}).fillna(0).astype(int)
        )
    else:
        df["label_approver"] = 0

    # label_pseudo akan di-override nanti di Tahap 13 dengan composite label
    df["label_pseudo"] = df["label_approver"]

    print(f"[TAHAP 3]  Label approver -> TOLAK: {df['label_approver'].sum()}, "
          f"Normal: {(df['label_approver']==0).sum()}")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 4 — FEATURE ENGINEERING WAKTU  [ST]               ║
    # ╚══════════════════════════════════════════════════════════╝
    # Ekstrak informasi temporal yang kaya dari kolom tanggal_kirim.
    # Semua kolom baru diprefix 'fe_' agar mudah diidentifikasi.
    # Fitur temporal ini memberi sinyal ke model tentang KAPAN presensi terjadi.

    # ── Fitur jam ────────────────────────────────────────────
    df["fe_jam"]         = df["tanggal_kirim"].dt.hour
    df["fe_menit"]       = df["tanggal_kirim"].dt.minute
    df["fe_detik"]       = df["tanggal_kirim"].dt.second
    # jam_desimal: representasi jam sebagai bilangan desimal
    # contoh: 08:30:00 -> 8.5, 16:45:00 -> 16.75
    df["fe_jam_desimal"] = df["fe_jam"] + df["fe_menit"]/60.0 + df["fe_detik"]/3600.0

    # ── Fitur hari & kalender ─────────────────────────────────
    df["fe_tanggal"]     = df["tanggal_kirim"].dt.date
    df["fe_weekday_num"] = df["tanggal_kirim"].dt.dayofweek   # 0=Senin, 6=Minggu
    df["fe_bulan"]       = df["tanggal_kirim"].dt.month
    df["fe_tahun"]       = df["tanggal_kirim"].dt.year
    # Weekend flag: absen di Sabtu/Minggu -> mencurigakan
    df["fe_is_weekend"]  = (df["fe_weekday_num"] >= 5).astype(int)

    # ── Flag approver ─────────────────────────────────────────
    if "approver_status" in df.columns:
        apv = df["approver_status"].astype(str).str.strip().str.upper()
        df["fe_is_terima"] = apv.str.contains("TERIMA", na=False).astype(int)
        df["fe_is_tolak"]  = apv.str.contains("TOLAK",  na=False).astype(int)

    print(f"[TAHAP 4]  Feature waktu: fe_jam, fe_jam_desimal, fe_weekday_num, "
          f"fe_is_weekend ditambahkan")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 5 — COORDINATE TRANSFORMATION  [NB C7A]           ║
    # ╚══════════════════════════════════════════════════════════╝
    # Konversi koordinat ke RADIAN untuk kompatibilitas DBSCAN Haversine metric.
    # time_seconds = selisih detik dari tanggal paling awal di dataset
    # -> fitur temporal untuk ST-DBSCAN (menggabungkan dimensi waktu ke clustering)

    df["lat_radian"]  = np.radians(df["lat"])
    df["long_radian"] = np.radians(df["long"])
    t_min = df["tanggal_kirim"].min()
    df["time_seconds"] = (df["tanggal_kirim"] - t_min).dt.total_seconds()

    print(f"[TAHAP 5]  lat/long -> radian, time_seconds dihitung "
          f"(range: 0 – {df['time_seconds'].max():.0f} detik)")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 6 — CENTROID KANTOR + ST-DBSCAN  [NB C7B]         ║
    # ╚══════════════════════════════════════════════════════════╝
    # PRIORITAS CENTROID:
    #   1. Jika data sudah punya kolom lat_kantor & long_kantor (dari file
    #      koordinat resmi) -> gunakan langsung, SKIP perhitungan DBSCAN centroid
    #   2. Jika tidak ada -> fallback ke cara lama (cluster terbesar ST-DBSCAN)
    #
    # ST-DBSCAN tetap dijalankan untuk fitur clustering:
    #   - cluster_id, is_noise, fe_dbscan_only, fe_both_noise, is_noise_dbscan
    #   Fitur-fitur ini mendeteksi pola spasial mencurigakan.
    #
    # DBSCAN murni (tanpa temporal) juga dijalankan untuk perbandingan:
    #   Titik yang masuk cluster di DBSCAN tapi TIDAK di ST-DBSCAN
    #   = lokasi sering dipakai absen tapi di hari berbeda-beda (mencurigakan)
    #
    # Parameter:
    #   EPS_SPATIAL  = 100m di bumi ≈ 0.0000157 radian
    #   EPS_TEMPORAL = 24 jam (86400 detik)
    #   TIME_SCALE   = normalisasi waktu ke skala spasial
    #   MIN_SAMPLES  = minimal 5 titik untuk membentuk cluster valid

    EPS_SPATIAL  = 100 / 6371000    # ≈ 100 meter dalam radian
    EPS_TEMPORAL = 24 * 3600        # 86400 detik = 24 jam
    MIN_SAMPLES  = 5
    TIME_SCALE   = EPS_SPATIAL / EPS_TEMPORAL  # normalisasi waktu ke skala spasial

    df["cluster_id"]       = -1   # ST-DBSCAN cluster (-1 = noise)
    df["dbscan_cluster"]   = -1   # DBSCAN murni (tanpa temporal)

    for skpd, group in df.groupby("id_skpd"):
        coords      = group[["lat_radian", "long_radian"]].values
        time_scaled = group["time_seconds"].values.reshape(-1, 1) * TIME_SCALE

        # ── DBSCAN murni (hanya spasial, tanpa waktu) ──
        dbscan_labels = DBSCAN(
            eps=EPS_SPATIAL, min_samples=MIN_SAMPLES, metric="euclidean"
        ).fit_predict(coords)
        df.loc[group.index, "dbscan_cluster"] = dbscan_labels

        # ── ST-DBSCAN (spasial + temporal) ──
        features = np.hstack([coords, time_scaled])
        stdbscan_labels = DBSCAN(
            eps=EPS_SPATIAL, min_samples=MIN_SAMPLES, metric="euclidean"
        ).fit_predict(features)
        df.loc[group.index, "cluster_id"] = stdbscan_labels

    # ── Feature: perbandingan DBSCAN vs ST-DBSCAN ──────────────
    # fe_dbscan_only = 1 jika titik masuk cluster di DBSCAN tapi noise di ST-DBSCAN
    # Artinya: lokasi ini sering dipakai absen (DBSCAN mendeteksi cluster),
    # tapi oleh orang yang datang di hari berbeda-beda (ST-DBSCAN tidak membentuk cluster)
    df["fe_dbscan_only"] = (
        (df["dbscan_cluster"] != -1) & (df["cluster_id"] == -1)
    ).astype(int)

    # fe_both_noise = 1 jika noise di kedua metode (lokasi benar-benar tidak biasa)
    df["fe_both_noise"] = (
        (df["dbscan_cluster"] == -1) & (df["cluster_id"] == -1)
    ).astype(int)

    # is_noise_dbscan = 1 jika noise di DBSCAN murni (lokasi tidak pernah membentuk cluster spasial)
    df["is_noise_dbscan"] = (df["dbscan_cluster"] == -1).astype(int)

    n_dbscan_only = df["fe_dbscan_only"].sum()
    n_both_noise  = df["fe_both_noise"].sum()
    print(f"[TAHAP 6]  DBSCAN vs ST-DBSCAN: "
          f"{n_dbscan_only} titik cluster di DBSCAN tapi noise di ST-DBSCAN (mencurigakan), "
          f"{n_both_noise} titik noise di keduanya")

    # ── Hitung centroid kantor ────────────────────────────────
    # PRIORITAS: Jika data sudah punya kolom lat_kantor & long_kantor
    # (dari file koordinat resmi yang sudah di-merge sebelum upload),
    # gunakan langsung tanpa perlu menghitung dari DBSCAN.
    # DBSCAN tetap jalan di atas untuk fitur clustering (noise detection),
    # tapi TIDAK lagi menentukan lokasi kantor.

    if "lat_kantor" in df.columns and "long_kantor" in df.columns and df["lat_kantor"].notna().any():
        # ── DATA SUDAH PUNYA KOORDINAT KANTOR RESMI ──
        # Pastikan tipe numerik
        df["lat_kantor"]  = pd.to_numeric(df["lat_kantor"], errors="coerce")
        df["long_kantor"] = pd.to_numeric(df["long_kantor"], errors="coerce")

        n_punya = df["lat_kantor"].notna().sum()
        n_kosong = df["lat_kantor"].isna().sum()
        print(f"[TAHAP 6]  Menggunakan lat_kantor/long_kantor dari data upload (koordinat resmi)")
        print(f"[TAHAP 6]  Terisi: {n_punya} baris, Kosong: {n_kosong} baris")

        # Fallback untuk baris yang lat_kantor masih NaN: pakai median per SKPD
        no_centroid = df["lat_kantor"].isna()
        if no_centroid.any():
            src = df[df["lat_kantor"].notna()]
            if len(src) > 0:
                fallback = (
                    src.groupby("id_skpd")[["lat_kantor", "long_kantor"]]
                    .median().reset_index()
                    .rename(columns={"lat_kantor": "lat_kantor_fb", "long_kantor": "long_kantor_fb"})
                )
                df = df.merge(fallback, on="id_skpd", how="left")
                df.loc[no_centroid, "lat_kantor"]  = df.loc[no_centroid, "lat_kantor_fb"]
                df.loc[no_centroid, "long_kantor"] = df.loc[no_centroid, "long_kantor_fb"]
                df.drop(columns=["lat_kantor_fb", "long_kantor_fb"], errors="ignore", inplace=True)
                print(f"[TAHAP 6]  Fallback (median SKPD) untuk {no_centroid.sum()} baris tanpa koordinat resmi")

        # Buat office_centroids DataFrame untuk kompatibilitas (disimpan ke pkl)
        office_centroids = (
            df[df["lat_kantor"].notna()]
            .groupby("id_skpd")[["lat_kantor", "long_kantor"]]
            .first().reset_index()
        )
        # Tambah kolom cluster_id dummy untuk kompatibilitas format
        office_centroids["cluster_id"] = 0

    else:
        # ── FALLBACK: HITUNG CENTROID DARI DBSCAN (cara lama) ──
        print(f"[TAHAP 6]  Kolom lat_kantor/long_kantor tidak ada di data, "
              f"menghitung centroid dari ST-DBSCAN...")

        cluster_sizes = (
            df[df["cluster_id"] != -1]
            .groupby(["id_skpd", "cluster_id"])
            .size().reset_index(name="cluster_size")
        )
        largest_cluster = (
            cluster_sizes.sort_values("cluster_size", ascending=False)
            .drop_duplicates(subset="id_skpd")
            [["id_skpd", "cluster_id"]]
        )
        centroids_raw = (
            df[df["cluster_id"] != -1]
            .groupby(["id_skpd", "cluster_id"])[["lat", "long"]]
            .mean().reset_index()
            .rename(columns={"lat": "lat_kantor", "long": "long_kantor"})
        )
        office_centroids = largest_cluster.merge(centroids_raw, on=["id_skpd", "cluster_id"])

        # Merge centroid ke df utama berdasarkan SKPD
        df = df.merge(
            office_centroids[["id_skpd", "lat_kantor", "long_kantor"]],
            on="id_skpd", how="left"
        )

        # Fallback: SKPD yang tidak punya cluster valid -> pakai median koordinat masuk
        no_centroid = df["lat_kantor"].isna()
        if no_centroid.any():
            src = df[df.get("jenis", pd.Series("M", index=df.index)) == "M"] \
                  if "jenis" in df.columns else df
            fallback = (
                src.groupby("id_skpd")[["lat", "long"]]
                .median().reset_index()
                .rename(columns={"lat": "lat_kantor_fb", "long": "long_kantor_fb"})
            )
            df = df.merge(fallback, on="id_skpd", how="left")
            df.loc[no_centroid, "lat_kantor"]  = df.loc[no_centroid, "lat_kantor_fb"]
            df.loc[no_centroid, "long_kantor"] = df.loc[no_centroid, "long_kantor_fb"]
            df.drop(columns=["lat_kantor_fb", "long_kantor_fb"], errors="ignore", inplace=True)
            print(f"[TAHAP 6]  Fallback centroid (median) untuk {no_centroid.sum()} baris")

    n_cluster = df[df["cluster_id"] != -1]["cluster_id"].nunique()
    n_noise   = (df["cluster_id"] == -1).sum()
    print(f"[TAHAP 6]  ST-DBSCAN selesai: {n_cluster} cluster terbentuk, "
          f"{n_noise} titik noise ({n_noise/len(df)*100:.1f}%)")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 7 — FEATURE GEOSPASIAL LANJUTAN  [ST + NB]        ║
    # ╚══════════════════════════════════════════════════════════╝
    # Menghitung jarak ke kantor dengan dua cara:
    #   1. haversine_scalar (apply per baris) -> hasil meter -> fitur 'jarak_ke_kantor' [NB]
    #   2. haversine_vectorized (NumPy) -> hasil km -> fitur 'fe_dist_km' [ST]
    # Keduanya tetap ada karena SHAP sudah dilatih dengan 'jarak_ke_kantor' dari NB.
    #
    # Zona jarak [ST]: kategorisasi jarak untuk interpretasi mudah
    #   SANGAT_DEKAT (<50m) -> DEKAT (<100m) -> SEDANG (<500m) -> JAUH (<2km) -> dst

    # ── 1. Jarak per baris (meter) -> untuk fitur model NB ────
    def get_dist_m(row):
        if pd.isna(row.get("lat_kantor")) or pd.isna(row.get("long_kantor")):
            return np.nan
        return haversine_scalar(row["lat"], row["long"],
                                row["lat_kantor"], row["long_kantor"])
    df["jarak_ke_kantor"] = df.apply(get_dist_m, axis=1)   # dalam METER

    # ── 2. Jarak vektorisasi (km) -> lebih cepat, untuk flag zona ─
    df["fe_dist_km"] = haversine_vectorized(
        df["lat"].fillna(0),          df["long"].fillna(0),
        df["lat_kantor"].fillna(df["lat"]),
        df["long_kantor"].fillna(df["long"])
    )
    df["fe_dist_m"] = df["fe_dist_km"] * 1000   # km -> meter

    # ── Flag zona jarak [ST] ──────────────────────────────────
    df["fe_dalam_100m"]   = (df["fe_dist_km"] <= 0.1).astype(int)  # di dalam radius normal
    df["fe_outside_100m"] = (df["fe_dist_km"] >  0.1).astype(int)  # di luar radius normal
    df["fe_outside_500m"] = (df["fe_dist_km"] >  0.5).astype(int)  # jauh dari kantor
    df["fe_very_far_5km"] = (df["fe_dist_km"] >  5.0).astype(int)  # sangat jauh -> sangat mencurigakan

    def zona_jarak(d):
        """Label kategori zona jarak untuk kemudahan interpretasi."""
        if d <= 0.05:  return "SANGAT_DEKAT_50m"
        if d <= 0.1:   return "DEKAT_100m"
        if d <= 0.5:   return "SEDANG_500m"
        if d <= 2.0:   return "JAUH_2km"
        if d <= 5.0:   return "SANGAT_JAUH_5km"
        return "EKSTREM_5km_PLUS"

    df["fe_zona_jarak"] = df["fe_dist_km"].apply(zona_jarak)

    # ── Fitur turunan lain dari NB ────────────────────────────
    df["is_noise"]  = (df["cluster_id"] == -1).astype(int)  # 1 jika noise ST-DBSCAN
    df["jam"]       = df["tanggal_kirim"].dt.hour            # alias untuk fitur model NB
    df["hari_ke"]   = df["tanggal_kirim"].dt.dayofweek       # alias untuk fitur model NB

    # Koordinat kantor dalam radian (diperlukan sebagai fitur model)
    df["lat_kantor_radian"]  = np.radians(df["lat_kantor"].fillna(df["lat"]))
    df["long_kantor_radian"] = np.radians(df["long_kantor"].fillna(df["long"]))

    # status_lokasi dari sistem (0=dalam area, 1=di luar) — buat default jika tidak ada
    if "status_lokasi" not in df.columns:
        df["status_lokasi"] = 0

    print(f"[TAHAP 7]  Jarak ke kantor: "
          f"median={df['fe_dist_km'].median():.3f} km, "
          f"luar 100m={df['fe_outside_100m'].sum()} baris "
          f"({df['fe_outside_100m'].mean()*100:.1f}%)")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 7B — FEATURE JARAK MASUK-PULANG (ABSEN LOMPAT)    ║
    # ╚══════════════════════════════════════════════════════════╝
    # Menghitung jarak antara lokasi absen MASUK dan PULANG di hari yang sama.
    # Jika pegawai masuk di titik A tapi pulang di titik B yang jauh,
    # ini mencurigakan (kemungkinan salah satu absen dari lokasi lain).
    #
    # fe_jarak_masuk_pulang = jarak (km) antara titik masuk dan pulang
    # fe_absen_lompat       = 1 jika jarak masuk-pulang > 1 km

    df["fe_jarak_masuk_pulang"] = 0.0
    df["fe_absen_lompat"]       = 0

    if "jenis" in df.columns:
        df["_tgl_join"] = df["tanggal_kirim"].dt.date

        # Ambil koordinat masuk per (karyawan, hari) + jam
        df_m = (df[df["jenis"] == "M"]
                .sort_values("tanggal_kirim")
                .groupby(["karyawan_id", "_tgl_join"])
                .agg(lat_masuk=("lat","first"), long_masuk=("long","first"), jam_masuk=("tanggal_kirim","first"))
                .reset_index())

        # Ambil koordinat pulang per (karyawan, hari) + jam
        df_p = (df[df["jenis"] == "P"]
                .sort_values("tanggal_kirim")
                .groupby(["karyawan_id", "_tgl_join"])
                .agg(lat_pulang=("lat","last"), long_pulang=("long","last"), jam_pulang=("tanggal_kirim","last"))
                .reset_index())

        # Join masuk-pulang
        df_mp = df_m.merge(df_p, on=["karyawan_id", "_tgl_join"], how="inner")

        if len(df_mp) > 0:
            # Hitung jarak masuk-pulang
            df_mp["jarak_mp_km"] = haversine_vectorized(
                df_mp["lat_masuk"], df_mp["long_masuk"],
                df_mp["lat_pulang"], df_mp["long_pulang"]
            )

            # Merge kembali ke df utama (jarak + koordinat masuk & pulang + jam)
            merge_cols = df_mp.set_index(["karyawan_id", "_tgl_join"])[["jarak_mp_km", "lat_masuk", "long_masuk", "lat_pulang", "long_pulang", "jam_masuk", "jam_pulang"]]
            df["_key"] = list(zip(df["karyawan_id"], df["_tgl_join"]))
            df["fe_jarak_masuk_pulang"] = df["_key"].map(merge_cols["jarak_mp_km"]).fillna(0.0).round(4)
            df["lat_masuk"]  = df["_key"].map(merge_cols["lat_masuk"])
            df["long_masuk"] = df["_key"].map(merge_cols["long_masuk"])
            df["lat_pulang"]  = df["_key"].map(merge_cols["lat_pulang"])
            df["long_pulang"] = df["_key"].map(merge_cols["long_pulang"])
            df["jam_masuk_str"]  = df["_key"].map(merge_cols["jam_masuk"]).apply(lambda x: str(x)[:16] if pd.notna(x) else "-")
            df["jam_pulang_str"] = df["_key"].map(merge_cols["jam_pulang"]).apply(lambda x: str(x)[:16] if pd.notna(x) else "-")
            df["fe_absen_lompat"] = (df["fe_jarak_masuk_pulang"] > 1.0).astype(int)
            df.drop(columns=["_key"], errors="ignore", inplace=True)

        df.drop(columns=["_tgl_join"], errors="ignore", inplace=True)

        n_lompat = df["fe_absen_lompat"].sum()
        print(f"[TAHAP 7B] Absen lompat: {n_lompat} rekaman dengan jarak masuk-pulang > 1km "
              f"({n_lompat/len(df)*100:.1f}%)")
    else:
        print("[TAHAP 7B] Kolom 'jenis' tidak ada, skip fitur absen lompat")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 7C — FEATURE DRIFT LOKASI & KONSISTENSI           ║
    # ╚══════════════════════════════════════════════════════════╝
    # Mendeteksi pegawai yang awalnya disiplin (absen di lokasi tugas)
    # lalu bergeser ke lokasi lain (misal rumah).
    #
    # fe_dist_personal     = jarak (km) ke centroid personal pegawai
    # fe_drift_lokasi      = jarak (km) ke lokasi absen sebelumnya (hari berbeda)
    # fe_konsistensi_lokasi = std deviasi jarak antar rekaman pegawai (makin tinggi = makin tidak konsisten)

    # ── Centroid personal per pegawai ──
    personal_centroid = (
        df.groupby("karyawan_id")[["lat", "long"]]
        .mean().reset_index()
        .rename(columns={"lat": "lat_personal", "long": "long_personal"})
    )
    df = df.merge(personal_centroid, on="karyawan_id", how="left")
    df["fe_dist_personal"] = haversine_vectorized(
        df["lat"], df["long"],
        df["lat_personal"].fillna(df["lat"]),
        df["long_personal"].fillna(df["long"])
    ).round(4)
    df.drop(columns=["lat_personal", "long_personal"], errors="ignore", inplace=True)

    # ── Drift lokasi: jarak ke rekaman sebelumnya per pegawai ──
    df_sorted = df.sort_values(["karyawan_id", "tanggal_kirim"]).reset_index(drop=True)
    df_sorted["_prev_lat"]  = df_sorted.groupby("karyawan_id")["lat"].shift(1)
    df_sorted["_prev_long"] = df_sorted.groupby("karyawan_id")["long"].shift(1)

    mask_has_prev = df_sorted["_prev_lat"].notna()
    df_sorted["fe_drift_lokasi"] = 0.0
    if mask_has_prev.any():
        df_sorted.loc[mask_has_prev, "fe_drift_lokasi"] = haversine_vectorized(
            df_sorted.loc[mask_has_prev, "lat"],
            df_sorted.loc[mask_has_prev, "long"],
            df_sorted.loc[mask_has_prev, "_prev_lat"],
            df_sorted.loc[mask_has_prev, "_prev_long"]
        )
    df_sorted["fe_drift_lokasi"] = df_sorted["fe_drift_lokasi"].round(4)
    df_sorted.drop(columns=["_prev_lat", "_prev_long"], errors="ignore", inplace=True)

    # ── Konsistensi lokasi: std jarak ke centroid per pegawai ──
    konsistensi = (
        df_sorted.groupby("karyawan_id")["fe_dist_personal"]
        .std().fillna(0).reset_index()
        .rename(columns={"fe_dist_personal": "fe_konsistensi_lokasi"})
    )
    df_sorted = df_sorted.merge(konsistensi, on="karyawan_id", how="left")
    df_sorted["fe_konsistensi_lokasi"] = df_sorted["fe_konsistensi_lokasi"].round(4)

    # Kembalikan ke df utama (urutan asli)
    df = df_sorted.sort_index().reset_index(drop=True) if df_sorted.index.is_monotonic_increasing else df_sorted

    n_drift = (df["fe_drift_lokasi"] > 1.0).sum()
    print(f"[TAHAP 7C] Drift lokasi: {n_drift} rekaman dengan drift > 1km, "
          f"median dist_personal={df['fe_dist_personal'].median():.3f} km")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 8 — FEATURE DEVIASI WAKTU  [ST]                   ║
    # ╚══════════════════════════════════════════════════════════╝
    # Hitung berapa menit pegawai terlambat masuk atau terlalu cepat pulang.
    # Fitur ini memberi sinyal kuat ke model tentang deviasi jam kerja normal.
    #
    # Aturan:
    #   fe_menit_telat        = (jam_desimal - 08:15) × 60   jika jenis=M dan > 0
    #   fe_menit_pulang_cepat = (16:00 - jam_desimal) × 60   jika jenis=P dan > 0
    #
    # Contoh:
    #   Masuk jam 09:00 -> fe_menit_telat = (9.0 - 8.25) × 60 = 45 menit
    #   Pulang jam 15:00 -> fe_menit_pulang_cepat = (16.0 - 15.0) × 60 = 60 menit

    if "jenis" in df.columns and "fe_jam_desimal" in df.columns:
        def hitung_menit_telat(row):
            """Menit keterlambatan masuk. Nol jika tepat waktu atau jenis Pulang."""
            if row["jenis"] == "M":
                delta = (row["fe_jam_desimal"] - JAM_MASUK_BATAS) * 60
                return max(round(delta, 1), 0)
            return 0

        def hitung_menit_cepat(row):
            """Menit terlalu cepat pulang. Nol jika tepat waktu atau jenis Masuk."""
            if row["jenis"] == "P":
                delta = (JAM_PULANG_BATAS - row["fe_jam_desimal"]) * 60
                return max(round(delta, 1), 0)
            return 0

        df["fe_menit_telat"]        = df.apply(hitung_menit_telat, axis=1)
        df["fe_menit_pulang_cepat"] = df.apply(hitung_menit_cepat, axis=1)
        print(f"[TAHAP 8]  Deviasi waktu: "
              f"rata-rata telat={df['fe_menit_telat'].mean():.1f} mnt, "
              f"rata-rata cepat pulang={df['fe_menit_pulang_cepat'].mean():.1f} mnt")
    else:
        df["fe_menit_telat"]        = 0
        df["fe_menit_pulang_cepat"] = 0
        print("[TAHAP 8]  Kolom 'jenis' tidak ada, fe_menit_telat/cepat diisi 0")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 9 — FEATURE AGREGAT PER KARYAWAN  [ST]            ║
    # ╚══════════════════════════════════════════════════════════╝
    print("[TAHAP 9] Skip agregat indisipliner (status_presensi ditakeout)")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 10 — OHE SHIFT_ID  [NB C7C]                       ║
    # ╚══════════════════════════════════════════════════════════╝
    # shift_id sudah dikonversi ke string (TAHAP 3) -> perlu OHE agar
    # bisa masuk model yang hanya menerima input numerik.
    # Kolom string asli shift_id dihapus setelah encoding.
    # Contoh: shift_id='PAGI' -> shift_PAGI=1, shift_SORE=0, shift_MALAM=0
    if "shift_id" in df.columns:
        shift_ohe  = pd.get_dummies(df["shift_id"], prefix="shift", dtype=int)
        df         = pd.concat([df, shift_ohe], axis=1)
        df.drop(columns=["shift_id"], inplace=True)
        shift_cols = [c for c in df.columns if c.startswith("shift_")]
        print(f"[TAHAP 10] OHE shift_id -> {len(shift_cols)} kolom: {shift_cols}")
    else:
        print("[TAHAP 10] Kolom shift_id tidak ada, skip OHE shift")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 11 — OHE JENIS & STATUS_PRESENSI  [NB C8]         ║
    # ╚══════════════════════════════════════════════════════════╝
    # One-Hot Encoding kolom kategorikal jenis (M/P) dan status_presensi.
    # drop_first=True menghindari multikolinearitas (dummy variable trap).
    #
    # ⚠️ approver_status TIDAK di-encode di sini!
    #    approver_status adalah sumber label_pseudo -> jika masuk model = DATA LEAKAGE
    cat_cols = [c for c in ["jenis"] if c in df.columns]
    if cat_cols:
        df_encoded = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols, drop_first=True)
        new_enc    = [c for c in df_encoded.columns
                      if any(c.startswith(p + "_") for p in cat_cols)]
        df_encoded[new_enc] = df_encoded[new_enc].astype(int)
        df = df_encoded.copy()
        print(f"[TAHAP 11] OHE {cat_cols} -> {len(new_enc)} kolom baru")
    else:
        print("[TAHAP 11] Tidak ada kolom kategorikal untuk di-OHE, skip")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 12 — KLASIFIKASI CATATAN + OHE  [NB C8.0]         ║
    # ╚══════════════════════════════════════════════════════════╝
    # Rule-based text classification pada kolom 'catatan':
    #   • Cocokkan keyword dari kamus -> kategori
    #   • Urutan: Dinas -> Kendala_Teknis -> Alasan_Pribadi -> Lainnya
    #   • Kosong/tidak bermakna -> Tidak_Ada_Catatan
    # Setelah klasifikasi -> OHE agar bisa masuk model numerik.
    # Kolom teks asli dihapus setelah encoding (tidak diperlukan model).
    if "catatan" in df.columns:
        df["kategori_catatan"] = df["catatan"].apply(klasifikasi_catatan)
        dist_cat = df["kategori_catatan"].value_counts().to_dict()
        print(f"[TAHAP 12] Distribusi kategori catatan: {dist_cat}")

        ohe_cat = pd.get_dummies(df["kategori_catatan"], prefix="catatan", dtype=int)
        df      = pd.concat([df, ohe_cat], axis=1)
        df.drop(columns=["catatan", "kategori_catatan"], errors="ignore", inplace=True)
    else:
        print("[TAHAP 12] Kolom 'catatan' tidak ada, skip klasifikasi")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 13 — SELEKSI FITUR MODEL  [NB]                    ║
    # ╚══════════════════════════════════════════════════════════╝
    # Pilih subset kolom sebagai input model — hanya fitur yang relevan.
    #
    # Kolom yang TIDAK diikutsertakan:
    #   • karyawan_id, id_skpd -> ID teknis, bukan fitur perilaku
    #   • tanggal_kirim -> sudah diekstrak ke fitur waktu turunan
    #   • label_pseudo, approver_status -> data leakage
    #   • fe_nama_hari, fe_zona_jarak, fe_tanggal -> string, tidak bisa masuk model
    #   • kolom teks mentah (message, imei) -> sudah diekstrak

    FITUR_MODEL = [
        # ── Spasial ─────────────────────────────────────────
        "lat_radian",            # koordinat lintang dalam radian
        "long_radian",           # koordinat bujur dalam radian
        "lat_kantor_radian",     # koordinat kantor dalam radian
        "long_kantor_radian",    # koordinat kantor dalam radian
        "jarak_ke_kantor",       # jarak haversine scalar ke kantor (meter) [NB]
        "status_lokasi",         # status lokasi dari sistem (0/1)
        "is_noise",              # flag: noise ST-DBSCAN
        "fe_dbscan_only",       # 1 jika cluster di DBSCAN tapi noise di ST-DBSCAN
        "fe_both_noise",        # 1 jika noise di kedua metode
        "is_noise_dbscan",      # 1 jika noise di DBSCAN murni
        "fe_jarak_masuk_pulang", # jarak (km) antara lokasi masuk dan pulang hari yang sama
        "fe_absen_lompat",      # 1 jika jarak masuk-pulang > 1km
        "fe_dist_personal",     # jarak (km) ke centroid personal pegawai
        "fe_drift_lokasi",      # jarak (km) ke lokasi absen sebelumnya
        "fe_konsistensi_lokasi", # std deviasi jarak — makin tinggi makin tidak konsisten
        "is_absen_lengkap",     # 0 jika hanya masuk tanpa pulang atau sebaliknya

        # ── Temporal ────────────────────────────────────────
        "time_seconds",          # detik sejak tanggal pertama dataset [NB]
        "jam",                   # jam (0-23) — alias dari fe_jam [NB]
        "hari_ke",               # hari dalam minggu [NB]

        # ── Kategori Catatan (OHE) ────────────────────────────
        "catatan_Dinas",
        "catatan_Kendala_Teknis",
        "catatan_Alasan_Pribadi",
        "catatan_Lainnya",
        "catatan_Tidak_Ada_Catatan",
    ]

    # Tambahkan kolom OHE dinamis dari jenis_
    ohe_prefix = ["jenis_"]
    FITUR_MODEL += [c for c in df.columns if any(c.startswith(p) for p in ohe_prefix)]

    # Tambahkan kolom OHE shift_ — DIHAPUS (tidak berpengaruh signifikan)
    # FITUR_MODEL += [c for c in df.columns if c.startswith("shift_")]

    # Safety check: hanya ambil yang benar-benar ada di df
    FITUR_MODEL = [c for c in FITUR_MODEL if c in df.columns]

    # Deduplikasi (jaga-jaga nama ganda)
    seen = set()
    FITUR_MODEL = [x for x in FITUR_MODEL if not (x in seen or seen.add(x))]

    df_model  = df[FITUR_MODEL].copy()

    # ── Composite Pseudo-Label ──────────────────────────────────
    # Menggantikan label approver (TOLAK/TERIMA) dengan label berbasis
    # sinyal anomali yang terukur secara objektif.
    # Rekaman dianggap pseudo-anomali jika memenuhi ≥2 dari kondisi:
    composite_signals = pd.DataFrame(index=df.index)
    composite_signals["sig_dbscan_only"]  = (df["fe_dbscan_only"] == 1).astype(int) if "fe_dbscan_only" in df.columns else 0
    composite_signals["sig_jauh_kantor"]  = (df["fe_dist_km"] > 5).astype(int) if "fe_dist_km" in df.columns else 0
    composite_signals["sig_absen_lompat"] = (df["fe_absen_lompat"] == 1).astype(int) if "fe_absen_lompat" in df.columns else 0
    composite_signals["sig_tidak_lengkap"]= (df["is_absen_lengkap"] == 0).astype(int) if "is_absen_lengkap" in df.columns else 0
    composite_signals["sig_drift_besar"]  = (df["fe_drift_lokasi"] > 3).astype(int) if "fe_drift_lokasi" in df.columns else 0

    df["composite_score"] = composite_signals.sum(axis=1)
    df["label_pseudo"]    = (df["composite_score"] >= 2).astype(int)

    n_composite = df["label_pseudo"].sum()
    print(f"[TAHAP 13] Composite pseudo-label: {n_composite} anomali "
          f"({n_composite/len(df)*100:.2f}%) — berbasis ≥2 sinyal objektif")

    y_true    = df["label_pseudo"].values   # untuk evaluasi saja, bukan input model

    print(f"[TAHAP 13] {len(FITUR_MODEL)} fitur dipilih untuk model")
    print(f"           Contoh fitur: {FITUR_MODEL[:6]} ... dst")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 14 — IMPUTE NaN + ROBUSTSCALER  [NB C9A]          ║
    # ╚══════════════════════════════════════════════════════════╝
    # Imputasi NaN dengan median per kolom (robust terhadap outlier).
    # NaN bisa muncul jika SKPD tidak punya centroid kantor (noise semua).
    #
    # RobustScaler dipilih karena:
    #   • Data presensi memiliki banyak outlier ekstrem (pegawai sangat jauh)
    #   • RobustScaler menggunakan IQR (bukan mean/std) -> lebih tahan outlier
    #   • StandardScaler/MinMaxScaler akan terdistorsi oleh outlier ekstrem

    cols_impute = [
        "jarak_ke_kantor", "fe_dist_km", "fe_dist_m",
        "lat_kantor_radian", "long_kantor_radian"
    ]
    for col in cols_impute:
        if col in df_model.columns:
            median_val = df_model[col].median()
            n_nan      = df_model[col].isna().sum()
            if n_nan > 0:
                df_model[col] = df_model[col].fillna(median_val)
                print(f"[TAHAP 14] Impute {col}: {n_nan} NaN -> median {median_val:.4f}")

    scaler        = RobustScaler()
    X_scaled      = scaler.fit_transform(df_model)
    feature_names = list(df_model.columns)

    print(f"[TAHAP 14] RobustScaler selesai. X_scaled shape: {X_scaled.shape}")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 15 — ISOLATION FOREST  [NB C9B]                   ║
    # ╚══════════════════════════════════════════════════════════╝
    # Isolation Forest: anomaly detection berbasis pohon keputusan acak.
    # Prinsip: outlier lebih mudah diisolasi (butuh sedikit split) daripada normal.
    #
    # Parameter:
    #   n_estimators=200   -> 200 pohon (lebih banyak = lebih stabil, lebih lambat)
    #   contamination=0.015 -> asumsi 1.5% data adalah anomali (hasil fine-tuning)
    #   random_state=42    -> reproducibility hasil
    #
    # Skor dinormalisasi min-max ke [0,1]:
    #   0 = paling normal, 1 = paling anomali

    IF_MODEL = IsolationForest(n_estimators=200, contamination=0.015, random_state=42)
    IF_MODEL.fit(X_scaled)

    if_pred  = (IF_MODEL.predict(X_scaled) == -1).astype(int)
    if_score = -IF_MODEL.score_samples(X_scaled)
    if_score = (if_score - if_score.min()) / (if_score.max() - if_score.min() + 1e-9)

    df["if_pred"]  = if_pred
    df["if_score"] = if_score.round(4)
    print(f"[TAHAP 15] Isolation Forest: {if_pred.sum()} anomali "
          f"({if_pred.sum()/len(df)*100:.1f}%)")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 16 — LOCAL OUTLIER FACTOR  [NB C9C]               ║
    # ╚══════════════════════════════════════════════════════════╝
    # LOF: mengukur kepadatan lokal setiap titik vs tetangganya.
    # Titik dengan kepadatan lokal jauh lebih rendah -> outlier.
    #
    # ⚠️ novelty=False -> LOF hanya bisa dipakai pada data training
    #    (tidak bisa predict data baru tanpa refit dari awal)
    #
    # Parameter:
    #   n_neighbors=70     -> jumlah tetangga untuk estimasi kepadatan (hasil fine-tuning)
    #   contamination=0.005 -> proporsi anomali yang diharapkan (hasil fine-tuning)

    LOF_MODEL = LocalOutlierFactor(n_neighbors=70, contamination=0.005, novelty=False)
    lof_pred  = (LOF_MODEL.fit_predict(X_scaled) == -1).astype(int)
    lof_score = -LOF_MODEL.negative_outlier_factor_
    lof_score = (lof_score - lof_score.min()) / (lof_score.max() - lof_score.min() + 1e-9)

    df["lof_pred"]  = lof_pred
    df["lof_score"] = lof_score.round(4)
    print(f"[TAHAP 16] LOF: {lof_pred.sum()} anomali ({lof_pred.sum()/len(df)*100:.1f}%)")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 17 — ECOD  [NB C9D]                               ║
    # ╚══════════════════════════════════════════════════════════╝
    # ECOD: Empirical Cumulative distribution-based Outlier Detection.
    # Mendeteksi outlier menggunakan distribusi kumulatif empiris per fitur.
    # Keunggulan: bebas asumsi distribusi, sangat efisien O(n log n).

    ECOD_MODEL = ECOD(contamination=0.005)
    ECOD_MODEL.fit(X_scaled)

    ecod_pred  = ECOD_MODEL.labels_
    ecod_score = ECOD_MODEL.decision_scores_
    ecod_score = (ecod_score - ecod_score.min()) / (ecod_score.max() - ecod_score.min() + 1e-9)

    df["ecod_pred"]  = ecod_pred
    df["ecod_score"] = ecod_score.round(4)
    print(f"[TAHAP 17] ECOD: {ecod_pred.sum()} anomali ({ecod_pred.sum()/len(df)*100:.1f}%)")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 18 — ENSEMBLE MAJORITY VOTING  [NB C9E]           ║
    # ╚══════════════════════════════════════════════════════════╝
    # Keputusan akhir: ANOMALI jika minimal 2 dari 3 algoritma setuju.
    # Ensemble mengurangi false positive dari masing-masing algoritma.
    #
    # vote_count = 0 -> semua normal -> pasti normal
    # vote_count = 1 -> satu algoritma mendeteksi -> NORMAL (bukti masih lemah)
    # vote_count = 2 -> dua setuju anomali -> ANOMALI (bukti lebih kuat)
    # vote_count = 3 -> semua setuju anomali -> ANOMALI (paling kuat)
    #
    # ensemble_score = rata-rata ketiga skor ternormalisasi
    # Semakin tinggi skor -> semakin mencurigakan

    df["vote_count"]     = df["if_pred"] + df["lof_pred"] + df["ecod_pred"]
    df["anomali_final"]  = (df["vote_count"] >= 2).astype(int)
    df["ensemble_score"] = (
        (df["if_score"] + df["lof_score"] + df["ecod_score"]) / 3
    ).round(4)

    n_anomali  = df["anomali_final"].sum()
    pct_anomali = n_anomali / len(df) * 100
    print(f"[TAHAP 18] Ensemble voting: {n_anomali} anomali final "
          f"({pct_anomali:.1f}%) | "
          f"Vote=3: {(df['vote_count']==3).sum()}, "
          f"Vote=2: {(df['vote_count']==2).sum()}, "
          f"Vote=1: {(df['vote_count']==1).sum()}")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 19 — EVALUASI METRIK  [NB C9F]                    ║
    # ╚══════════════════════════════════════════════════════════╝
    # Evaluasi kinerja setiap algoritma + ensemble terhadap composite pseudo-label.
    # label_pseudo dibentuk di Tahap 13 dari minimal dua sinyal objektif.
    #
    # ⚠️ AUC-PR (average_precision_score) digunakan, BUKAN AUC-ROC.
    #    Alasan: data imbalanced (anomali << normal).
    #    AUC-PR lebih informatif pada kasus class imbalance.
    #
    # Metrik yang dihitung:
    #   Precision = TP / (TP + FP) -> seberapa tepat prediksi anomali
    #   Recall    = TP / (TP + FN) -> seberapa banyak anomali terdeteksi
    #   F1-Score  = harmonic mean Precision & Recall
    #   AUC-PR    = area under Precision-Recall curve

    results    = {}
    eval_pairs = {
        "IF"      : (if_pred,                    if_score),
        "LOF"     : (lof_pred,                   lof_score),
        "ECOD"    : (ecod_pred,                  ecod_score),
        "Ensemble": (df["anomali_final"].values,  df["ensemble_score"].values),
    }
    for name, (pred, score) in eval_pairs.items():
        cm           = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        results[name] = {
            "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
            "recall"   : round(float(recall_score(y_true,    pred, zero_division=0)), 4),
            "f1"       : round(float(f1_score(y_true,        pred, zero_division=0)), 4),
            "auc"      : round(float(average_precision_score(y_true, score)), 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }
        print(f"[TAHAP 19] {name:8s}: Precision={results[name]['precision']:.4f} "
              f"Recall={results[name]['recall']:.4f} "
              f"F1={results[name]['f1']:.4f} "
              f"AUC-PR={results[name]['auc']:.4f}")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ TAHAP 20 — SHAP INTERPRETABILITY  [NB C10]              ║
    # ╚══════════════════════════════════════════════════════════╝
    # SHAP menjelaskan MENGAPA setiap rekaman dianggap anomali.
    # TreeExplainer digunakan karena IF adalah model berbasis pohon.
    #
    # Output per rekaman:
    #   shap_top1_fitur -> fitur dengan kontribusi SHAP terbesar (|nilai|)
    #   shap_top2_fitur -> fitur kedua terbesar
    #   shap_top1_nilai -> besar kontribusi fitur pertama
    #   shap_top2_nilai -> besar kontribusi fitur kedua
    #   alasan_utama    -> narasi bahasa Indonesia dari shap_top1_fitur
    #   alasan_kedua    -> narasi bahasa Indonesia dari shap_top2_fitur

    explainer    = shap.TreeExplainer(IF_MODEL)
    shap_values  = explainer.shap_values(X_scaled)
    expected_val = float(np.mean(explainer.expected_value))

    shap_abs = np.abs(shap_values)
    top1_idx = shap_abs.argmax(axis=1)                     # fitur kontribusi terbesar
    top2_idx = np.argsort(shap_abs, axis=1)[:, -2]        # fitur kontribusi kedua

    df["shap_top1_fitur"] = [feature_names[i] for i in top1_idx]
    df["shap_top1_nilai"] = shap_values[np.arange(len(shap_values)), top1_idx].round(4)
    df["shap_top2_fitur"] = [feature_names[i] for i in top2_idx]
    df["shap_top2_nilai"] = shap_values[np.arange(len(shap_values)), top2_idx].round(4)

    # Konversi nama fitur teknis -> narasi bahasa Indonesia

    df["alasan_utama"]   = df["shap_top1_fitur"].apply(get_narasi)
    df["alasan_kedua"]   = df["shap_top2_fitur"].apply(get_narasi)

    print(f"[TAHAP 20] SHAP selesai. Expected value IF: {expected_val:.4f}")
    print(f"           Top fitur terbanyak: "
          f"{pd.Series(df['shap_top1_fitur']).value_counts().head(3).to_dict()}")


    # ╔══════════════════════════════════════════════════════════╗
    # ║ SIMPAN SEMUA KE DISK  [models/*.pkl]                    ║
    # ╚══════════════════════════════════════════════════════════╝
    # Setelah tersimpan -> app tidak perlu ulang training saat restart.
    # Cukup panggil load_models() di setiap request API.

    d = MODEL_DIR
    with open(f"{d}/if_model.pkl",         "wb") as f: pickle.dump(IF_MODEL,         f)
    with open(f"{d}/lof_model.pkl",        "wb") as f: pickle.dump(LOF_MODEL,        f)
    with open(f"{d}/ecod_model.pkl",       "wb") as f: pickle.dump(ECOD_MODEL,       f)
    with open(f"{d}/scaler.pkl",           "wb") as f: pickle.dump(scaler,           f)
    with open(f"{d}/expected_val.pkl",     "wb") as f: pickle.dump(expected_val,     f)
    with open(f"{d}/feature_names.pkl",    "wb") as f: pickle.dump(feature_names,    f)
    with open(f"{d}/results.pkl",          "wb") as f: pickle.dump(results,          f)
    with open(f"{d}/office_centroids.pkl", "wb") as f: pickle.dump(office_centroids, f)
    np.save(f"{d}/shap_values.npy", shap_values)
    np.save(f"{d}/X_scaled.npy",    X_scaled)
    df.to_pickle(f"{d}/df_hasil.pkl")

    # Simpan statistik preprocessing ke disk
    import json
    stats_data = {
        "n_raw": int(n_raw),
        "n_duplicate_deleted": int(n_duplicate_deleted),
        "n_gps_deleted": int(n_gps_deleted),
        "n_missing_gps": int(n_missing_gps),
        "n_out_of_range": int(n_out_of_range),
        "n_null_island": int(n_null_island),
        "n_clean": int(len(df))
    }
    with open(os.path.join(d, "preprocess_stats.json"), "w") as sf:
        json.dump(stats_data, sf)

    # Hapus cache SHAP lama (akan dihitung ulang saat diakses)
    for m in ["if", "lof", "ecod", "ensemble"]:
        cache_f = os.path.join(d, f"shap_cache_{m}.pkl")
        if os.path.exists(cache_f):
            os.remove(cache_f)

    print(f"\n[SELESAI]  Semua model tersimpan di '{MODEL_DIR}/'")
    print(f"           Total baris: {len(df)} | "
          f"Anomali final: {df['anomali_final'].sum()} | "
          f"Normal: {(df['anomali_final']==0).sum()}")

    return df, shap_values, expected_val, feature_names, results, office_centroids, X_scaled


# ══════════════════════════════════════════════════════════════════
#  HELPER: SERIALISASI DATA -> JSON-SAFE
# ══════════════════════════════════════════════════════════════════

def df_to_safe(df, cols=None):
    """
    Konversi DataFrame -> list of dict yang aman untuk jsonify().
    - datetime kolom -> string ISO
    - NaN, inf, -inf -> None (JSON null)
    """
    sub = df[cols].copy() if cols else df.copy()
    for col in sub.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        sub[col] = sub[col].astype(str)
    return sub.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient="records")


# ══════════════════════════════════════════════════════════════════
#  BUILDER DATA PER HALAMAN
# ══════════════════════════════════════════════════════════════════

def build_dashboard_data(df, shap_values, feature_names, results):
    """Siapkan semua data untuk halaman Dashboard (Halaman 1)."""
    df_an = df[df["anomali_final"] == 1].copy().sort_values("ensemble_score", ascending=False)

    total_data     = len(df)
    total_anomali  = len(df_an)
    pct_anomali    = round(total_anomali / total_data * 100, 2) if total_data else 0
    skpd_terdampak = int(df_an["id_skpd"].nunique()) if total_anomali else 0

    # Tren harian % anomali (30 hari terakhir)
    tren_raw = (
        df.groupby(df["tanggal_kirim"].dt.date)["anomali_final"]
        .agg(["sum", "count"]).reset_index()
    )
    tren_raw.columns = ["tanggal", "anomali", "total"]
    tren_raw["pct"]     = (tren_raw["anomali"] / tren_raw["total"] * 100).round(2)
    tren_raw["tanggal"] = tren_raw["tanggal"].astype(str)
    tren = tren_raw.tail(30).to_dict(orient="records")



    # Top 5 SKPD dengan anomali terbanyak
    top_skpd_raw = (
        df_an.groupby("id_skpd").size()
        .sort_values(ascending=False).head(5).reset_index()
    )
    top_skpd_raw.columns = ["skpd", "jumlah"]
    top_skpd = top_skpd_raw.to_dict(orient="records")

    # Alert cards: 10 pegawai UNIK dengan skor tertinggi (1 per pegawai)
    alert_cols = ["karyawan_id", "id_skpd", "tanggal_kirim",
                  "ensemble_score", "alasan_utama", "alasan_kedua",
                  "shap_top1_fitur", "shap_top1_nilai", "shap_top2_fitur", "shap_top2_nilai",
                  "lat", "long", "fe_jarak_masuk_pulang", "fe_absen_lompat",
                  "lat_masuk", "long_masuk", "lat_pulang", "long_pulang",
                  "fe_dist_personal", "fe_drift_lokasi"]
    alert_cols = [c for c in alert_cols if c in df_an.columns]
    df_alerts  = df_an.drop_duplicates(subset="karyawan_id", keep="first").head(10)
    alerts     = df_to_safe(df_alerts, alert_cols)

    return {
        "total_data"    : total_data,
        "total_anomali" : total_anomali,
        "pct_anomali"   : pct_anomali,
        "skpd_terdampak": skpd_terdampak,
        "tren"          : tren,

        "top_skpd"      : top_skpd,
        "alerts"        : alerts,
    }


def build_kinerja_data(results):
    """Siapkan data kinerja algoritma (Halaman 2)."""
    algoritma = ["IF", "LOF", "ECOD", "Ensemble"]
    best = max(algoritma, key=lambda m: results.get(m, {}).get("f1", 0))
    return {"results": results, "best": best}


def build_detail_data(df):
    """Siapkan data tabel detail anomali per SKPD (Halaman 3)."""
    df_an = df[df["anomali_final"] == 1].copy().sort_values("ensemble_score", ascending=False)
    cols  = ["karyawan_id", "id_skpd", "tanggal_kirim", "ensemble_score",
             "alasan_utama", "alasan_kedua",
             "if_pred", "lof_pred", "ecod_pred",
             "lat", "long", "fe_dist_km", "fe_drift_lokasi",
             "fe_dist_personal", "fe_jarak_masuk_pulang", "fe_absen_lompat",
             "lat_masuk", "long_masuk", "lat_pulang", "long_pulang",
             "fe_dbscan_only", "fe_both_noise", "is_noise", "is_noise_dbscan",
             "dbscan_cluster", "cluster_id",
             "is_absen_lengkap", "fe_konsistensi_lokasi"]
    cols      = [c for c in cols if c in df_an.columns]
    skpd_list = sorted(df["id_skpd"].unique().tolist())
    return {"rows": df_to_safe(df_an, cols), "skpd_list": skpd_list}


def build_map_data(df):
    """Siapkan data peta sebaran anomali (Halaman 4)."""
    df_an = df[
        (df["anomali_final"] == 1) &
        df["lat"].notna() & df["long"].notna()
    ].copy()

    # Rekonstruksi kolom 'jenis' dari OHE (jenis_P) jika kolom asli tidak ada
    if "jenis" not in df_an.columns and "jenis_P" in df_an.columns:
        df_an["jenis"] = df_an["jenis_P"].apply(lambda x: "P" if x == 1 else "M")

    cols    = ["karyawan_id", "id_skpd", "tanggal_kirim", "lat", "long",
               "ensemble_score", "alasan_utama", "jenis"]
    cols    = [c for c in cols if c in df_an.columns]

    return {"points": df_to_safe(df_an, cols)}


# ══════════════════════════════════════════════════════════════════
#  ROUTES — HALAMAN
# ══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html",
                           nama_instansi=NAMA_INSTANSI,
                           has_model=models_exist())

@app.route("/kinerja")
def kinerja():
    return render_template("kinerja.html",
                           nama_instansi=NAMA_INSTANSI,
                           has_model=models_exist())

@app.route("/detail")
def detail():
    return render_template("detail.html",
                           nama_instansi=NAMA_INSTANSI,
                           has_model=models_exist())

@app.route("/peta")
def peta():
    embed = request.args.get("embed") == "1"
    if embed:
        return render_template("peta.html",
                               nama_instansi=NAMA_INSTANSI,
                               has_model=models_exist(),
                               embed_mode=True)
    return render_template("peta.html",
                           nama_instansi=NAMA_INSTANSI,
                           has_model=models_exist(),
                           embed_mode=False)

@app.route("/upload")
def upload_page():
    return render_template("upload.html",
                           nama_instansi=NAMA_INSTANSI,
                           has_model=models_exist())


# ══════════════════════════════════════════════════════════════════
#  ROUTES — API
# ══════════════════════════════════════════════════════════════════

@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    Upload file Excel/CSV.
    Validasi kolom wajib -> simpan path -> kembalikan preview.
    """
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "Tidak ada file yang dikirim."})

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"ok": False, "error": "Nama file kosong."})

    ext = f.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"ok": False,
                        "error": "Format tidak didukung. Gunakan .xlsx / .xls / .csv"})

    path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename))
    f.save(path)

    try:
        if ext in ("xlsx", "xls"):
            xf = pd.ExcelFile(path)
            # File hasil ekspor deteksi anomali sebelumnya -> load model lama, skip reprocessing
            if "Semua Anomali" in xf.sheet_names and models_exist():
                return jsonify({"ok": True, "mode": "load",
                                "msg": "Model sudah ada. Menggunakan hasil sebelumnya."})
            df_raw = pd.read_excel(path)
        else:
            df_raw = pd.read_csv(path)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Gagal membaca file: {str(e)}"})

    # Validasi kolom wajib minimum
    required_cols = ["karyawan_id", "id_skpd", "tanggal_kirim", "lat", "long"]
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        return jsonify({"ok": False,
                        "error": f"Kolom wajib tidak ditemukan: {', '.join(missing)}"})

    # Preview singkat untuk ditampilkan ke user
    preview = {
        "total_baris"    : len(df_raw),
        "rentang_tanggal": [
            str(pd.to_datetime(df_raw["tanggal_kirim"], errors="coerce").min())[:10],
            str(pd.to_datetime(df_raw["tanggal_kirim"], errors="coerce").max())[:10],
        ],
        "kolom": df_raw.columns.tolist(),
    }

    # Simpan path di server — lebih reliable dari cookie session
    PENDING_FILE["path"] = path
    return jsonify({"ok": True, "mode": "preview", "preview": preview})


@app.route("/api/proses", methods=["POST"])
def api_proses():
    """
    Jalankan seluruh pipeline preprocessing + prediksi.
    Bisa memakan waktu 2–10 menit untuk data besar.
    use_reloader=False di app.run() mencegah server restart di tengah proses.
    """
    file_path = PENDING_FILE.get("path")
    if not file_path or not os.path.exists(file_path):
        return jsonify({"ok": False,
                        "error": "File tidak ditemukan. Silakan upload ulang."})
    try:
        ext    = file_path.rsplit(".", 1)[-1].lower()
        df_raw = pd.read_excel(file_path) if ext in ("xlsx", "xls") else pd.read_csv(file_path)

        # Simpan raw upload ke disk sebelum preprocessing
        os.makedirs(MODEL_DIR, exist_ok=True)
        df_raw.to_pickle(os.path.join(MODEL_DIR, "df_raw.pkl"))

        df, shap_values, expected_val, feature_names, results, _, X_scaled = \
            run_preprocessing_and_prediction(df_raw)

        save_eda_summary_cache(df_raw)

        return jsonify({
            "ok"           : True,
            "total_anomali": int(df["anomali_final"].sum()),
            "total_data"   : len(df),
        })
    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e), "detail": traceback.format_exc()})


@app.route("/api/dashboard")
def api_dashboard():
    if not models_exist():
        return jsonify({"ok": False,
                        "error": "Belum ada data. Silakan upload file terlebih dahulu."})
    (IF_MODEL, LOF_MODEL, ECOD_MODEL, scaler,
     shap_values, expected_val, feature_names,
     results, office_centroids, X_scaled, df) = load_models()
    return jsonify({"ok": True,
                    **build_dashboard_data(df, shap_values, feature_names, results)})


@app.route("/api/kinerja")
def api_kinerja():
    if not models_exist():
        return jsonify({"ok": False, "error": "Belum ada data."})
    (_, _, _, _, _, _, _, results, _, _, _) = load_models()
    return jsonify({"ok": True, **build_kinerja_data(results)})


@app.route("/api/detail")
def api_detail():
    if not models_exist():
        return jsonify({"ok": False, "error": "Belum ada data."})
    (_, _, _, _, _, _, _, _, _, _, df) = load_models()
    return jsonify({"ok": True, **build_detail_data(df)})


@app.route("/api/peta")
def api_peta():
    if not models_exist():
        return jsonify({"ok": False, "error": "Belum ada data."})
    
    if request.args.get("eda") == "1":
        df_raw = load_raw_df()
        if df_raw is None:
            return jsonify({"ok": False, "error": "Belum ada data mentah."})
            
        lat_num = pd.to_numeric(df_raw["lat"], errors="coerce")
        long_num = pd.to_numeric(df_raw["long"], errors="coerce")
        
        valid_gps_mask = (
            df_raw["lat"].notna() & df_raw["long"].notna() &
            lat_num.notna() & long_num.notna() &
            (lat_num != 0) & (long_num != 0) &
            (lat_num >= -90) & (lat_num <= 90) &
            (long_num >= -180) & (long_num <= 180)
        )
        df_clean = df_raw[valid_gps_mask].copy()
        
        df_clean["lat"] = df_clean["lat"].astype(float)
        df_clean["long"] = df_clean["long"].astype(float)

        if "jenis" not in df_clean.columns:
            df_clean["jenis"] = "M"
        else:
            df_clean["jenis"] = df_clean["jenis"].astype(str).str.strip().str.upper().map({
                "M": "M", "MASUK": "M",
                "P": "P", "PULANG": "P"
            }).fillna("M")

        # EDA map tetap ringan, tapi jangan sampai titik anomali langka
        # seperti lokasi luar kota hilang karena random sampling raw data.
        max_eda_points = 3000
        df_priority = pd.DataFrame()
        hasil_path = os.path.join(MODEL_DIR, "df_hasil.pkl")
        if os.path.exists(hasil_path):
            try:
                df_hasil_map = pd.read_pickle(hasil_path)
                if "anomali_final" in df_hasil_map.columns:
                    df_priority = df_hasil_map[
                        (df_hasil_map["anomali_final"] == 1) &
                        df_hasil_map["lat"].notna() &
                        df_hasil_map["long"].notna()
                    ].copy()
                    df_priority["lat"] = pd.to_numeric(df_priority["lat"], errors="coerce")
                    df_priority["long"] = pd.to_numeric(df_priority["long"], errors="coerce")
                    df_priority = df_priority.dropna(subset=["lat", "long"])
                    if "jenis" not in df_priority.columns and "jenis_P" in df_priority.columns:
                        df_priority["jenis"] = df_priority["jenis_P"].apply(lambda x: "P" if x == 1 else "M")
                    elif "jenis" not in df_priority.columns:
                        df_priority["jenis"] = "M"
            except Exception:
                df_priority = pd.DataFrame()

        remaining = max(max_eda_points - len(df_priority), 0)
        if len(df_clean) > remaining and remaining > 0:
            df_clean = df_clean.sample(remaining, random_state=42)
        elif remaining == 0:
            df_clean = df_clean.iloc[0:0]

        if not df_priority.empty:
            df_clean = pd.concat([df_priority, df_clean], ignore_index=True, sort=False)
            dedupe_cols = [c for c in ["karyawan_id", "tanggal_kirim", "jenis", "lat", "long"] if c in df_clean.columns]
            if dedupe_cols:
                df_clean = df_clean.drop_duplicates(subset=dedupe_cols, keep="first")
            
        cols = ["karyawan_id", "id_skpd", "tanggal_kirim", "lat", "long", "jenis",
                "anomali_final", "ensemble_score", "alasan_utama"]
        cols = [c for c in cols if c in df_clean.columns]
        return jsonify({"ok": True, "points": df_to_safe(df_clean, cols)})
        
    (_, _, _, _, _, _, _, _, _, _, df) = load_models()
    return jsonify({"ok": True, **build_map_data(df)})


@app.route("/eda")
def eda_page():
    return render_template("eda.html",
                           nama_instansi=NAMA_INSTANSI,
                           has_model=models_exist())


def build_eda_summary(df_raw):
    """
    Exploratory Data Analysis — semua statistik dan distribusi untuk halaman EDA.
    Murni menggunakan data mentah (df_raw) sebelum preprocessing/model.
    """
    try:
        if df_raw is None:
            raise ValueError("Belum ada data mentah.")
        df_raw = df_raw.copy()

        # Ensure datetime is parsed for temporal analytics
        df_raw["tanggal_kirim_dt"] = pd.to_datetime(df_raw["tanggal_kirim"], errors="coerce")

        n_rows = len(df_raw)
        n_cols = len(df_raw.columns)
        n_skpd = int(df_raw["id_skpd"].nunique()) if "id_skpd" in df_raw.columns else 0
        n_pegawai = int(df_raw["karyawan_id"].nunique()) if "karyawan_id" in df_raw.columns else 0

        # Normalize jenis for counting
        if "jenis" in df_raw.columns:
            jenis_raw = df_raw["jenis"].astype(str).str.strip().str.upper()
            n_masuk = int(jenis_raw.str.startswith("M").sum())
            n_pulang = int(jenis_raw.str.startswith("P").sum())
        else:
            n_masuk = 0
            n_pulang = 0

        periode_start = str(df_raw["tanggal_kirim_dt"].min())[:10] if df_raw["tanggal_kirim_dt"].notna().any() else "-"
        periode_end = str(df_raw["tanggal_kirim_dt"].max())[:10] if df_raw["tanggal_kirim_dt"].notna().any() else "-"

        ringkasan = {
            "n_rows"        : n_rows,
            "n_cols"        : n_cols,
            "n_skpd"        : n_skpd,
            "n_pegawai"     : n_pegawai,
            "n_masuk"       : n_masuk,
            "n_pulang"      : n_pulang,
            "periode_start" : periode_start,
            "periode_end"   : periode_end,
        }

        # ── 1. Profil Kualitas Data Mentah ───────────────────────────
        # A. Kandidat Duplikat
        df_temp = df_raw.copy()
        df_temp["_tgl"] = df_raw["tanggal_kirim_dt"].dt.date
        if "karyawan_id" in df_temp.columns and "jenis" in df_temp.columns:
            df_temp_clean = df_temp.dropna(subset=["karyawan_id", "_tgl", "jenis"])
            n_duplicates = int(df_temp_clean.duplicated(subset=["karyawan_id", "_tgl", "jenis"], keep="first").sum())
        else:
            n_duplicates = 0

        # B. GPS bermasalah
        lat_numeric = pd.to_numeric(df_raw["lat"], errors="coerce")
        long_numeric = pd.to_numeric(df_raw["long"], errors="coerce")
        
        is_gps_invalid_row = df_raw["lat"].isna() | df_raw["long"].isna() | lat_numeric.isna() | long_numeric.isna()
        n_gps_missing = int(is_gps_invalid_row.sum())
        
        is_gps_zero_row = (lat_numeric == 0) & (long_numeric == 0) & ~is_gps_invalid_row
        n_gps_null_island = int(is_gps_zero_row.sum())
        
        is_gps_out_row = ((lat_numeric < -90) | (lat_numeric > 90) | (long_numeric < -180) | (long_numeric > 180)) & ~is_gps_invalid_row & ~is_gps_zero_row
        n_gps_out_of_range = int(is_gps_out_row.sum())
        
        # C. Tanggal tidak valid
        n_tanggal_invalid = int(df_raw["tanggal_kirim"].isna().sum() + (df_raw["tanggal_kirim"].notna() & df_raw["tanggal_kirim_dt"].isna()).sum())
        
        # D. Kolom kritis kosong
        critical_cols = ["karyawan_id", "id_skpd", "tanggal_kirim", "lat", "long", "jenis"]
        n_kritis_kosong = 0
        for c in critical_cols:
            if c in df_raw.columns:
                n_kritis_kosong += int(df_raw[c].isna().sum())
            else:
                n_kritis_kosong += len(df_raw)

        # E. Jenis selain M/P
        if "jenis" in df_raw.columns:
            jenis_clean = df_raw["jenis"].astype(str).str.strip().str.upper()
            n_jenis_invalid = int((~jenis_clean.isin(["M", "P", "MASUK", "PULANG"])).sum())
        else:
            n_jenis_invalid = len(df_raw)

        kualitas_raw = {
            "n_raw": n_rows,
            "n_duplicates": n_duplicates,
            "n_gps_missing": n_gps_missing,
            "n_gps_null_island": n_gps_null_island,
            "n_gps_out_of_range": n_gps_out_of_range,
            "n_gps_total_error": n_gps_missing + n_gps_null_island + n_gps_out_of_range,
            "n_tanggal_invalid": n_tanggal_invalid,
            "n_kritis_kosong": n_kritis_kosong,
            "n_jenis_invalid": n_jenis_invalid,
        }

        # ── 2. Distribusi jenis (Masuk / Pulang) ─────────────────────
        dist_jenis = []
        if "jenis" in df_raw.columns:
            jenis_mapped = df_raw["jenis"].astype(str).str.strip().str.upper().map({
                "M": "Masuk", "MASUK": "Masuk",
                "P": "Pulang", "PULANG": "Pulang"
            }).fillna("Lainnya")
            vc = jenis_mapped.value_counts()
            dist_jenis = [{"jenis": k, "jumlah": int(v)} for k, v in vc.items()]

        # ── 3. Top 15 SKPD (Masuk vs Pulang) ─────────────────────────
        dist_skpd = []
        if "id_skpd" in df_raw.columns and "jenis" in df_raw.columns:
            df_temp = df_raw.copy()
            df_temp["jenis_clean"] = df_temp["jenis"].astype(str).str.strip().str.upper().map({
                "M": "M", "MASUK": "M",
                "P": "P", "PULANG": "P"
            }).fillna("Lainnya")
            skpd_counts = df_temp.groupby(["id_skpd", "jenis_clean"]).size().unstack(fill_value=0).reset_index()
            for j_col in ["M", "P", "Lainnya"]:
                if j_col not in skpd_counts.columns:
                    skpd_counts[j_col] = 0
            skpd_counts["total"] = skpd_counts["M"] + skpd_counts["P"] + skpd_counts["Lainnya"]
            skpd_counts = skpd_counts.sort_values("total", ascending=False).head(15)
            dist_skpd = [
                {
                    "skpd": str(row.id_skpd),
                    "masuk": int(row.M),
                    "pulang": int(row.P),
                    "lainnya": int(row.Lainnya),
                    "total": int(row.total)
                }
                for row in skpd_counts.itertuples()
            ]

        # ── 4. Tren harian (Masuk vs Pulang) ─────────────────────────
        tren_harian = []
        if df_raw["tanggal_kirim_dt"].notna().any() and "jenis" in df_raw.columns:
            df_temp = df_raw.copy()
            df_temp["jenis_clean"] = df_temp["jenis"].astype(str).str.strip().str.upper().map({
                "M": "M", "MASUK": "M",
                "P": "P", "PULANG": "P"
            }).fillna("Lainnya")
            df_temp = df_temp[df_temp["tanggal_kirim_dt"].notna()]
            tren = df_temp.groupby([df_temp["tanggal_kirim_dt"].dt.date, "jenis_clean"]).size().unstack(fill_value=0).reset_index()
            for j_col in ["M", "P", "Lainnya"]:
                if j_col not in tren.columns:
                    tren[j_col] = 0
            tren["total"] = tren["M"] + tren["P"] + tren["Lainnya"]
            tren = tren.sort_values("tanggal_kirim_dt")
            tren_harian = [
                {
                    "tanggal": str(row.tanggal_kirim_dt),
                    "masuk": int(row.M),
                    "pulang": int(row.P),
                    "lainnya": int(row.Lainnya),
                    "total": int(row.total)
                }
                for row in tren.itertuples()
            ]

        # ── 5. Distribusi jam presensi (histogram 24 bucket, Masuk vs Pulang) ──
        dist_jam = []
        if df_raw["tanggal_kirim_dt"].notna().any() and "jenis" in df_raw.columns:
            df_temp = df_raw.copy()
            df_temp["hour"] = df_temp["tanggal_kirim_dt"].dt.hour
            df_temp["jenis_clean"] = df_temp["jenis"].astype(str).str.strip().str.upper().map({
                "M": "M", "MASUK": "M",
                "P": "P", "PULANG": "P"
            }).fillna("Lainnya")
            jam_grp = df_temp.dropna(subset=["hour"]).groupby(["hour", "jenis_clean"]).size().unstack(fill_value=0)
            for j_col in ["M", "P", "Lainnya"]:
                if j_col not in jam_grp.columns:
                    jam_grp[j_col] = 0
            for h in range(24):
                m_count = int(jam_grp.loc[h, "M"]) if h in jam_grp.index else 0
                p_count = int(jam_grp.loc[h, "P"]) if h in jam_grp.index else 0
                l_count = int(jam_grp.loc[h, "Lainnya"]) if h in jam_grp.index else 0
                dist_jam.append({
                    "jam"    : f"{h:02d}:00",
                    "masuk"  : m_count,
                    "pulang" : p_count,
                    "lainnya": l_count,
                    "total"  : m_count + p_count + l_count
                })

        # ── 6. Distribusi jarak ke kantor (histogram 11 bin, Masuk vs Pulang) ──
        dist_jarak = []
        centroids_path = os.path.join(MODEL_DIR, "office_centroids.pkl")
        jarak_col = None
        if os.path.exists(centroids_path) and "id_skpd" in df_raw.columns:
            try:
                with open(centroids_path, "rb") as f:
                    office_centroids = pickle.load(f)
                
                df_temp = df_raw.copy()
                df_temp["id_skpd"] = df_temp["id_skpd"].astype(str)
                office_centroids["id_skpd"] = office_centroids["id_skpd"].astype(str)
                
                df_temp = pd.merge(df_temp, office_centroids, on="id_skpd", how="left")
                
                if "lat_kantor" in df_raw.columns and "long_kantor" in df_raw.columns:
                    lat_k = df_temp["lat_kantor_x"].fillna(df_temp["lat_kantor_y"])
                    long_k = df_temp["long_kantor_x"].fillna(df_temp["long_kantor_y"])
                else:
                    lat_k = df_temp["lat_kantor"]
                    long_k = df_temp["long_kantor"]
                    
                lat_absen = pd.to_numeric(df_temp["lat"], errors="coerce")
                long_absen = pd.to_numeric(df_temp["long"], errors="coerce")
                lat_k_numeric = pd.to_numeric(lat_k, errors="coerce").fillna(lat_absen)
                long_k_numeric = pd.to_numeric(long_k, errors="coerce").fillna(long_absen)

                df_raw["fe_dist_km"] = haversine_vectorized(
                    lat_absen.fillna(0),
                    long_absen.fillna(0),
                    lat_k_numeric.fillna(0),
                    long_k_numeric.fillna(0)
                )
                jarak_col = "fe_dist_km"
            except Exception as e:
                print("Error calculating raw distance on the fly:", e)
                
        if jarak_col is None:
            jarak_col = next((c for c in ["fe_dist_km", "jarak_ke_kantor"] if c in df_raw.columns), None)

        if jarak_col and "jenis" in df_raw.columns:
            sub_df = df_raw[[jarak_col, "jenis"]].dropna().copy()
            if jarak_col == "jarak_ke_kantor":
                sub_df[jarak_col] = pd.to_numeric(sub_df[jarak_col], errors="coerce") / 1000.0
            else:
                sub_df[jarak_col] = pd.to_numeric(sub_df[jarak_col], errors="coerce")
                
            sub_df = sub_df.dropna()
            bins  = list(range(0, 11)) + [999]
            labels = ["0–1km","1–2km","2–3km","3–4km","4–5km",
                      "5–6km","6–7km","7–8km","8–9km","9–10km",">10km"]
            sub_df["bin"] = pd.cut(sub_df[jarak_col], bins=bins, labels=labels, right=False)
            
            sub_df["jenis_clean"] = sub_df["jenis"].astype(str).str.strip().str.upper().map({
                "M": "M", "MASUK": "M",
                "P": "P", "PULANG": "P"
            }).fillna("Lainnya")
            
            jarak_grp = sub_df.groupby(["bin", "jenis_clean"]).size().unstack(fill_value=0)
            for j_col in ["M", "P", "Lainnya"]:
                if j_col not in jarak_grp.columns:
                    jarak_grp[j_col] = 0
            dist_jarak = [
                {
                    "bin": lb,
                    "masuk": int(jarak_grp.loc[lb, "M"]) if lb in jarak_grp.index else 0,
                    "pulang": int(jarak_grp.loc[lb, "P"]) if lb in jarak_grp.index else 0,
                    "lainnya": int(jarak_grp.loc[lb, "Lainnya"]) if lb in jarak_grp.index else 0,
                    "total": int(jarak_grp.loc[lb, "M"] + jarak_grp.loc[lb, "P"] + jarak_grp.loc[lb, "Lainnya"]) if lb in jarak_grp.index else 0
                }
                for lb in labels
            ]

        # ── 7. Kelengkapan absen harian (is_absen_lengkap dari raw) ───
        dist_kelengkapan = []
        if "karyawan_id" in df_raw.columns and df_raw["tanggal_kirim_dt"].notna().any() and "jenis" in df_raw.columns:
            df_temp = df_raw.dropna(subset=["karyawan_id", "tanggal_kirim_dt", "jenis"]).copy()
            df_temp["_tgl"] = df_temp["tanggal_kirim_dt"].dt.date
            df_temp["jenis_clean"] = df_temp["jenis"].astype(str).str.strip().str.upper().map({
                "M": "M", "MASUK": "M",
                "P": "P", "PULANG": "P"
            }).fillna("Lainnya")
            df_temp = df_temp[df_temp["jenis_clean"].isin(["M", "P"])]
            
            grouped = df_temp.groupby(["karyawan_id", "_tgl"])["jenis_clean"].unique()
            n_lengkap_days = 0
            n_hanya_masuk_days = 0
            n_hanya_pulang_days = 0
            
            for types in grouped:
                has_m = "M" in types
                has_p = "P" in types
                if has_m and has_p:
                    n_lengkap_days += 1
                elif has_m:
                    n_hanya_masuk_days += 1
                elif has_p:
                    n_hanya_pulang_days += 1
                    
            dist_kelengkapan = [
                {"status": "Lengkap M+P", "jumlah": n_lengkap_days},
                {"status": "Hanya Masuk", "jumlah": n_hanya_masuk_days},
                {"status": "Hanya Pulang", "jumlah": n_hanya_pulang_days}
            ]

        # ── 8. Missing values per kolom raw ──────────────────────────
        missing_vals = []
        mv = df_raw.isnull().sum()
        if "tanggal_kirim_dt" in mv:
            mv = mv.drop("tanggal_kirim_dt")
        if "fe_dist_km" in mv:
            mv = mv.drop("fe_dist_km")
        mv = mv.sort_values(ascending=False)
        for col, cnt in mv.items():
            missing_vals.append({
                "kolom": str(col),
                "jumlah": int(cnt),
                "pct": round(int(cnt)/n_rows*100, 2)
            })

        # ── 9. Distribusi kategori catatan dari raw ───────────────────
        dist_catatan = []
        catatan_counts = {"Dinas": 0, "Kendala_Teknis": 0, "Alasan_Pribadi": 0, "Lainnya": 0, "Tidak_Ada_Catatan": 0}
        
        if "catatan" in df_raw.columns:
            for val in df_raw["catatan"].dropna():
                val_str = str(val).strip()
                if val_str.lower() in [x.lower() for x in NILAI_KOSONG_CATATAN] or not val_str:
                    catatan_counts["Tidak_Ada_Catatan"] += 1
                else:
                    cat = klasifikasi_catatan(val_str)
                    catatan_counts[cat] = catatan_counts.get(cat, 0) + 1
        else:
            catatan_counts["Tidak_Ada_Catatan"] = len(df_raw)
            
        for cat_name, cnt in catatan_counts.items():
            dist_catatan.append({
                "kategori": cat_name,
                "total": cnt
            })
        dist_catatan.sort(key=lambda x: x["total"], reverse=True)

        # ── 10. Distribusi hari dalam seminggu (Masuk vs Pulang) ───────
        dist_hari = []
        if df_raw["tanggal_kirim_dt"].notna().any() and "jenis" in df_raw.columns:
            nama_hari = ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"]
            df_temp = df_raw.copy()
            df_temp["_dow"] = df_temp["tanggal_kirim_dt"].dt.dayofweek
            df_temp["jenis_clean"] = df_temp["jenis"].astype(str).str.strip().str.upper().map({
                "M": "M", "MASUK": "M",
                "P": "P", "PULANG": "P"
            }).fillna("Lainnya")
            
            dow_grp = df_temp.dropna(subset=["_dow"]).groupby(["_dow", "jenis_clean"]).size().unstack(fill_value=0)
            for j_col in ["M", "P", "Lainnya"]:
                if j_col not in dow_grp.columns:
                    dow_grp[j_col] = 0
            for i in range(7):
                m_count = int(dow_grp.loc[i, "M"]) if i in dow_grp.index else 0
                p_count = int(dow_grp.loc[i, "P"]) if i in dow_grp.index else 0
                l_count = int(dow_grp.loc[i, "Lainnya"]) if i in dow_grp.index else 0
                dist_hari.append({
                    "hari"   : nama_hari[i],
                    "masuk"  : m_count,
                    "pulang" : p_count,
                    "lainnya": l_count,
                    "total"  : m_count + p_count + l_count
                })

        result = {
            "ok"              : True,
            "ringkasan"       : ringkasan,
            "kualitas_raw"    : kualitas_raw,
            "dist_jenis"      : dist_jenis,
            "dist_skpd"       : dist_skpd,
            "tren_harian"     : tren_harian,
            "dist_jam"        : dist_jam,
            "dist_jarak"      : dist_jarak,
            "dist_kelengkapan": dist_kelengkapan,
            "missing_vals"    : missing_vals,
            "dist_catatan"    : dist_catatan,
            "dist_hari"       : dist_hari,
        }

        return result

    except Exception as e:
        import traceback
        raise RuntimeError(f"Gagal membangun cache EDA: {e}\n{traceback.format_exc()}") from e


def save_eda_summary_cache(df_raw):
    """Hitung sekali ringkasan EDA lalu simpan ke pkl dan json."""
    summary = build_eda_summary(df_raw)
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(EDA_CACHE_PKL, "wb") as f:
        pickle.dump(summary, f)

    with open(EDA_CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    return summary


def load_eda_summary_cache():
    """Load cache EDA tanpa menghitung ulang data mentah."""
    if os.path.exists(EDA_CACHE_PKL):
        try:
            with open(EDA_CACHE_PKL, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    if os.path.exists(EDA_CACHE_JSON):
        try:
            with open(EDA_CACHE_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return None


@app.route("/api/eda")
def api_eda():
    """
    Endpoint EDA cepat: hanya membaca cache yang dibuat setelah preprocessing.
    """
    if not models_exist():
        return jsonify({"ok": False, "error": "Belum ada data."})

    cache = load_eda_summary_cache()
    if cache is None:
        return jsonify({
            "ok": False,
            "error": "Cache EDA belum tersedia. Jalankan proses ulang dari halaman Upload agar ringkasan EDA dibuat sekali dan disimpan."
        })

    return jsonify(cache)


@app.route("/shap")
def shap_page():
    return render_template("shap.html",
                           nama_instansi=NAMA_INSTANSI,
                           has_model=models_exist())


@app.route("/api/shap")
def api_shap():
    """
    Siapkan data SHAP untuk visualisasi interaktif.
    Query param: model = if (default) | lof | ecod | ensemble
    Hasil di-cache ke file pkl agar tidak perlu dihitung ulang.
    """
    if not models_exist():
        return jsonify({"ok": False, "error": "Belum ada data."})

    model_choice = request.args.get("model", "if").lower()
    if model_choice not in ("if", "lof", "ecod", "ensemble"):
        model_choice = "if"

    # ── Cek cache: jika sudah pernah dihitung, langsung return ──
    cache_file = os.path.join(MODEL_DIR, f"shap_cache_{model_choice}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            return jsonify(cached)
        except Exception:
            pass  # Cache corrupt, hitung ulang

    try:
        (IF_MODEL, LOF_MODEL, ECOD_MODEL, _, shap_values_stored, expected_val_stored,
         feature_names, results, _, X_scaled, df) = load_models()

        feat_names = feature_names
        try:
            import shap
        except Exception:
            shap = None

        # ── Compute SHAP based on selected model ──────────────────
        if model_choice == "if":
            # TreeExplainer — fast & exact
            explainer = shap.TreeExplainer(IF_MODEL)
            shap_arr = explainer.shap_values(X_scaled)
            expected_val = float(np.mean(explainer.expected_value))
            model_label = "Isolation Forest"

        elif model_choice == "lof":
            # KernelExplainer on LOF decision_function (negative_outlier_factor_)
            # Use sampling for speed: background=100 samples, explain=500 samples max
            n_bg = min(100, len(X_scaled))
            bg_idx = np.random.choice(len(X_scaled), n_bg, replace=False)
            background = X_scaled[bg_idx]

            # LOF with novelty=True for predict capability
            from sklearn.neighbors import LocalOutlierFactor
            lof_novelty = LocalOutlierFactor(
                n_neighbors=20, contamination=0.05, novelty=True)
            lof_novelty.fit(X_scaled)

            def lof_predict_fn(X):
                return -lof_novelty.decision_function(X)

            explainer = shap.KernelExplainer(lof_predict_fn, background)
            n_explain = min(500, len(X_scaled))
            explain_idx = np.random.choice(len(X_scaled), n_explain, replace=False)
            shap_arr_partial = explainer.shap_values(X_scaled[explain_idx], nsamples=50)

            # Expand to full size (fill non-explained with zeros for display)
            shap_arr = np.zeros((len(X_scaled), len(feat_names)))
            shap_arr[explain_idx] = shap_arr_partial
            expected_val = float(explainer.expected_value)
            model_label = "Local Outlier Factor (LOF)"

        elif model_choice == "ecod":
            # KernelExplainer on ECOD decision_function
            n_bg = min(100, len(X_scaled))
            bg_idx = np.random.choice(len(X_scaled), n_bg, replace=False)
            background = X_scaled[bg_idx]

            def ecod_predict_fn(X):
                return ECOD_MODEL.decision_function(X)

            explainer = shap.KernelExplainer(ecod_predict_fn, background)
            n_explain = min(500, len(X_scaled))
            explain_idx = np.random.choice(len(X_scaled), n_explain, replace=False)
            shap_arr_partial = explainer.shap_values(X_scaled[explain_idx], nsamples=50)

            shap_arr = np.zeros((len(X_scaled), len(feat_names)))
            shap_arr[explain_idx] = shap_arr_partial
            expected_val = float(explainer.expected_value)
            model_label = "ECOD (Empirical CDF)"

        else:  # ensemble
            # KernelExplainer on ensemble average score
            n_bg = min(100, len(X_scaled))
            bg_idx = np.random.choice(len(X_scaled), n_bg, replace=False)
            background = X_scaled[bg_idx]

            from sklearn.neighbors import LocalOutlierFactor
            lof_novelty = LocalOutlierFactor(
                n_neighbors=20, contamination=0.05, novelty=True)
            lof_novelty.fit(X_scaled)

            def ensemble_predict_fn(X):
                # IF score (normalized)
                if_raw = -IF_MODEL.decision_function(X)
                if_score = (if_raw - if_raw.min()) / (if_raw.max() - if_raw.min() + 1e-9)
                # LOF score (normalized)
                lof_raw = -lof_novelty.decision_function(X)
                lof_score = (lof_raw - lof_raw.min()) / (lof_raw.max() - lof_raw.min() + 1e-9)
                # ECOD score (normalized)
                ecod_raw = ECOD_MODEL.decision_function(X)
                ecod_score = (ecod_raw - ecod_raw.min()) / (ecod_raw.max() - ecod_raw.min() + 1e-9)
                return (if_score + lof_score + ecod_score) / 3.0

            explainer = shap.KernelExplainer(ensemble_predict_fn, background)
            n_explain = min(500, len(X_scaled))
            explain_idx = np.random.choice(len(X_scaled), n_explain, replace=False)
            shap_arr_partial = explainer.shap_values(X_scaled[explain_idx], nsamples=50)

            shap_arr = np.zeros((len(X_scaled), len(feat_names)))
            shap_arr[explain_idx] = shap_arr_partial
            expected_val = float(explainer.expected_value)
            model_label = "Ensemble (IF + LOF + ECOD)"

        # ── 1. Global importance (top 20) ──────────────────────────
        mean_abs = np.abs(shap_arr).mean(axis=0)
        imp_df   = pd.Series(mean_abs, index=feat_names).sort_values(ascending=False)
        top20    = imp_df.head(20)

        global_importance = [
            {"fitur": fn, "narasi": get_narasi(fn), "nilai": round(float(v), 6)}
            for fn, v in top20.items()
        ]

        # ── 2. Distribusi nilai SHAP per fitur (top 10, untuk box plot) ──
        top10_names = list(top20.head(10).index)
        top10_idx   = [list(feat_names).index(fn) for fn in top10_names]

        box_data = []
        for fn, idx in zip(top10_names, top10_idx):
            vals = shap_arr[:, idx].tolist()
            box_data.append({
                "fitur" : fn,
                "narasi": get_narasi(fn),
                "values": [round(v, 6) for v in vals]
            })

        # ── 3. Top-50 tabel anomali dengan kolom SHAP ──────────────
        df_an = (df[df["anomali_final"] == 1]
                 .copy()
                 .sort_values("ensemble_score", ascending=False)
                 .head(50))

        # Recompute top SHAP reasons from current shap_arr
        shap_abs_all = np.abs(shap_arr)
        top1_idx_arr = shap_abs_all.argmax(axis=1)

        table_rows = []
        full_indices = df.index.tolist()
        for _, row in df_an.iterrows():
            row_pos = full_indices.index(row.name) if row.name in full_indices else None
            if row_pos is not None and row_pos < len(shap_arr):
                row_shap = shap_arr[row_pos]
                row_abs = np.abs(row_shap)
                t1 = int(row_abs.argmax())
                t2 = int(np.argsort(row_abs)[-2])
                alasan1 = get_narasi(feat_names[t1])
                alasan2 = get_narasi(feat_names[t2])
                shap1_val = round(float(row_shap[t1]), 4)
                shap2_val = round(float(row_shap[t2]), 4)
                shap1_fitur = feat_names[t1]
                shap2_fitur = feat_names[t2]
            else:
                alasan1 = str(row.get("alasan_utama", "-"))
                alasan2 = str(row.get("alasan_kedua", "-"))
                shap1_val = round(float(row.get("shap_top1_nilai", 0)), 4)
                shap2_val = round(float(row.get("shap_top2_nilai", 0)), 4)
                shap1_fitur = str(row.get("shap_top1_fitur", "-"))
                shap2_fitur = str(row.get("shap_top2_fitur", "-"))

            table_rows.append({
                "karyawan_id"    : str(row.get("karyawan_id", "-")),
                "id_skpd"        : str(row.get("id_skpd", "-")),
                "tanggal"        : str(row.get("tanggal_kirim", "-"))[:16],
                "ensemble_score" : round(float(row.get("ensemble_score", 0)), 4),
                "alasan_utama"   : alasan1,
                "alasan_kedua"   : alasan2,
                "shap_top1_fitur": shap1_fitur,
                "shap_top1_nilai": shap1_val,
                "shap_top2_fitur": shap2_fitur,
                "shap_top2_nilai": shap2_val,
                "if_pred"        : int(row.get("if_pred", 0)),
                "lof_pred"       : int(row.get("lof_pred", 0)),
                "ecod_pred"      : int(row.get("ecod_pred", 0)),
            })

        # ── 4. Dependence data – fitur #1 vs SHAP value ────────────
        dep_data = []
        if len(top10_names) >= 1:
            fn1   = top10_names[0]
            idx1  = top10_idx[0]
            if fn1 in df.columns:
                feat_vals = df[fn1].fillna(0).tolist()
            else:
                feat_vals = shap_arr[:, idx1].tolist()

            shap_vals1 = shap_arr[:, idx1].tolist()
            fn2 = top10_names[1] if len(top10_names) >= 2 else None
            color_vals = (df[fn2].fillna(0).tolist()
                          if fn2 and fn2 in df.columns else None)

            dep_data = {
                "fitur_x"   : fn1,
                "narasi_x"  : get_narasi(fn1),
                "fitur_color": fn2,
                "narasi_color": get_narasi(fn2) if fn2 else None,
                "x"         : [round(float(v), 4) for v in feat_vals],
                "y"         : [round(float(v), 6) for v in shap_vals1],
                "color"     : [round(float(v), 4) for v in color_vals] if color_vals else None,
            }

        # ── 5. Ringkasan top fitur per rekaman (frekuensi alasan) ──
        # Use current shap_arr to compute top reasons
        anomali_mask = (df["anomali_final"] == 1).values
        if anomali_mask.any():
            anomali_top1 = [feat_names[int(top1_idx_arr[i])]
                           for i in range(len(top1_idx_arr)) if i < len(anomali_mask) and anomali_mask[i]]
            freq = pd.Series(anomali_top1).value_counts().head(10)
            top_reasons = [
                {"fitur": fn, "narasi": get_narasi(fn), "jumlah": int(cnt)}
                for fn, cnt in freq.items()
            ]
        else:
            top_reasons = []

        response_data = {
            "ok"                : True,
            "model"             : model_choice,
            "model_label"       : model_label,
            "expected_val"      : round(float(expected_val), 6),
            "n_features"        : len(feat_names),
            "n_anomali"         : int((df["anomali_final"] == 1).sum()),
            "global_importance" : global_importance,
            "box_data"          : box_data,
            "table_rows"        : table_rows,
            "dep_data"          : dep_data,
            "top_reasons"       : top_reasons,
        }

        # ── Simpan ke cache agar tidak perlu hitung ulang ──
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(response_data, f)
        except Exception:
            pass  # Gagal cache tidak fatal

        return jsonify(response_data)

    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e), "detail": traceback.format_exc()})


@app.route("/visualisasi")
def visualisasi():
    from flask import redirect
    return redirect("/eda")

@app.route("/api/visualisasi_data")
def api_visualisasi_data():
    if not models_exist():
        return jsonify({"ok": False, "error": "Belum ada data."})
    
    (_, _, _, _, _, _, _, _, _, _, df) = load_models()
    
    skpd_list = sorted(df["id_skpd"].unique().tolist())
    
    skpd = request.args.get("skpd", "Semua")
    karyawan = request.args.get("karyawan", "")
    
    dft = df.copy()
    
    # Map label deteksi
    dft["Label Deteksi"] = np.where(dft["anomali_final"] == 1, "Anomali", "Normal")
    
    # Rekonstruksi kolom jenis jika terhapus akibat One-Hot Encoding
    if "jenis" not in dft.columns:
        if "jenis_P" in dft.columns:
            dft["jenis"] = np.where(dft["jenis_P"] == 1, "P", "M")
        elif "jenis_M" in dft.columns:
            dft["jenis"] = np.where(dft["jenis_M"] == 1, "M", "P")
        else:
            dft["jenis"] = "M"

    if skpd != "Semua":
        dft = dft[dft["id_skpd"] == skpd]
        
    # 1. Distribusi Hasil Deteksi (Normal vs Anomali)
    vc = dft["Label Deteksi"].value_counts().reset_index()
    vc.columns = ["Hasil Deteksi", "count"]
    fig_status = px.pie(vc, values="count", names="Hasil Deteksi",
                        color="Hasil Deteksi", color_discrete_map={"Normal": "#3498db", "Anomali": "#e74c3c"})
    fig_status.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
    
    # 1b. Distribusi Masuk dan Pulang
    masuk = dft[dft["jenis"] == "M"]
    fig_masuk = None
    if not masuk.empty:
        vm = masuk["Label Deteksi"].value_counts().reset_index()
        vm.columns = ["Hasil Deteksi", "count"]
        fig_masuk = px.pie(vm, values="count", names="Hasil Deteksi", title="⬆️ Absensi Masuk (Normal vs Anomali)",
                           color="Hasil Deteksi", color_discrete_map={"Normal": "#3498db", "Anomali": "#e74c3c"}, hole=0.4)
        fig_masuk.update_layout(height=360, margin=dict(t=30, b=10, l=10, r=10))

    pulang = dft[dft["jenis"] == "P"]
    fig_pulang = None
    if not pulang.empty:
        vp = pulang["Label Deteksi"].value_counts().reset_index()
        vp.columns = ["Hasil Deteksi", "count"]
        fig_pulang = px.pie(vp, values="count", names="Hasil Deteksi", title="⬇️ Absensi Pulang (Normal vs Anomali)",
                            color="Hasil Deteksi", color_discrete_map={"Normal": "#3498db", "Anomali": "#e74c3c"}, hole=0.4)
        fig_pulang.update_layout(height=360, margin=dict(t=30, b=10, l=10, r=10))

    # 1c. Ringkasan (Kosongkan)
    summary_rows = []
    
    # 2. Top SKPD Indiscipline
    df_bermasalah = dft[dft["anomali_final"] == 1].copy()
    if not df_bermasalah.empty:
        df_bermasalah["Jenis Presensi"] = np.where(df_bermasalah["jenis"] == "M", "Masuk", "Pulang")
        skpd_bermasalah = df_bermasalah.groupby(["id_skpd", "Jenis Presensi"]).size().reset_index(name="n")
        fig_skpd = px.bar(skpd_bermasalah, x="id_skpd", y="n", color="Jenis Presensi",
                          title="Top 15 Anomali per SKPD",
                          barmode="stack", color_discrete_map={"Masuk": "#3498db", "Pulang": "#e74c3c"})
    else:
        fig_skpd = px.bar(title="Tidak ada anomali terdeteksi")
    fig_skpd.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
    
    # 3. Trend Temporal
    fig_jam = None
    fig_hari = None
    if "tanggal_kirim" in dft.columns:
        dft["tanggal_kirim"] = pd.to_datetime(dft["tanggal_kirim"], errors="coerce")
        dft["tanggal"] = dft["tanggal_kirim"].dt.date
        dft["jam"] = dft["tanggal_kirim"].dt.hour
        dft["weekday"] = dft["tanggal_kirim"].dt.weekday
        
        # Trend Harian
        daily = dft.groupby(["tanggal", "Label Deteksi"]).size().reset_index(name="n")
        fig_trend = px.line(daily, x="tanggal", y="n", color="Label Deteksi", markers=True,
                            color_discrete_map={"Normal": "#3498db", "Anomali": "#e74c3c"})
        fig_trend.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
        
        # Status per Jam
        jam_dist = dft.groupby(['jam','Label Deteksi']).size().reset_index(name='n')
        fig_jam = px.bar(jam_dist, x='jam', y='n', color='Label Deteksi', title='Hasil Deteksi per Jam',
                         color_discrete_map={"Normal": "#3498db", "Anomali": "#e74c3c"})
        fig_jam.update_layout(height=350, margin=dict(t=30, b=10, l=10, r=10), yaxis=dict(tickformat=",d"))
        fig_jam.add_vrect(x0=7, x1=9, fillcolor='green', opacity=0.07, annotation_text='Masuk')
        fig_jam.add_vrect(x0=15, x1=17, fillcolor='purple', opacity=0.07, annotation_text='Pulang')

        # Status per Hari
        dm = {0:'Senin',1:'Selasa',2:'Rabu',3:'Kamis',4:'Jumat',5:'Sabtu',6:'Minggu'}
        dft['hari'] = dft['weekday'].map(dm)
        hari_dist = dft.groupby(['hari','Label Deteksi']).size().reset_index(name='n')
        fig_hari = px.bar(hari_dist, x='hari', y='n', color='Label Deteksi', title='Hasil Deteksi per Hari',
                          color_discrete_map={"Normal": "#3498db", "Anomali": "#e74c3c"},
                          category_orders={'hari': list(dm.values())})
        fig_hari.update_layout(height=350, margin=dict(t=30, b=10, l=10, r=10))

    else:
        fig_trend = px.line(title="Tidak ada data temporal")
        
    # 4. Karyawan Timeline (Hunting)
    fig_pegawai = None
    if karyawan:
        df_pegawai = dft[dft["karyawan_id"] == karyawan].copy()
        if not df_pegawai.empty and "tanggal" in df_pegawai.columns and "fe_jam_desimal" in df_pegawai.columns:
            df_pegawai["ukuran"] = (df_pegawai["anomali_final"] == 1).astype(int) * 8 + 4
            fig_p = px.scatter(df_pegawai, x="tanggal", y="fe_jam_desimal", color="Label Deteksi",
                               symbol="jenis", size="ukuran", color_discrete_map={"Normal": "#3498db", "Anomali": "#e74c3c"},
                               hover_data=["dist_km"] if "dist_km" in df_pegawai.columns else [])
            fig_p.add_hline(y=8.25, line_dash="dot", line_color="#3498db", annotation_text="08:15")
            fig_p.add_hline(y=16.0, line_dash="dot", line_color="#9b59b6", annotation_text="16:00")
            fig_p.update_layout(height=400, plot_bgcolor="#fafafa")
            fig_pegawai = fig_p.to_json()

    return jsonify({
        "ok": True,
        "skpd_list": skpd_list,
        "fig_status": fig_status.to_json(),
        "fig_masuk": fig_masuk.to_json() if fig_masuk else None,
        "fig_pulang": fig_pulang.to_json() if fig_pulang else None,
        "summary_rows": summary_rows,
        "fig_skpd": fig_skpd.to_json(),
        "fig_trend": fig_trend.to_json(),
        "fig_jam": fig_jam.to_json() if fig_jam else None,
        "fig_hari": fig_hari.to_json() if fig_hari else None,
        "fig_pegawai": fig_pegawai
    })


# ══════════════════════════════════════════════════════════════════
#  ROUTES — PARAMETER TUNING
# ══════════════════════════════════════════════════════════════════

@app.route("/tuning")
def tuning():
    return render_template("tuning.html", nama_instansi=NAMA_INSTANSI)


# ══════════════════════════════════════════════════════════════════
#  GRID SEARCH — Hyperparameter Tuning Otomatis
# ══════════════════════════════════════════════════════════════════

GRID_SEARCH_FILE = os.path.join(MODEL_DIR, "grid_search_results.pkl")
GRID_SEARCH_CHECKPOINT_FILE = os.path.join(MODEL_DIR, "grid_search_checkpoint.pkl")
GRID_SEARCH_PROGRESS_LOCK = threading.Lock()
GRID_SEARCH_PROGRESS = {
    "run_id": None,
    "running": False,
    "status": "idle",
    "phase": "Idle",
    "message": "Belum ada grid search yang berjalan.",
    "percent": 0,
    "current": 0,
    "total": 0,
    "started_at": None,
    "updated_at": None,
    "finished_at": None,
}


def _set_grid_search_progress(**updates):
    """Update progress grid search untuk dibaca frontend via polling."""
    with GRID_SEARCH_PROGRESS_LOCK:
        GRID_SEARCH_PROGRESS.update(updates)
        GRID_SEARCH_PROGRESS["updated_at"] = time.time()


def _get_grid_search_progress():
    with GRID_SEARCH_PROGRESS_LOCK:
        progress = dict(GRID_SEARCH_PROGRESS)
    started_at = progress.get("started_at")
    finished_at = progress.get("finished_at")
    if started_at:
        end_time = finished_at or time.time()
        progress["elapsed_seconds"] = max(0, round(end_time - started_at, 1))
    else:
        progress["elapsed_seconds"] = 0
    return progress


def _save_grid_search_checkpoint(payload):
    """Simpan checkpoint grid search secara atomic agar aman saat proses panjang."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp_path = GRID_SEARCH_CHECKPOINT_FILE + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp_path, GRID_SEARCH_CHECKPOINT_FILE)


def _load_grid_search_checkpoint():
    if not os.path.exists(GRID_SEARCH_CHECKPOINT_FILE):
        return None
    try:
        with open(GRID_SEARCH_CHECKPOINT_FILE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _clear_grid_search_checkpoint():
    try:
        if os.path.exists(GRID_SEARCH_CHECKPOINT_FILE):
            os.remove(GRID_SEARCH_CHECKPOINT_FILE)
    except Exception:
        pass


def _grid_search_data_signature():
    """Tanda data agar checkpoint lama tidak dipakai untuk data upload yang berbeda."""
    sig = {}
    for name in ["df_hasil.pkl", "X_scaled.npy"]:
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            sig[name] = {
                "mtime": round(os.path.getmtime(path), 6),
                "size": os.path.getsize(path),
            }
        else:
            sig[name] = None
    return sig


def _if_grid_key(n_est, cont):
    return (int(n_est), round(float(cont), 10))


def _lof_grid_key(metric, n_nb, cont):
    return (str(metric), int(n_nb), round(float(cont), 10))


def _ecod_grid_key(cont):
    return (round(float(cont), 10),)


def _eval_preds(y_true, pred, score):
    """Hitung metrik evaluasi standar."""
    from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
    return {
        "f1":        round(float(f1_score(y_true, pred, zero_division=0)),        4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, pred, zero_division=0)),    4),
        "auc":       round(float(average_precision_score(y_true, score)),         4),
        "n_anomali": int(pred.sum()),
    }


@app.route("/api/tuning/gridsearch", methods=["POST"])
def api_tuning_gridsearch():
    """
    Jalankan Grid Search untuk IF, LOF, ECOD.
    Mencari kombinasi parameter terbaik berdasarkan F1 Score terhadap label_pseudo.
    Hasil disimpan ke models/grid_search_results.pkl
    """
    if not models_exist():
        _set_grid_search_progress(
            running=False,
            status="failed",
            phase="Validasi",
            message="Belum ada data. Upload dan proses data terlebih dahulu.",
            percent=0,
            current=0,
            total=0,
            finished_at=time.time()
        )
        return jsonify({"ok": False, "error": "Belum ada data. Upload dan proses data terlebih dahulu."})

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
        from pyod.models.ecod import ECOD

        body = request.get_json(force=True) or {}
        run_id = str(body.get("run_id") or f"grid-{int(time.time() * 1000)}")
        resume_requested = bool(body.get("resume"))

        # Grid params (dengan default — fine-grained)
        if_contamination_list  = body.get("if_contamination_list",  [0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05])
        if_n_estimators_list   = body.get("if_n_estimators_list",   [100, 150, 200, 250, 300])
        lof_contamination_list = body.get("lof_contamination_list", [0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05])
        lof_n_neighbors_list   = body.get("lof_n_neighbors_list",   [40, 50, 60, 70, 80, 90, 100])
        lof_metric_list        = body.get("lof_metric_list",        ["minkowski", "euclidean", "manhattan", "chebyshev"])
        ecod_contamination_list= body.get("ecod_contamination_list",[0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05])
        allowed_lof_metrics = {"minkowski", "euclidean", "manhattan", "chebyshev"}
        lof_metric_list = [str(m).strip().lower() for m in lof_metric_list if str(m).strip().lower() in allowed_lof_metrics]
        if not lof_metric_list:
            lof_metric_list = ["minkowski"]

        grid_config = {
            "if_contamination_list":   [float(v) for v in if_contamination_list],
            "if_n_estimators_list":    [int(v) for v in if_n_estimators_list],
            "lof_contamination_list":  [float(v) for v in lof_contamination_list],
            "lof_n_neighbors_list":    [int(v) for v in lof_n_neighbors_list],
            "lof_metric_list":         list(lof_metric_list),
            "ecod_contamination_list": [float(v) for v in ecod_contamination_list],
        }
        if_contamination_list = grid_config["if_contamination_list"]
        if_n_estimators_list = grid_config["if_n_estimators_list"]
        lof_contamination_list = grid_config["lof_contamination_list"]
        lof_n_neighbors_list = grid_config["lof_n_neighbors_list"]
        lof_metric_list = grid_config["lof_metric_list"]
        ecod_contamination_list = grid_config["ecod_contamination_list"]
        data_signature = _grid_search_data_signature()
        resume_checkpoint = None
        if resume_requested:
            checkpoint = _load_grid_search_checkpoint()
            if not checkpoint:
                return jsonify({"ok": False, "error": "Checkpoint grid search tidak ditemukan. Jalankan grid search baru."})
            if checkpoint.get("grid") != grid_config:
                return jsonify({"ok": False, "error": "Checkpoint tidak cocok dengan rentang grid saat ini. Samakan parameter grid atau jalankan grid baru."})
            if checkpoint.get("data_signature") != data_signature:
                return jsonify({"ok": False, "error": "Checkpoint tidak cocok dengan data/model saat ini. Data kemungkinan sudah diproses ulang."})
            resume_checkpoint = checkpoint
        else:
            _clear_grid_search_checkpoint()

        total_if = len(if_n_estimators_list) * len(if_contamination_list)
        total_lof = len(lof_metric_list) * len(lof_n_neighbors_list) * len(lof_contamination_list)
        total_ecod = len(ecod_contamination_list)
        total_steps = max(total_if + total_lof + total_ecod + 3, 1)
        done_steps = 0
        start_time = time.time()

        def update_progress(phase, message, current=None, status="running"):
            percent = round((done_steps / total_steps) * 100, 1)
            _set_grid_search_progress(
                run_id=run_id,
                running=(status == "running"),
                status=status,
                phase=phase,
                message=message,
                percent=percent,
                current=done_steps if current is None else current,
                total=total_steps,
                started_at=start_time,
                finished_at=None if status == "running" else time.time()
            )

        def mark_step_done(phase, message):
            nonlocal done_steps
            done_steps += 1
            update_progress(phase, message)

        _set_grid_search_progress(
            run_id=run_id,
            running=True,
            status="running",
            phase="Persiapan",
            message=f"Menyiapkan grid search: IF {total_if} kombinasi, LOF {total_lof} kombinasi, ECOD {total_ecod} kombinasi.",
            percent=0,
            current=0,
            total=total_steps,
            started_at=start_time,
            finished_at=None
        )

        # Load X_scaled & y_true
        update_progress("Persiapan", "Memuat X_scaled.npy dan df_hasil.pkl.")
        d = MODEL_DIR
        X_scaled = np.load(f"{d}/X_scaled.npy")
        df_orig  = pd.read_pickle(f"{d}/df_hasil.pkl")
        if "label_pseudo" not in df_orig.columns:
            update_progress("Validasi", "Kolom label_pseudo tidak ditemukan, grid search dibatalkan.", status="failed")
            return jsonify({"ok": False, "error": "Kolom label_pseudo tidak ditemukan — tidak bisa evaluasi F1."})
        y_true = df_orig["label_pseudo"].values

        # ───── BASELINE (model saat ini) untuk perbandingan ─────
        baseline = {}
        if all(c in df_orig.columns for c in ["if_pred", "lof_pred", "ecod_pred", "if_score", "lof_score", "ecod_score"]):
            for name, pred_col, score_col in [
                ("IF", "if_pred", "if_score"),
                ("LOF", "lof_pred", "lof_score"),
                ("ECOD", "ecod_pred", "ecod_score"),
            ]:
                pred = df_orig[pred_col].values
                score = df_orig[score_col].values
                baseline[name] = _eval_preds(y_true, pred, score)
            # Ensemble baseline
            ens_pred = df_orig["anomali_final"].values if "anomali_final" in df_orig.columns else (df_orig["if_pred"] + df_orig["lof_pred"] + df_orig["ecod_pred"] >= 2).astype(int).values
            ens_score = ((df_orig["if_score"] + df_orig["lof_score"] + df_orig["ecod_score"]) / 3).values
            baseline["Ensemble"] = _eval_preds(y_true, ens_pred, ens_score)

        results_if = list(resume_checkpoint.get("results_if", [])) if resume_checkpoint else []
        results_lof = list(resume_checkpoint.get("results_lof", [])) if resume_checkpoint else []
        results_ecod = list(resume_checkpoint.get("results_ecod", [])) if resume_checkpoint else []
        done_steps = len(results_if) + len(results_lof) + len(results_ecod)
        completed_if = {_if_grid_key(r["n_estimators"], r["contamination"]) for r in results_if}
        completed_lof = {_lof_grid_key(r.get("metric", "minkowski"), r["n_neighbors"], r["contamination"]) for r in results_lof}
        completed_ecod = {_ecod_grid_key(r["contamination"]) for r in results_ecod}

        def save_checkpoint(phase, message, status="running"):
            _save_grid_search_checkpoint({
                "run_id": run_id,
                "status": status,
                "phase": phase,
                "message": message,
                "grid": grid_config,
                "data_signature": data_signature,
                "n_samples": int(len(X_scaled)),
                "n_anomali_true": int(y_true.sum()),
                "results_if": results_if,
                "results_lof": results_lof,
                "results_ecod": results_ecod,
                "baseline": baseline,
                "done_steps": done_steps,
                "total_steps": total_steps,
                "percent": round((done_steps / total_steps) * 100, 1),
                "started_at": start_time,
                "updated_at": time.time(),
            })

        if resume_checkpoint:
            update_progress(
                "Resume",
                f"Melanjutkan checkpoint: {done_steps}/{total_steps} langkah sudah selesai."
            )
        save_checkpoint("Persiapan", "Checkpoint grid search siap.")

        # ───── ISOLATION FOREST ─────
        if_idx = 0
        for n_est in if_n_estimators_list:
            for cont in if_contamination_list:
                if_idx += 1
                if _if_grid_key(n_est, cont) in completed_if:
                    continue
                update_progress(
                    "Isolation Forest",
                    f"IF {if_idx}/{total_if}: n_estimators={int(n_est)}, contamination={float(cont)}"
                )
                m = IsolationForest(n_estimators=int(n_est), contamination=float(cont),
                                    random_state=42, n_jobs=-1)
                m.fit(X_scaled)
                pred  = (m.predict(X_scaled) == -1).astype(int)
                score = -m.score_samples(X_scaled)
                metrics = _eval_preds(y_true, pred, score)
                results_if.append({
                    "n_estimators":  int(n_est),
                    "contamination": float(cont),
                    **metrics
                })
                completed_if.add(_if_grid_key(n_est, cont))
                mark_step_done("Isolation Forest", f"Selesai IF {if_idx}/{total_if}.")
                save_checkpoint("Isolation Forest", f"Selesai IF {if_idx}/{total_if}.")

        # ───── LOF ─────
        lof_idx = 0
        for metric in lof_metric_list:
            for n_nb in lof_n_neighbors_list:
                for cont in lof_contamination_list:
                    lof_idx += 1
                    if _lof_grid_key(metric, n_nb, cont) in completed_lof:
                        continue
                    update_progress(
                        "Local Outlier Factor",
                        f"LOF {lof_idx}/{total_lof}: metric={metric}, n_neighbors={int(n_nb)}, contamination={float(cont)}"
                    )
                    m = LocalOutlierFactor(
                        n_neighbors=int(n_nb),
                        contamination=float(cont),
                        metric=metric,
                        n_jobs=-1
                    )
                    pred  = (m.fit_predict(X_scaled) == -1).astype(int)
                    score = -m.negative_outlier_factor_
                    metrics = _eval_preds(y_true, pred, score)
                    results_lof.append({
                        "metric":        metric,
                        "n_neighbors":   int(n_nb),
                        "contamination": float(cont),
                        **metrics
                    })
                    completed_lof.add(_lof_grid_key(metric, n_nb, cont))
                    mark_step_done("Local Outlier Factor", f"Selesai LOF {lof_idx}/{total_lof}.")
                    save_checkpoint("Local Outlier Factor", f"Selesai LOF {lof_idx}/{total_lof}.")

        # ───── ECOD ─────
        ecod_idx = 0
        for cont in ecod_contamination_list:
            ecod_idx += 1
            if _ecod_grid_key(cont) in completed_ecod:
                continue
            update_progress("ECOD", f"ECOD {ecod_idx}/{total_ecod}: contamination={float(cont)}")
            m = ECOD(contamination=float(cont))
            m.fit(X_scaled)
            pred  = m.labels_
            score = m.decision_scores_
            metrics = _eval_preds(y_true, pred, score)
            results_ecod.append({
                "contamination": float(cont),
                **metrics
            })
            completed_ecod.add(_ecod_grid_key(cont))
            mark_step_done("ECOD", f"Selesai ECOD {ecod_idx}/{total_ecod}.")
            save_checkpoint("ECOD", f"Selesai ECOD {ecod_idx}/{total_ecod}.")

        # Cari parameter terbaik (F1 tertinggi)
        best_if   = max(results_if,   key=lambda r: r["f1"])
        best_lof  = max(results_lof,  key=lambda r: r["f1"])
        best_ecod = max(results_ecod, key=lambda r: r["f1"])

        # ───── ENSEMBLE THRESHOLD OPTIMIZATION ─────
        # Jalankan model terbaik dan coba threshold 1, 2, 3
        update_progress("Ensemble", "Menghitung ulang Isolation Forest terbaik untuk ensemble.")
        best_if_m = IsolationForest(
            n_estimators=int(best_if["n_estimators"]),
            contamination=float(best_if["contamination"]),
            random_state=42, n_jobs=-1)
        best_if_m.fit(X_scaled)
        if_pred_b = (best_if_m.predict(X_scaled) == -1).astype(int)
        if_score_b = -best_if_m.score_samples(X_scaled)
        if_score_b = (if_score_b - if_score_b.min()) / (if_score_b.max() - if_score_b.min() + 1e-9)
        mark_step_done("Ensemble", "Selesai menghitung Isolation Forest terbaik.")

        update_progress("Ensemble", "Menghitung ulang LOF terbaik untuk ensemble.")
        best_lof_m = LocalOutlierFactor(
            n_neighbors=int(best_lof["n_neighbors"]),
            contamination=float(best_lof["contamination"]),
            metric=best_lof.get("metric", "minkowski"),
            n_jobs=-1)
        lof_pred_b = (best_lof_m.fit_predict(X_scaled) == -1).astype(int)
        lof_score_b = -best_lof_m.negative_outlier_factor_
        lof_score_b = (lof_score_b - lof_score_b.min()) / (lof_score_b.max() - lof_score_b.min() + 1e-9)
        mark_step_done("Ensemble", "Selesai menghitung LOF terbaik.")

        update_progress("Ensemble", "Menghitung ulang ECOD terbaik untuk ensemble.")
        best_ecod_m = ECOD(contamination=float(best_ecod["contamination"]))
        best_ecod_m.fit(X_scaled)
        ecod_pred_b = best_ecod_m.labels_
        ecod_score_b = best_ecod_m.decision_scores_
        ecod_score_b = (ecod_score_b - ecod_score_b.min()) / (ecod_score_b.max() - ecod_score_b.min() + 1e-9)
        mark_step_done("Ensemble", "Selesai menghitung ECOD terbaik.")

        vote_b = if_pred_b + lof_pred_b + ecod_pred_b
        ens_score_b = (if_score_b + lof_score_b + ecod_score_b) / 3

        ensemble_results = []
        for min_v in [1, 2, 3]:
            ens_pred = (vote_b >= min_v).astype(int)
            ensemble_results.append({
                "min_votes": min_v,
                "n_anomali": int(ens_pred.sum()),
                "pct": round(int(ens_pred.sum()) / len(y_true) * 100, 2),
                "f1": round(float(f1_score(y_true, ens_pred, zero_division=0)), 4),
                "precision": round(float(precision_score(y_true, ens_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, ens_pred, zero_division=0)), 4),
                "auc_pr": round(float(average_precision_score(y_true, ens_score_b)), 4),
            })
        best_ensemble = max(ensemble_results, key=lambda r: r["f1"])

        # Score distribution for visualization
        score_dist_normal = ens_score_b[y_true == 0].tolist()
        score_dist_anomali = ens_score_b[y_true == 1].tolist()
        # Sample for performance (max 2000 points each)
        if len(score_dist_normal) > 2000:
            score_dist_normal = list(np.random.choice(score_dist_normal, 2000, replace=False))
        if len(score_dist_anomali) > 2000:
            score_dist_anomali = list(np.random.choice(score_dist_anomali, 2000, replace=False))

        import datetime
        payload = {
            "timestamp":   datetime.datetime.now().strftime("%d %b %Y, %H:%M:%S"),
            "run_id":      run_id,
            "n_samples":   int(len(X_scaled)),
            "n_anomali_true": int(y_true.sum()),
            "grid": {
                "if_contamination_list":   if_contamination_list,
                "if_n_estimators_list":    if_n_estimators_list,
                "lof_contamination_list":  lof_contamination_list,
                "lof_n_neighbors_list":    lof_n_neighbors_list,
                "lof_metric_list":         lof_metric_list,
                "ecod_contamination_list": ecod_contamination_list,
            },
            "results_if":   results_if,
            "results_lof":  results_lof,
            "results_ecod": results_ecod,
            "best_if":      best_if,
            "best_lof":     best_lof,
            "best_ecod":    best_ecod,
            "ensemble_results": ensemble_results,
            "best_ensemble": best_ensemble,
            "score_dist_normal": [round(float(v), 4) for v in score_dist_normal],
            "score_dist_anomali": [round(float(v), 4) for v in score_dist_anomali],
            "baseline": baseline,
        }

        # Simpan hasil ke disk (pickle — konsisten dengan file model lain)
        with open(GRID_SEARCH_FILE, "wb") as f:
            pickle.dump(payload, f)
        _clear_grid_search_checkpoint()

        _set_grid_search_progress(
            run_id=run_id,
            running=False,
            status="complete",
            phase="Selesai",
            message="Grid search selesai. Hasil terbaik sudah disimpan.",
            percent=100,
            current=total_steps,
            total=total_steps,
            started_at=start_time,
            finished_at=time.time()
        )

        return jsonify({"ok": True, **payload})

    except Exception as e:
        import traceback
        try:
            if "save_checkpoint" in locals():
                save_checkpoint("Gagal", str(e), status="failed")
        except Exception:
            pass
        _set_grid_search_progress(
            running=False,
            status="failed",
            phase="Gagal",
            message=str(e),
            finished_at=time.time()
        )
        return jsonify({"ok": False, "error": str(e), "detail": traceback.format_exc()})


@app.route("/api/tuning/gridsearch/progress")
def api_tuning_gridsearch_progress():
    return jsonify({"ok": True, **_get_grid_search_progress()})


@app.route("/api/tuning/gridsearch/checkpoint")
def api_tuning_gridsearch_checkpoint():
    checkpoint = _load_grid_search_checkpoint()
    if not checkpoint:
        return jsonify({"ok": False, "error": "Tidak ada checkpoint grid search."})

    done_if = len(checkpoint.get("results_if", []))
    done_lof = len(checkpoint.get("results_lof", []))
    done_ecod = len(checkpoint.get("results_ecod", []))
    done_steps = int(checkpoint.get("done_steps", done_if + done_lof + done_ecod))
    total_steps = int(checkpoint.get("total_steps", max(done_steps, 1)))
    data_matches = checkpoint.get("data_signature") == _grid_search_data_signature()
    status = checkpoint.get("status", "running")
    resumable = data_matches and status != "complete" and done_steps < total_steps

    return jsonify({
        "ok": True,
        "resumable": bool(resumable),
        "data_matches": bool(data_matches),
        "status": status,
        "phase": checkpoint.get("phase", "-"),
        "message": checkpoint.get("message", "-"),
        "percent": round(done_steps / max(total_steps, 1) * 100, 1),
        "done_steps": done_steps,
        "total_steps": total_steps,
        "done_if": done_if,
        "done_lof": done_lof,
        "done_ecod": done_ecod,
        "grid": checkpoint.get("grid", {}),
        "updated_at": checkpoint.get("updated_at"),
    })


@app.route("/api/tuning/gridsearch/latest")
def api_tuning_gridsearch_latest():
    """
    Mengembalikan hasil grid search terakhir yang tersimpan (jika ada).
    Digunakan saat halaman dibuka agar user bisa langsung melihat hasil sebelumnya.
    """
    if not os.path.exists(GRID_SEARCH_FILE):
        return jsonify({"ok": False, "error": "Belum ada hasil grid search tersimpan."})
    try:
        with open(GRID_SEARCH_FILE, "rb") as f:
            payload = pickle.load(f)
        return jsonify({"ok": True, **payload})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/tuning/baseline")
def api_tuning_baseline():
    """
    Mengembalikan data baseline (model saat ini) untuk ditampilkan
    di halaman tuning SEBELUM user melakukan tuning.
    """
    if not models_exist():
        return jsonify({"ok": False, "error": "Belum ada data. Upload dan proses data terlebih dahulu."})

    try:
        (_, _, _, _, _, _, _, results, _, _, df) = load_models()

        n_total = len(df)

        # Jumlah anomali per algoritma
        n_if   = int(df["if_pred"].sum())   if "if_pred"   in df.columns else 0
        n_lof  = int(df["lof_pred"].sum())  if "lof_pred"  in df.columns else 0
        n_ecod = int(df["ecod_pred"].sum()) if "ecod_pred" in df.columns else 0
        n_ens  = int(df["anomali_final"].sum()) if "anomali_final" in df.columns else 0

        # Vote distribution
        vote_count = df["vote_count"] if "vote_count" in df.columns else (
            df.get("if_pred", 0) + df.get("lof_pred", 0) + df.get("ecod_pred", 0)
        )
        vote_dist = {
            "vote_3": int((vote_count == 3).sum()),
            "vote_2": int((vote_count == 2).sum()),
            "vote_1": int((vote_count == 1).sum()),
            "vote_0": int((vote_count == 0).sum()),
        }

        # Agreement matrix
        if all(c in df.columns for c in ["if_pred", "lof_pred", "ecod_pred"]):
            if_pred   = df["if_pred"].values
            lof_pred  = df["lof_pred"].values
            ecod_pred = df["ecod_pred"].values
            agree_if_lof   = int(((if_pred == 1) & (lof_pred == 1)).sum())
            agree_if_ecod  = int(((if_pred == 1) & (ecod_pred == 1)).sum())
            agree_lof_ecod = int(((lof_pred == 1) & (ecod_pred == 1)).sum())
            agree_all      = int(((if_pred == 1) & (lof_pred == 1) & (ecod_pred == 1)).sum())
        else:
            agree_if_lof = agree_if_ecod = agree_lof_ecod = agree_all = 0

        return jsonify({
            "ok": True,
            "n_total": n_total,
            "results": {
                "if":   {"n": n_if,   "pct": round(n_if/n_total*100, 2)},
                "lof":  {"n": n_lof,  "pct": round(n_lof/n_total*100, 2)},
                "ecod": {"n": n_ecod, "pct": round(n_ecod/n_total*100, 2)},
                "ensemble": {"n": n_ens, "pct": round(n_ens/n_total*100, 2)},
            },
            "vote_dist": vote_dist,
            "agreement": {
                "if_lof":    {"n": agree_if_lof,   "pct": round(agree_if_lof/n_total*100, 2)},
                "if_ecod":   {"n": agree_if_ecod,  "pct": round(agree_if_ecod/n_total*100, 2)},
                "lof_ecod":  {"n": agree_lof_ecod, "pct": round(agree_lof_ecod/n_total*100, 2)},
                "all_three": {"n": agree_all,      "pct": round(agree_all/n_total*100, 2)},
            },
            "metrics": results,
        })

    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e), "detail": traceback.format_exc()})


@app.route("/api/tuning", methods=["POST"])
def api_tuning():
    """
    Jalankan ulang TAHAP 15-20 (IF, LOF, ECOD, Ensemble, SHAP)
    dengan parameter custom dari user, dan SIMPAN hasilnya secara otomatis.
    Mengembalikan metrik perbandingan dan ringkasan hasil.
    """
    if not models_exist():
        return jsonify({"ok": False, "error": "Belum ada data. Upload dan proses data terlebih dahulu."})

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        from pyod.models.ecod import ECOD

        body = request.get_json(force=True) or {}

        # --- Ambil parameter dari request ---
        # Isolation Forest
        if_n_estimators   = int(body.get("if_n_estimators", 200))
        if_contamination  = float(body.get("if_contamination", 0.015))
        if_max_samples    = body.get("if_max_samples", "auto")
        if if_max_samples not in ("auto",):
            if_max_samples = int(if_max_samples)

        # LOF
        lof_n_neighbors   = int(body.get("lof_n_neighbors", 70))
        lof_contamination = float(body.get("lof_contamination", 0.005))
        lof_metric        = body.get("lof_metric", "minkowski")

        # ECOD
        ecod_contamination = float(body.get("ecod_contamination", 0.005))

        # Ensemble threshold
        ensemble_min_votes = int(body.get("ensemble_min_votes", 2))

        # Validasi batas wajar
        if_n_estimators   = max(10,  min(if_n_estimators,  1000))
        if_contamination  = max(0.001, min(if_contamination,  0.5))
        lof_n_neighbors   = max(2,   min(lof_n_neighbors,   200))
        lof_contamination = max(0.001, min(lof_contamination,  0.5))
        ecod_contamination= max(0.001, min(ecod_contamination, 0.5))
        ensemble_min_votes= max(1,   min(ensemble_min_votes,  3))

        # --- Load X_scaled dari model yang sudah tersimpan ---
        (_, _, _, _, _, _, _, _, _, _, df_orig) = load_models()
        d = MODEL_DIR
        x_scaled_path = f"{d}/X_scaled.npy"
        if not os.path.exists(x_scaled_path):
            return jsonify({"ok": False, "error": "X_scaled.npy tidak ditemukan. Silakan re-upload dan proses ulang data."})
        X_scaled = np.load(x_scaled_path)

        n_total = len(X_scaled)

        # --- Isolation Forest ---
        if_model = IsolationForest(
            n_estimators=if_n_estimators,
            contamination=if_contamination,
            max_samples=if_max_samples,
            random_state=42
        )
        if_model.fit(X_scaled)
        if_pred  = (if_model.predict(X_scaled) == -1).astype(int)
        if_score = -if_model.score_samples(X_scaled)
        if_score = (if_score - if_score.min()) / (if_score.max() - if_score.min() + 1e-9)
        n_if = int(if_pred.sum())

        # --- LOF ---
        lof_model = LocalOutlierFactor(
            n_neighbors=lof_n_neighbors,
            contamination=lof_contamination,
            metric=lof_metric,
            novelty=False
        )
        lof_pred  = (lof_model.fit_predict(X_scaled) == -1).astype(int)
        lof_score = -lof_model.negative_outlier_factor_
        lof_score = (lof_score - lof_score.min()) / (lof_score.max() - lof_score.min() + 1e-9)
        n_lof = int(lof_pred.sum())

        # --- ECOD ---
        ecod_model = ECOD(contamination=ecod_contamination)
        ecod_model.fit(X_scaled)
        ecod_pred  = ecod_model.labels_
        ecod_score = ecod_model.decision_scores_
        ecod_score = (ecod_score - ecod_score.min()) / (ecod_score.max() - ecod_score.min() + 1e-9)
        n_ecod = int(ecod_pred.sum())

        # --- Ensemble ---
        vote_count    = if_pred + lof_pred + ecod_pred
        anomali_final = (vote_count >= ensemble_min_votes).astype(int)
        n_final       = int(anomali_final.sum())
        pct_final     = round(n_final / n_total * 100, 2)

        # Perbandingan dengan hasil sebelumnya (dari df_hasil.pkl)
        prev_if   = int(df_orig["if_pred"].sum())   if "if_pred"   in df_orig.columns else None
        prev_lof  = int(df_orig["lof_pred"].sum())  if "lof_pred"  in df_orig.columns else None
        prev_ecod = int(df_orig["ecod_pred"].sum()) if "ecod_pred" in df_orig.columns else None
        prev_ens  = int(df_orig["anomali_final"].sum()) if "anomali_final" in df_orig.columns else None

        # --- Update df_orig dengan hasil terbaru ---
        df_orig["if_pred"] = if_pred
        df_orig["if_score"] = if_score.round(4)
        df_orig["lof_pred"] = lof_pred
        df_orig["lof_score"] = lof_score.round(4)
        df_orig["ecod_pred"] = ecod_pred
        df_orig["ecod_score"] = ecod_score.round(4)
        df_orig["vote_count"] = vote_count
        df_orig["anomali_final"] = anomali_final
        df_orig["ensemble_score"] = ((if_score + lof_score + ecod_score) / 3).round(4)

        # --- Re-Evaluate Metrik ---
        from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix
        y_true = df_orig["label_pseudo"].values if "label_pseudo" in df_orig.columns else np.zeros(len(df_orig))
        new_results = {}
        eval_pairs = {
            "IF"      : (if_pred,                    if_score),
            "LOF"     : (lof_pred,                   lof_score),
            "ECOD"    : (ecod_pred,                  ecod_score),
            "Ensemble": (df_orig["anomali_final"].values,  df_orig["ensemble_score"].values),
        }
        for name, (pred, score) in eval_pairs.items():
            cm           = confusion_matrix(y_true, pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            new_results[name] = {
                "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
                "recall"   : round(float(recall_score(y_true,    pred, zero_division=0)), 4),
                "f1"       : round(float(f1_score(y_true,        pred, zero_division=0)), 4),
                "auc"      : round(float(average_precision_score(y_true, score)), 4),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            }

        # --- Re-calculate SHAP ---
        try:
            import shap
        except Exception:
            shap = None
        (_, _, _, _, _, _, feature_names, _, _, _, _) = load_models()
        explainer    = shap.TreeExplainer(if_model)
        shap_values  = explainer.shap_values(X_scaled)
        expected_val = float(np.mean(explainer.expected_value))

        shap_abs = np.abs(shap_values)
        top1_idx = shap_abs.argmax(axis=1)
        top2_idx = np.argsort(shap_abs, axis=1)[:, -2]

        df_orig["shap_top1_fitur"] = [feature_names[i] for i in top1_idx]
        df_orig["shap_top1_nilai"] = shap_values[np.arange(len(shap_values)), top1_idx].round(4)
        df_orig["shap_top2_fitur"] = [feature_names[i] for i in top2_idx]
        df_orig["shap_top2_nilai"] = shap_values[np.arange(len(shap_values)), top2_idx].round(4)

        df_orig["alasan_utama"]   = df_orig["shap_top1_fitur"].apply(get_narasi)
        df_orig["alasan_kedua"]   = df_orig["shap_top2_fitur"].apply(get_narasi)

        # --- Simpan Semua Ke Disk ---
        with open(f"{d}/if_model.pkl",         "wb") as f: pickle.dump(if_model,         f)
        with open(f"{d}/lof_model.pkl",        "wb") as f: pickle.dump(lof_model,        f)
        with open(f"{d}/ecod_model.pkl",       "wb") as f: pickle.dump(ecod_model,       f)
        with open(f"{d}/expected_val.pkl",     "wb") as f: pickle.dump(expected_val,     f)
        with open(f"{d}/results.pkl",          "wb") as f: pickle.dump(new_results,      f)
        np.save(f"{d}/shap_values.npy", shap_values)
        df_orig.to_pickle(f"{d}/df_hasil.pkl")

        # Hapus cache SHAP lama
        for m in ["if", "lof", "ecod", "ensemble"]:
            cache_f = os.path.join(d, f"shap_cache_{m}.pkl")
            if os.path.exists(cache_f):
                os.remove(cache_f)

        # Vote breakdown
        vote_dist = {
            "vote_3": int((vote_count == 3).sum()),
            "vote_2": int((vote_count == 2).sum()),
            "vote_1": int((vote_count == 1).sum()),
            "vote_0": int((vote_count == 0).sum()),
        }

        # Agreement matrix (berapa persen IF & LOF setuju, dst)
        n = n_total
        agree_if_lof  = int(((if_pred == 1) & (lof_pred == 1)).sum())
        agree_if_ecod = int(((if_pred == 1) & (ecod_pred == 1)).sum())
        agree_lof_ecod= int(((lof_pred == 1) & (ecod_pred == 1)).sum())
        agree_all     = int(((if_pred == 1) & (lof_pred == 1) & (ecod_pred == 1)).sum())

        return jsonify({
            "ok": True,
            "n_total": n_total,
            "params": {
                "if_n_estimators":   if_n_estimators,
                "if_contamination":  if_contamination,
                "if_max_samples":    if_max_samples,
                "lof_n_neighbors":   lof_n_neighbors,
                "lof_contamination": lof_contamination,
                "lof_metric":        lof_metric,
                "ecod_contamination":ecod_contamination,
                "ensemble_min_votes":ensemble_min_votes,
            },
            "results": {
                "if":   {"n": n_if,   "pct": round(n_if/n_total*100,2),   "prev": prev_if},
                "lof":  {"n": n_lof,  "pct": round(n_lof/n_total*100,2),  "prev": prev_lof},
                "ecod": {"n": n_ecod, "pct": round(n_ecod/n_total*100,2), "prev": prev_ecod},
                "ensemble": {"n": n_final, "pct": pct_final, "prev": prev_ens,
                             "min_votes": ensemble_min_votes},
            },
            "vote_dist": vote_dist,
            "agreement": {
                "if_lof":   {"n": agree_if_lof,   "pct": round(agree_if_lof/n_total*100,2)},
                "if_ecod":  {"n": agree_if_ecod,  "pct": round(agree_if_ecod/n_total*100,2)},
                "lof_ecod": {"n": agree_lof_ecod, "pct": round(agree_lof_ecod/n_total*100,2)},
                "all_three":{"n": agree_all,       "pct": round(agree_all/n_total*100,2)},
            }
        })

    except FileNotFoundError:
        return jsonify({"ok": False, "error": "X_scaled.npy tidak ditemukan. Silakan re-upload dan proses ulang data."})
    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e), "detail": traceback.format_exc()})



@app.route("/api/export/anomali")
def export_anomali():
    """Export laporan deteksi anomali lengkap ke Excel multi-sheet."""
    if not models_exist():
        return "Data tidak tersedia", 404

    (_, _, _, _, _, _, _, results, _, _, df) = load_models()
    df_an = df[df["anomali_final"] == 1].copy().sort_values("ensemble_score", ascending=False)

    cols = ["karyawan_id", "id_skpd", "tanggal_kirim", "lat", "long",
            "ensemble_score", "if_pred", "lof_pred", "ecod_pred", "vote_count",
            "shap_top1_fitur", "shap_top1_nilai", "shap_top2_fitur", "shap_top2_nilai",
            "label_pseudo", "alasan_utama", "alasan_kedua"]
    cols = [c for c in cols if c in df_an.columns]

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df_an[cols].to_excel(w, sheet_name="Semua Anomali", index=False)
        rekap = (
            df_an.groupby("id_skpd")
            .agg(total_anomali  =("karyawan_id", "count"),
                 skor_rata_rata =("ensemble_score", "mean"))
            .round(4).sort_values("total_anomali", ascending=False).reset_index()
        )
        rekap.to_excel(w, sheet_name="Rekap per SKPD", index=False)
        metrik_rows = [{"Algoritma": alg, **m} for alg, m in results.items()]
        pd.DataFrame(metrik_rows).to_excel(w, sheet_name="Metrik", index=False)
    buf.seek(0)
    return send_file(buf,
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     as_attachment=True,
                     download_name="laporan_deteksi_anomali.xlsx")


@app.route("/api/export/detail")
def export_detail():
    """Export tabel detail anomali ke Excel."""
    if not models_exist():
        return "Data tidak tersedia", 404

    (_, _, _, _, _, _, _, _, _, _, df) = load_models()
    df_an = df[df["anomali_final"] == 1].copy().sort_values("ensemble_score", ascending=False)
    cols  = ["karyawan_id", "id_skpd", "tanggal_kirim", "ensemble_score",
             "alasan_utama", "alasan_kedua",
             "if_pred", "lof_pred", "ecod_pred"]
    cols  = [c for c in cols if c in df_an.columns]

    buf = io.BytesIO()
    df_an[cols].to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return send_file(buf,
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     as_attachment=True,
                     download_name="detail_anomali.xlsx")


# ══════════════════════════════════════════════════════════════════
#  API STATUS — timestamp data terakhir diproses
# ══════════════════════════════════════════════════════════════════

@app.route("/api/status")
def api_status():
    """Kembalikan timestamp data terakhir diproses dan info singkat."""
    ts = get_last_processed()
    info = {}
    if ts and models_exist():
        try:
            df = pd.read_pickle(os.path.join(MODEL_DIR, "df_hasil.pkl"))
            info = {
                "n_rows"    : len(df),
                "n_anomali" : int((df["anomali_final"] == 1).sum()),
                "periode_start": str(df["tanggal_kirim"].min())[:10] if "tanggal_kirim" in df.columns else "-",
                "periode_end"  : str(df["tanggal_kirim"].max())[:10] if "tanggal_kirim" in df.columns else "-",
            }
        except Exception:
            pass
    return jsonify({"ok": True, "last_processed": ts, **info})


# ══════════════════════════════════════════════════════════════════
#  PROFIL PEGAWAI INDIVIDUAL
# ══════════════════════════════════════════════════════════════════

@app.route("/profil")
@app.route("/profil/<karyawan_id>")
def profil_page(karyawan_id=None):
    return render_template("profil.html",
                           nama_instansi=NAMA_INSTANSI,
                           has_model=models_exist(),
                           karyawan_id=karyawan_id)


@app.route("/api/profil/list")
def api_profil_list():
    """Return list of all karyawan_id for the employee selector."""
    if not models_exist():
        return jsonify({"ok": False, "error": "Belum ada data."})
    try:
        df = pd.read_pickle(os.path.join(MODEL_DIR, "df_hasil.pkl"))
        employees = sorted(df["karyawan_id"].dropna().unique().astype(str).tolist())
        return jsonify({"ok": True, "employees": employees})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/profil/<karyawan_id>")
def api_profil(karyawan_id):
    """Return all profile data for a given employee."""
    if not models_exist():
        return jsonify({"ok": False, "error": "Belum ada data yang diproses."})

    try:
        (_, _, _, _, shap_values, expected_val,
         feature_names, results, _, _, df) = load_models()

        # Filter data pegawai
        df["karyawan_id"] = df["karyawan_id"].astype(str)

        # Rekonstruksi kolom jenis jika sudah di-OHE
        if "jenis" not in df.columns:
            if "jenis_P" in df.columns:
                df["jenis"] = df["jenis_P"].apply(lambda x: "P" if x == 1 else "M")
            elif "jenis_M" in df.columns:
                df["jenis"] = df["jenis_M"].apply(lambda x: "M" if x == 1 else "P")

        df_emp = df[df["karyawan_id"] == str(karyawan_id)]

        if df_emp.empty:
            return jsonify({"ok": False, "error": f"Pegawai '{karyawan_id}' tidak ditemukan."})

        # ── Summary card ──
        total_records = len(df_emp)
        total_anomali = int((df_emp["anomali_final"] == 1).sum())
        pct_anomali = round(total_anomali / total_records * 100, 1) if total_records > 0 else 0
        skpd = str(df_emp["id_skpd"].mode().iloc[0]) if "id_skpd" in df_emp.columns else "-"

        # ── Attendance history table ──
        history = []
        for _, row in df_emp.sort_values("tanggal_kirim").iterrows():
            history.append({
                "tanggal": str(row.get("tanggal_kirim", ""))[:16],
                "jenis": str(row.get("jenis", "-")),
                "anomali": int(row.get("anomali_final", 0)),
                "ensemble_score": round(float(row.get("ensemble_score", 0)), 4),
                "jam_desimal": round(float(row.get("fe_jam_desimal", 0)), 2),
            })

        # ── Monthly anomaly trend ──
        df_emp_copy = df_emp.copy()
        df_emp_copy["bulan"] = pd.to_datetime(df_emp_copy["tanggal_kirim"]).dt.to_period("M").astype(str)
        monthly = df_emp_copy.groupby("bulan").agg(
            total=("anomali_final", "count"),
            anomali=("anomali_final", "sum")
        ).reset_index()
        monthly["pct"] = (monthly["anomali"] / monthly["total"] * 100).round(1)
        monthly_trend = monthly.to_dict(orient="records")

        # ── Clock pattern ──
        df_masuk = df_emp[df_emp["jenis"] == "M"] if "jenis" in df_emp.columns else pd.DataFrame()
        df_pulang = df_emp[df_emp["jenis"] == "P"] if "jenis" in df_emp.columns else pd.DataFrame()

        clock_in_times = df_masuk["fe_jam_desimal"].dropna().tolist() if "fe_jam_desimal" in df_emp.columns and len(df_masuk) > 0 else []
        clock_out_times = df_pulang["fe_jam_desimal"].dropna().tolist() if "fe_jam_desimal" in df_emp.columns and len(df_pulang) > 0 else []

        clock_stats = {
            "mean_in": round(float(np.mean(clock_in_times)), 2) if clock_in_times else None,
            "std_in": round(float(np.std(clock_in_times)), 2) if clock_in_times else None,
            "mean_out": round(float(np.mean(clock_out_times)), 2) if clock_out_times else None,
            "std_out": round(float(np.std(clock_out_times)), 2) if clock_out_times else None,
            "n_masuk": len(clock_in_times),
            "n_pulang": len(clock_out_times),
        }

        # ── Anomaly score over time ──
        score_timeline = []
        for _, row in df_emp.sort_values("tanggal_kirim").iterrows():
            score_timeline.append({
                "tanggal": str(row.get("tanggal_kirim", ""))[:10],
                "score": round(float(row.get("ensemble_score", 0)), 4),
                "anomali": int(row.get("anomali_final", 0)),
            })

        # Score trend (current month vs previous month)
        df_emp_copy["tgl"] = pd.to_datetime(df_emp_copy["tanggal_kirim"])
        if len(df_emp_copy) > 0:
            max_date = df_emp_copy["tgl"].max()
            cur_month_start = max_date.replace(day=1)
            prev_month_start = (cur_month_start - pd.DateOffset(months=1))
            cur_scores = df_emp_copy[df_emp_copy["tgl"] >= cur_month_start]["ensemble_score"]
            prev_scores = df_emp_copy[(df_emp_copy["tgl"] >= prev_month_start) & (df_emp_copy["tgl"] < cur_month_start)]["ensemble_score"]
            avg_cur = float(cur_scores.mean()) if len(cur_scores) > 0 else 0
            avg_prev = float(prev_scores.mean()) if len(prev_scores) > 0 else 0
            if avg_prev > 0:
                trend_dir = "increasing" if avg_cur > avg_prev * 1.05 else ("decreasing" if avg_cur < avg_prev * 0.95 else "stable")
            else:
                trend_dir = "stable"
        else:
            avg_cur, avg_prev, trend_dir = 0, 0, "stable"

        score_summary = {
            "avg_current": round(avg_cur, 4),
            "avg_previous": round(avg_prev, 4),
            "trend": trend_dir,
        }

        # ── SHAP explanations ──
        df_anomali = df_emp[df_emp["anomali_final"] == 1]
        shap_reasons = []
        shap_feature_importance = []

        if len(df_anomali) > 0 and "shap_top1_fitur" in df.columns:
            # Top 5 frequent reasons
            freq = df_anomali["shap_top1_fitur"].value_counts().head(5)
            shap_reasons = [
                {"fitur": fn, "narasi": get_narasi(fn), "jumlah": int(cnt)}
                for fn, cnt in freq.items()
            ]

            # Average SHAP per feature for this employee's anomalies
            emp_indices = df_anomali.index.tolist()
            # Map to positional indices in the full df
            full_indices = df.index.tolist()
            pos_indices = [full_indices.index(idx) for idx in emp_indices if idx in full_indices]

            if pos_indices and len(shap_values) > 0:
                emp_shap = shap_values[pos_indices]
                mean_abs_shap = np.abs(emp_shap).mean(axis=0)
                top_idx = np.argsort(mean_abs_shap)[::-1][:10]
                for i in top_idx:
                    if i < len(feature_names):
                        shap_feature_importance.append({
                            "fitur": feature_names[i],
                            "narasi": get_narasi(feature_names[i]),
                            "nilai": round(float(mean_abs_shap[i]), 6),
                        })

        # ── Peer comparison ──
        df_skpd = df[df["id_skpd"].astype(str) == skpd]
        peer_total = len(df_skpd["karyawan_id"].unique())
        peer_avg_anomali_pct = 0
        peer_scores = []
        emp_rank = 0

        if len(df_skpd) > 0:
            peer_stats = df_skpd.groupby("karyawan_id").agg(
                total=("anomali_final", "count"),
                anomali=("anomali_final", "sum")
            ).reset_index()
            peer_stats["pct"] = (peer_stats["anomali"] / peer_stats["total"] * 100).round(1)
            peer_avg_anomali_pct = round(float(peer_stats["pct"].mean()), 1)

            # Rank
            peer_stats_sorted = peer_stats.sort_values("anomali", ascending=False).reset_index(drop=True)
            rank_match = peer_stats_sorted[peer_stats_sorted["karyawan_id"].astype(str) == str(karyawan_id)]
            emp_rank = int(rank_match.index[0]) + 1 if len(rank_match) > 0 else 0

            # Peer score distribution (for box plot)
            peer_avg_scores = df_skpd.groupby("karyawan_id")["ensemble_score"].mean()
            peer_scores = [round(float(v), 4) for v in peer_avg_scores.tolist()]

        # Peer clock comparison
        peer_clock = {"mean_in": None, "mean_out": None, "pct_anomali": peer_avg_anomali_pct}
        if "fe_jam_desimal" in df_skpd.columns and "jenis" in df_skpd.columns:
            peer_masuk = df_skpd[df_skpd["jenis"] == "M"]["fe_jam_desimal"].dropna()
            peer_pulang = df_skpd[df_skpd["jenis"] == "P"]["fe_jam_desimal"].dropna()
            if len(peer_masuk) > 0:
                peer_clock["mean_in"] = round(float(peer_masuk.mean()), 2)
            if len(peer_pulang) > 0:
                peer_clock["mean_out"] = round(float(peer_pulang.mean()), 2)

        return jsonify({
            "ok": True,
            "karyawan_id": karyawan_id,
            "skpd": skpd,
            "summary": {
                "total_records": total_records,
                "total_anomali": total_anomali,
                "pct_anomali": pct_anomali,
            },
            "history": history,
            "monthly_trend": monthly_trend,
            "clock_in_times": [round(v, 2) for v in clock_in_times],
            "clock_out_times": [round(v, 2) for v in clock_out_times],
            "clock_stats": clock_stats,
            "score_timeline": score_timeline,
            "score_summary": score_summary,
            "shap_reasons": shap_reasons,
            "shap_feature_importance": shap_feature_importance,
            "peer": {
                "total_pegawai": peer_total,
                "avg_anomali_pct": peer_avg_anomali_pct,
                "emp_anomali_pct": pct_anomali,
                "emp_rank": emp_rank,
                "peer_scores": peer_scores,
                "emp_avg_score": round(float(df_emp["ensemble_score"].mean()), 4),
                "peer_clock": peer_clock,
            },
        })

    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e), "detail": traceback.format_exc()})


# ══════════════════════════════════════════════════════════════════
#  API: KASUS ANOMALI + CLUSTER DATA (untuk peta side-by-side)
# ══════════════════════════════════════════════════════════════════

@app.route("/api/anomali_kasus")
def api_anomali_kasus():
    """Kasus anomali spesifik + data cluster untuk peta side-by-side."""
    if not models_exist():
        return jsonify({"ok": False, "error": "Belum ada data."})
    try:
        df = pd.read_pickle(os.path.join(MODEL_DIR, "df_hasil.pkl"))
        if "jenis" not in df.columns and "jenis_P" in df.columns:
            df["jenis"] = df["jenis_P"].apply(lambda x: "P" if x == 1 else "M")
        df_anomali = df[df["anomali_final"] == 1].copy() if "anomali_final" in df.columns else df.copy()

        # Kasus DBSCAN-only
        kasus_dbscan = []
        if "fe_dbscan_only" in df_anomali.columns:
            df_db = df_anomali[
                (df_anomali["fe_dbscan_only"] == 1) &
                df_anomali["lat"].notna() &
                df_anomali["long"].notna()
            ].copy()
            kejadian_per_pegawai = df_db.groupby("karyawan_id").size().to_dict()
            df_db = df_db.sort_values(["karyawan_id", "tanggal_kirim"])
            for _, r in df_db.iterrows():
                kid = r.get("karyawan_id", "-")
                kasus_dbscan.append({"karyawan_id":str(kid),"id_skpd":str(r.get("id_skpd","-")),
                    "tanggal":str(r.get("tanggal_kirim",""))[:16],"lat":round(float(r.get("lat",0)),6),
                    "long":round(float(r.get("long",0)),6),"catatan":str(r.get("catatan","-")) if "catatan" in df.columns else "-",
                    "jumlah_kejadian":int(kejadian_per_pegawai.get(kid, 1)),"ensemble_score":round(float(r.get("ensemble_score",0)),4),
                    "dbscan_cluster":int(r.get("dbscan_cluster",-1)),
                    "cluster_id":int(r.get("cluster_id",-1))})

        # Kasus Absen Lompat
        kasus_lompat = []
        if "fe_absen_lompat" in df_anomali.columns and "lat_masuk" in df_anomali.columns:
            df_lp = df_anomali[
                (df_anomali["fe_absen_lompat"] == 1) &
                df_anomali["lat_masuk"].notna() &
                df_anomali["long_masuk"].notna() &
                df_anomali["lat_pulang"].notna() &
                df_anomali["long_pulang"].notna()
            ].sort_values("fe_jarak_masuk_pulang", ascending=False)
            for _, r in df_lp.iterrows():
                kasus_lompat.append({"karyawan_id":str(r.get("karyawan_id","-")),"id_skpd":str(r.get("id_skpd","-")),
                    "tanggal":str(r.get("tanggal_kirim",""))[:10],
                    "jam_masuk":str(r.get("jam_masuk_str","-")),
                    "jam_pulang":str(r.get("jam_pulang_str","-")),
                    "lat_masuk":round(float(r.get("lat_masuk",0)),6),"long_masuk":round(float(r.get("long_masuk",0)),6),
                    "lat_pulang":round(float(r.get("lat_pulang",0)),6),"long_pulang":round(float(r.get("long_pulang",0)),6),
                    "jarak_km":round(float(r.get("fe_jarak_masuk_pulang",0)),2),
                    "ensemble_score":round(float(r.get("ensemble_score",0)),4)})

        # Cluster data untuk side-by-side: kirim semua titik valid.
        cluster_points = []
        if "dbscan_cluster" in df_anomali.columns and "cluster_id" in df_anomali.columns:
            df_cluster = df_anomali[df_anomali["lat"].notna() & df_anomali["long"].notna()]
            for _, r in df_cluster.iterrows():
                cluster_points.append({
                    "lat":round(float(r["lat"]),6),"long":round(float(r["long"]),6),
                    "db_cluster":int(r.get("dbscan_cluster",-1)),
                    "st_cluster":int(r.get("cluster_id",-1)),
                    "kid":str(r.get("karyawan_id","-")),
                })

        return jsonify({"ok":True,"kasus_dbscan":kasus_dbscan,"kasus_lompat":kasus_lompat,
                        "cluster_points":cluster_points})
    except Exception as e:
        import traceback
        return jsonify({"ok":False,"error":str(e),"detail":traceback.format_exc()})


@app.route("/api/cluster_detail")
def api_cluster_detail():
    """Detail absensi di dalam suatu cluster."""
    if not models_exist():
        return jsonify({"ok": False, "error": "Belum ada data."})
    try:
        cluster_type = request.args.get("type", "dbscan")  # 'dbscan' atau 'stdbscan'
        cluster_id = request.args.get("cluster_id")
        skpd_filter = request.args.get("skpd", "")
        kid_filter = request.args.get("kid", "")

        if cluster_id is None:
            return jsonify({"ok": False, "error": "Parameter cluster_id diperlukan."})

        cluster_id = int(cluster_id)

        df = pd.read_pickle(os.path.join(MODEL_DIR, "df_hasil.pkl"))

        # Rekonstruksi kolom jenis jika sudah di-OHE
        if "jenis" not in df.columns:
            if "jenis_P" in df.columns:
                df["jenis"] = df["jenis_P"].apply(lambda x: "P" if x == 1 else "M")
            elif "jenis_M" in df.columns:
                df["jenis"] = df["jenis_M"].apply(lambda x: "M" if x == 1 else "P")

        # Filter berdasarkan tipe cluster
        col = "dbscan_cluster" if cluster_type == "dbscan" else "cluster_id"
        if col not in df.columns:
            return jsonify({"ok": False, "error": f"Kolom {col} tidak ada di data."})

        df_cluster = df[df[col] == cluster_id]

        # Filter SKPD jika diberikan
        if skpd_filter:
            df_cluster = df_cluster[df_cluster["id_skpd"].astype(str) == skpd_filter]

        # Filter karyawan jika diberikan
        if kid_filter:
            df_cluster = df_cluster[df_cluster["karyawan_id"].astype(str) == kid_filter]

        if df_cluster.empty:
            return jsonify({"ok": True, "records": [], "total": 0,
                            "cluster_type": cluster_type, "cluster_id": cluster_id})

        # Urutkan berdasarkan tanggal
        df_cluster = df_cluster.sort_values("tanggal_kirim", ascending=False)

        records = []
        for _, r in df_cluster.iterrows():
            records.append({
                "karyawan_id": str(r.get("karyawan_id", "-")),
                "id_skpd": str(r.get("id_skpd", "-")),
                "tanggal": str(r.get("tanggal_kirim", ""))[:16],
                "jenis": str(r.get("jenis", "-")) if "jenis" in df.columns else "-",
                "lat": round(float(r.get("lat", 0)), 6),
                "long": round(float(r.get("long", 0)), 6),
                "ensemble_score": round(float(r.get("ensemble_score", 0)), 4),
                "anomali_final": int(r.get("anomali_final", 0)),
            })

        # Ringkasan cluster
        n_pegawai = df_cluster["karyawan_id"].nunique()
        n_anomali = int((df_cluster["anomali_final"] == 1).sum())
        tanggal_list = sorted(df_cluster["tanggal_kirim"].dropna().astype(str).unique().tolist())

        return jsonify({
            "ok": True,
            "cluster_type": cluster_type,
            "cluster_id": cluster_id,
            "total": len(records),
            "n_pegawai": n_pegawai,
            "n_anomali": n_anomali,
            "tanggal_range": {
                "min": tanggal_list[0][:10] if tanggal_list else "-",
                "max": tanggal_list[-1][:10] if tanggal_list else "-",
            },
            "records": records,
        })
    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e), "detail": traceback.format_exc()})


# ══════════════════════════════════════════════════════════════════
#  PERBANDINGAN ANTAR-PERIODE
# ══════════════════════════════════════════════════════════════════

@app.route("/perbandingan")
def perbandingan_page():
    return render_template("perbandingan.html",
                           nama_instansi=NAMA_INSTANSI,
                           has_model=models_exist())


@app.route("/api/perbandingan_bulanan")
def api_perbandingan_bulanan():
    """
    Multi-period monthly comparison — otomatis deteksi bulan yang ada di data.
    Default: semua bulan yang tersedia. Bisa difilter via ?months=9,10,11
    """
    if not models_exist():
        return jsonify({"ok": False, "error": "Belum ada data yang diproses."})

    try:
        df = pd.read_pickle(os.path.join(MODEL_DIR, "df_hasil.pkl"))
        df["tanggal_kirim"] = pd.to_datetime(df["tanggal_kirim"])

        # Deteksi bulan-bulan yang tersedia di data
        df["_bulan"] = df["tanggal_kirim"].dt.to_period("M")
        available_months = sorted(df["_bulan"].unique())

        # Filter bulan tertentu jika diminta
        months_param = request.args.get("months", "")
        if months_param:
            try:
                month_nums = [int(m.strip()) for m in months_param.split(",")]
                available_months = [m for m in available_months if m.month in month_nums]
            except ValueError:
                pass

        if not available_months:
            return jsonify({"ok": False, "error": "Tidak ada data pada bulan yang dipilih."})

        # ── Per-month summary ──
        monthly_data = []
        for period in available_months:
            dfm = df[df["_bulan"] == period]
            total = len(dfm)
            anomali = int((dfm["anomali_final"] == 1).sum())
            normal = total - anomali
            pct = round(anomali / total * 100, 1) if total > 0 else 0
            skpd_affected = dfm[dfm["anomali_final"] == 1]["id_skpd"].nunique() if total > 0 else 0
            pegawai_affected = dfm[dfm["anomali_final"] == 1]["karyawan_id"].nunique() if total > 0 else 0

            monthly_data.append({
                "period": str(period),
                "label": period.strftime("%b %Y"),
                "month": period.month,
                "year": period.year,
                "total": total,
                "anomali": anomali,
                "normal": normal,
                "pct": pct,
                "skpd_affected": skpd_affected,
                "pegawai_affected": pegawai_affected,
            })

        # ── Per-SKPD per-month breakdown ──
        skpd_monthly = []
        all_skpd = sorted(df["id_skpd"].dropna().unique().astype(str))
        for s in all_skpd:
            row = {"id_skpd": s}
            for md in monthly_data:
                period = pd.Period(md["period"])
                dfm = df[(df["_bulan"] == period) & (df["id_skpd"].astype(str) == s)]
                row[md["period"]] = int((dfm["anomali_final"] == 1).sum()) if len(dfm) > 0 else 0
            skpd_monthly.append(row)

        # ── Per-employee per-month breakdown (top movers) ──
        emp_monthly = {}
        for md in monthly_data:
            period = pd.Period(md["period"])
            dfm = df[df["_bulan"] == period]
            emp_stats = dfm.groupby(["karyawan_id", "id_skpd"]).agg(
                anomali=("anomali_final", "sum")
            ).reset_index()
            for _, row in emp_stats.iterrows():
                kid = str(row["karyawan_id"])
                if kid not in emp_monthly:
                    emp_monthly[kid] = {"karyawan_id": kid, "id_skpd": str(row["id_skpd"])}
                emp_monthly[kid][md["period"]] = int(row["anomali"])

        emp_list = list(emp_monthly.values())

        # Hitung total anomali dan trend (last - first)
        period_keys = [md["period"] for md in monthly_data]
        for emp in emp_list:
            vals = [emp.get(pk, 0) for pk in period_keys]
            emp["total_anomali"] = sum(vals)
            emp["trend"] = vals[-1] - vals[0] if len(vals) >= 2 else 0

        # Top deteriorators (trend naik) & improvers (trend turun)
        emp_sorted_up = sorted(emp_list, key=lambda x: x["trend"], reverse=True)
        emp_sorted_down = sorted(emp_list, key=lambda x: x["trend"])
        top_deteriorators = [e for e in emp_sorted_up[:10] if e["trend"] > 0]
        top_improvers = [e for e in emp_sorted_down[:10] if e["trend"] < 0]

        # ── Trend antar bulan (perubahan pct) ──
        trends = []
        for i in range(1, len(monthly_data)):
            prev = monthly_data[i-1]
            curr = monthly_data[i]
            change_anomali = curr["anomali"] - prev["anomali"]
            change_pct = round((curr["pct"] - prev["pct"]), 1)
            trends.append({
                "from": prev["label"],
                "to": curr["label"],
                "change_anomali": change_anomali,
                "change_pct": change_pct,
                "direction": "up" if change_pct > 1 else ("down" if change_pct < -1 else "stable"),
            })

        return jsonify({
            "ok": True,
            "available_months": [{"period": str(m), "label": m.strftime("%b %Y"), "month": m.month, "year": m.year} for m in sorted(df["_bulan"].unique())],
            "selected_months": monthly_data,
            "skpd_monthly": skpd_monthly,
            "emp_comparison": emp_list,
            "top_deteriorators": top_deteriorators,
            "top_improvers": top_improvers,
            "trends": trends,
            "period_keys": period_keys,
        })

    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e), "detail": traceback.format_exc()})


@app.route("/api/perbandingan")
def api_perbandingan():
    """Compare anomaly data between two time periods."""
    if not models_exist():
        return jsonify({"ok": False, "error": "Belum ada data yang diproses."})

    try:
        pa_start = request.args.get("period_a_start")
        pa_end = request.args.get("period_a_end")
        pb_start = request.args.get("period_b_start")
        pb_end = request.args.get("period_b_end")

        if not all([pa_start, pa_end, pb_start, pb_end]):
            return jsonify({"ok": False, "error": "Parameter periode tidak lengkap. Butuh: period_a_start, period_a_end, period_b_start, period_b_end"})

        # Validate dates
        try:
            pa_start_dt = pd.to_datetime(pa_start)
            pa_end_dt = pd.to_datetime(pa_end)
            pb_start_dt = pd.to_datetime(pb_start)
            pb_end_dt = pd.to_datetime(pb_end)
        except Exception:
            return jsonify({"ok": False, "error": "Format tanggal tidak valid. Gunakan YYYY-MM-DD."})

        if pa_start_dt > pa_end_dt or pb_start_dt > pb_end_dt:
            return jsonify({"ok": False, "error": "Tanggal mulai harus sebelum tanggal akhir."})

        df = pd.read_pickle(os.path.join(MODEL_DIR, "df_hasil.pkl"))
        df["tanggal_kirim"] = pd.to_datetime(df["tanggal_kirim"])

        # Filter periods
        df_a = df[(df["tanggal_kirim"] >= pa_start_dt) & (df["tanggal_kirim"] <= pa_end_dt)]
        df_b = df[(df["tanggal_kirim"] >= pb_start_dt) & (df["tanggal_kirim"] <= pb_end_dt)]

        if df_a.empty and df_b.empty:
            return jsonify({"ok": False, "error": "Tidak ada data pada kedua periode yang dipilih."})

        # ── Summary metrics ──
        def period_summary(dfp):
            total = len(dfp)
            anomali = int((dfp["anomali_final"] == 1).sum()) if total > 0 else 0
            pct = round(anomali / total * 100, 1) if total > 0 else 0
            skpd_count = dfp[dfp["anomali_final"] == 1]["id_skpd"].nunique() if total > 0 else 0
            return {"total": total, "anomali": anomali, "pct": pct, "skpd_affected": skpd_count}

        summary_a = period_summary(df_a)
        summary_b = period_summary(df_b)

        # Trend indicators
        def calc_trend(val_a, val_b):
            if val_a == 0:
                return {"change_pct": 0, "direction": "stable"}
            change = round((val_b - val_a) / val_a * 100, 1)
            direction = "up" if change > 5 else ("down" if change < -5 else "stable")
            return {"change_pct": change, "direction": direction}

        trends = {
            "total": calc_trend(summary_a["total"], summary_b["total"]),
            "anomali": calc_trend(summary_a["anomali"], summary_b["anomali"]),
            "pct": calc_trend(summary_a["pct"], summary_b["pct"]),
            "skpd": calc_trend(summary_a["skpd_affected"], summary_b["skpd_affected"]),
        }

        # ── SKPD-level comparison ──
        def skpd_stats(dfp):
            if dfp.empty:
                return pd.DataFrame(columns=["id_skpd", "anomali"])
            return dfp.groupby("id_skpd").agg(anomali=("anomali_final", "sum")).reset_index()

        skpd_a = skpd_stats(df_a)
        skpd_b = skpd_stats(df_b)

        all_skpd = sorted(set(skpd_a["id_skpd"].tolist() + skpd_b["id_skpd"].tolist()))
        skpd_comparison = []
        for s in all_skpd:
            a_val = int(skpd_a[skpd_a["id_skpd"] == s]["anomali"].sum()) if s in skpd_a["id_skpd"].values else 0
            b_val = int(skpd_b[skpd_b["id_skpd"] == s]["anomali"].sum()) if s in skpd_b["id_skpd"].values else 0
            change = round((b_val - a_val) / a_val * 100, 1) if a_val > 0 else (100.0 if b_val > 0 else 0)
            skpd_comparison.append({
                "id_skpd": str(s),
                "anomali_a": a_val,
                "anomali_b": b_val,
                "change_pct": change,
                "direction": "up" if change > 20 else ("down" if change < -20 else "stable"),
            })

        # ── Employee-level comparison ──
        def emp_stats(dfp):
            if dfp.empty:
                return pd.DataFrame(columns=["karyawan_id", "id_skpd", "anomali"])
            return dfp.groupby(["karyawan_id", "id_skpd"]).agg(anomali=("anomali_final", "sum")).reset_index()

        emp_a = emp_stats(df_a)
        emp_b = emp_stats(df_b)

        all_emp = set()
        emp_skpd_map = {}
        for _, row in pd.concat([emp_a, emp_b]).iterrows():
            kid = str(row["karyawan_id"])
            all_emp.add(kid)
            emp_skpd_map[kid] = str(row["id_skpd"])

        emp_comparison = []
        for kid in all_emp:
            a_val = int(emp_a[emp_a["karyawan_id"].astype(str) == kid]["anomali"].sum()) if kid in emp_a["karyawan_id"].astype(str).values else 0
            b_val = int(emp_b[emp_b["karyawan_id"].astype(str) == kid]["anomali"].sum()) if kid in emp_b["karyawan_id"].astype(str).values else 0
            change = round((b_val - a_val) / a_val * 100, 1) if a_val > 0 else (100.0 if b_val > 0 else 0)
            emp_comparison.append({
                "karyawan_id": kid,
                "id_skpd": emp_skpd_map.get(kid, "-"),
                "anomali_a": a_val,
                "anomali_b": b_val,
                "change_pct": change,
                "diff": b_val - a_val,
            })

        # Top deteriorators and improvers
        emp_comparison.sort(key=lambda x: x["diff"], reverse=True)
        top_deteriorators = emp_comparison[:10]
        top_improvers = sorted(emp_comparison, key=lambda x: x["diff"])[:10]

        return jsonify({
            "ok": True,
            "period_a": {"start": pa_start, "end": pa_end},
            "period_b": {"start": pb_start, "end": pb_end},
            "summary_a": summary_a,
            "summary_b": summary_b,
            "trends": trends,
            "skpd_comparison": skpd_comparison,
            "emp_comparison": emp_comparison,
            "top_deteriorators": top_deteriorators,
            "top_improvers": top_improvers,
        })

    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e), "detail": traceback.format_exc()})


# ══════════════════════════════════════════════════════════════════
#  CUSTOM ERROR HANDLERS
# ══════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html",
                           nama_instansi=NAMA_INSTANSI,
                           has_model=models_exist()), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template("404.html",
                           nama_instansi=NAMA_INSTANSI,
                           has_model=models_exist(),
                           is_500=True), 500


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # use_reloader=False -> WAJIB: mencegah server restart di tengah
    #                      proses prediksi yang memutus koneksi browser
    # threaded=True      -> handle multiple request secara paralel
    app.run(debug=True, port=5000, use_reloader=False, threaded=True)
