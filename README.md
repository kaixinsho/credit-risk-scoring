# Credit Risk Scoring

Simulasi penilaian risiko kredit: memperkirakan peluang seorang pemohon pinjaman gagal bayar. Proyek ini mencakup model, API, dan website yang bisa dicoba langsung.

- Demo: https://credit-risk-scoring-i8dq.vercel.app
- Dokumentasi API: https://credit-risk-scoring-i8dq.vercel.app/docs

Server memakai paket gratis, jadi permintaan pertama setelah lama tidak dipakai bisa memakan waktu beberapa detik.

## Masalah yang dijawab

Sebelum menyetujui pinjaman, pemberi kredit perlu tahu seberapa besar peluang pemohon tidak melunasinya. Proyek ini melatih model dari data permohonan pinjaman historis, lalu menyajikannya sebagai alat bantu analis kredit: isi data pemohon, lihat perkiraan peluang gagal bayar beserta pembandingnya dengan rata-rata pemohon.

## Hasil versi 1

| | |
|---|---|
| Data | 307.511 permohonan, 122 kolom, sekitar 8% gagal bayar |
| Model | LightGBM (maksimal 500 pohon, learning rate 0,05, early stopping) |
| Validasi | 20% data, stratified, `random_state=42` |
| AUC validasi | 0,7602 |

Ini baseline dari satu tabel (`application_train.csv`) tanpa tuning. AUC dipakai sebagai metrik utama karena kelasnya timpang: model yang menebak "semua lancar" sudah 92% akurat tetapi tidak berguna.

## Temuan data

Kolom `DAYS_EMPLOYED` memuat nilai 365243 (sekitar 18% baris). Itu penanda "tidak bekerja atau pensiun", bukan lama bekerja. Nilainya diubah menjadi kosong dan dicatat sebagai fitur terpisah (`DAYS_EMPLOYED_ANOM`), karena kelompok ini gagal bayar sekitar 5,4% dibanding 8,7% pada kelompok lain.

## Cara kerja

```
Browser  ->  FastAPI di Vercel  ->  Model LightGBM
```

Aplikasi FastAPI yang sama melayani halaman web (`/`), endpoint prediksi (`/predict`), dan dokumentasi (`/docs`). Karena satu alamat, tidak ada urusan CORS.

## Struktur repo

```
app/main.py            API dan penyaji halaman web
web/index.html         halaman web (HTML, CSS, JavaScript dalam satu file)
src/train.py           melatih model dan menyimpannya ke models/
models/lgbm_v1.joblib  model hasil latihan
notebooks/             eksplorasi awal
lib/libgomp.so.1       pustaka OpenMP yang dibutuhkan LightGBM (lihat catatan)
requirements.txt       library minimal untuk server
requirements-dev.txt   semua library untuk pengembangan
vercel.json            pengaturan file yang ikut ke server
```

Catatan `lib/libgomp.so.1`: server Vercel tidak menyediakan pustaka ini, sedangkan LightGBM membutuhkannya. File ini salinan libgomp (GNU OpenMP, GCC Runtime Library Exception) dari paket scikit-learn, dengan nama internal disesuaikan menjadi `libgomp.so.1`. `app/main.py` memuatnya lebih dulu di server Linux.

## Endpoint

| Metode | Alamat | Fungsi |
|---|---|---|
| GET | `/` | halaman web |
| GET | `/health` | status server dan AUC model |
| POST | `/predict` | perkiraan peluang gagal bayar |

Contoh permintaan:

```bash
curl -X POST https://credit-risk-scoring-i8dq.vercel.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "income_total": 202500,
    "credit_amount": 406597,
    "annuity": 24700,
    "goods_price": 351000,
    "age_years": 26,
    "years_employed": 1.7,
    "income_type": "Working",
    "family_status": "Single / not married",
    "ext_source_1": 0.08,
    "ext_source_2": 0.26,
    "ext_source_3": 0.14
  }'
```

Contoh jawaban:

```json
{"default_probability": 0.5258, "model_version": "v1"}
```

Field `income_total`, `credit_amount`, `annuity`, dan `age_years` wajib diisi. Field lain punya nilai bawaan atau boleh dikosongkan. Kosongkan `years_employed` untuk pemohon yang tidak bekerja atau pensiun.

## Menjalankan di komputer sendiri

1. Unduh `application_train.csv` dari halaman kompetisi di bawah, lalu taruh di folder `data/`.
2. Siapkan lingkungan Python (diuji dengan Python 3.12):

```bash
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
```

3. Latih model, lalu jalankan aplikasi:

```bash
python src/train.py
uvicorn app.main:app --reload
```

4. Buka http://127.0.0.1:8000.

## Data

Home Credit Default Risk (Kaggle): https://www.kaggle.com/competitions/home-credit-default-risk

Data mentah tidak disertakan di repo sesuai aturan kompetisi.

## Keterbatasan

- Data berasal dari pemberi pinjaman internasional, bukan dari Indonesia.
- Nominal mengikuti satuan data asli, bukan Rupiah.
- Peluang belum dikalibrasi dan belum ada batas keputusan yang ditentukan dari biaya bisnis.
- Pemeriksaan keadilan antarkelompok belum dilakukan. Model saat ini memakai atribut demografis (termasuk jenis kelamin dengan nilai bawaan pada API).
- Tabel riwayat kredit lain belum dipakai.

## Rencana berikutnya

- Menggabungkan tabel tambahan (`bureau`, `previous_application`, `installments_payments`, `credit_card_balance`) lewat feature engineering.
- Kalibrasi probabilitas.
- Menentukan batas keputusan dari biaya salah menyetujui dibanding salah menolak.
- Penjelasan per keputusan dengan SHAP.
- Pemeriksaan keadilan antarkelompok.
- Test otomatis dan Docker.
