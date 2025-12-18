# MLOps YouTube Comments Project

Proyek ini dirancang untuk membangun pipeline MLOps end-to-end
untuk analisis komentar YouTube (misalnya sentiment analysis / toxicity / topic classification).

## Struktur Direktori

```text
mlops-youtube-comments/
├─ src/
│  ├─ scraping/
│  │  └─ youtube_comment_scraper.py
│  ├─ api/
│  │  └─ app.py
│  └─ models/
│     └─ __init__.py
├─ data/
│  └─ raw/
├─ notebooks/
│  └─ 01_eda_and_baseline.ipynb
├─ .env.example
├─ .gitignore
├─ requirements.txt
├─ Dockerfile
└─ docker-compose.yml
```

## Langkah Awal

1. Buat virtual environment dan install dependensi:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. Salin `.env.example` menjadi `.env` dan isi nilai `YOUTUBE_API_KEY`:

   ```bash
   cp .env.example .env
   ```

3. Jalankan API secara lokal:

   ```bash
   uvicorn src.api.app:app --reload
   ```

4. (Opsional) Jalankan dengan Docker Compose:

   ```bash
   docker-compose up --build
   ```

## API Inferensi

- Endpoint utama: `POST /analyze` _(tanpa payload, cukup panggil endpoint-nya)._  
   Contoh: `curl -X POST http://localhost:8000/analyze`

Endpoint ini otomatis:

1. Mencari file komentar terbaru dengan pola `data/raw/comments_*.csv`.
2. Melakukan inferensi sentimen (positive/neutral/negative) terhadap seluruh komentar pada file tersebut memakai artefak `models/baseline_sentiment.joblib`.
3. Mengembalikan:
   - `source_file`, `video_id`, dan `total_comments`.
   - `predictions` per komentar (label numerik, label teks, confidence, teks asli).
   - Ringkasan distribusi sentimen (`breakdown`) plus nilai akurasi model yang dibaca dari `models/model_metadata.json`.

Pastikan file metadata tersebut diperbarui setiap kali model dilatih ulang supaya angka akurasinya tetap relevan.

## Streamlit Dashboard

Dashboard sederhana tersedia di `streamlit_app.py` untuk memvisualisasikan hasil inferensi:

```bash
streamlit run streamlit_app.py
```

- Pastikan FastAPI (`uvicorn src.api.app:app --reload`) sudah aktif dan terdapat file `data/raw/comments_*.csv` hasil scraping.
- Dashboard akan memanggil `/analyze`, menampilkan ringkasan distribusi sentimen, metrik akurasi model, serta tabel komentar dengan label & confidence.
- Base URL API dapat disesuaikan langsung di antarmuka Streamlit jika berjalan di host berbeda.

```

```
