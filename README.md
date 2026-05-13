# Handwritten Notes AI 🚀

> **AI-powered handwritten note recognition** — Upload any handwritten image and get clean, editable digital text in seconds.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=flat-square)
![EasyOCR](https://img.shields.io/badge/EasyOCR-1.7-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 Deep Learning OCR | CNN + LSTM model trained on IAM Handwriting Dataset |
| ⚡ Fast Inference | EasyOCR + OpenCV pipeline — results in ~3 seconds |
| 🌐 10+ Languages | English, French, German, Spanish, Hindi, Chinese & more |
| 🎯 Confidence Score | Real-time AI confidence meter per prediction |
| 📥 Export | Download results as **TXT** or **PDF** |
| 🔊 Voice Readback | Built-in text-to-speech using Web Speech API |
| 📋 History | SQLite-backed history of all extractions |
| 🎨 Handwritten UI | Dark glassmorphism design with neon glow & particles |

---

## 🗂 Project Structure

```
ML Project/
├── static/
│   ├── css/style.css        ← AntiGravity dark theme
│   ├── js/main.js           ← Upload, OCR, animations
│   ├── js/particles.js      ← Canvas particle background
│   └── uploads/             ← Uploaded images (auto-created)
├── templates/
│   ├── index.html           ← Landing + upload + output
│   ├── result.html          ← Per-note result view
│   └── history.html         ← All conversions history
├── models/
│   ├── train_model.py       ← CNN+LSTM trainer (IAM dataset)
│   └── predict.py           ← EasyOCR inference pipeline
├── dataset/                 ← Place IAM dataset here
├── app.py                   ← Flask app + API routes
├── database.py              ← SQLite models (SQLAlchemy)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone / Navigate to project

```bash
cd "ML Project"
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First run downloads EasyOCR model weights (~200 MB). Requires internet connection.

### 4. Run the server

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🧠 Training the Custom Model (Optional)

The app uses **EasyOCR** out-of-the-box with no training required.  
To train the custom CNN+LSTM model on the **IAM Handwriting Dataset**:

### Step 1 — Download IAM Dataset

1. Register at https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
2. Download `words.tgz` and `words.txt`
3. Extract into `dataset/iam_words/`:

```
dataset/iam_words/
├── words.txt
└── words/
    ├── a01/
    └── ...
```

### Step 2 — Train

```bash
python models/train_model.py --data dataset/iam_words --epochs 50
```

The trained model is saved as `models/handwritten_model.h5`.

---

## 🌐 API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Landing page |
| `GET` | `/history` | History page |
| `GET` | `/result/<id>` | Result page for note |
| `POST` | `/api/upload` | Upload image + run OCR |
| `GET` | `/api/download/txt/<id>` | Download TXT |
| `GET` | `/api/download/pdf/<id>` | Download PDF |
| `DELETE` | `/api/delete/<id>` | Delete note |

### POST /api/upload

**Form fields:**
- `file` — image file (PNG/JPG/BMP/TIFF/WebP, max 16 MB)
- `language` — language code (default: `en`)

**Response:**
```json
{
  "note_id": 1,
  "text": "The quick brown fox…",
  "confidence": 97.3,
  "blocks": [{"text": "The quick", "confidence": 98.1}],
  "redirect": "/result/1"
}
```

---

## ☁️ Deployment

### Render

1. Push to GitHub
2. New Web Service → connect repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`

### Railway

```bash
railway init
railway up
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `handwritten-secret-key-2024` | Flask secret key |
| `PORT` | `5000` | Server port |

---

## 🛠 Tech Stack

- **Backend:** Python 3.10+, Flask 3.0, Flask-SQLAlchemy
- **OCR:** EasyOCR 1.7 (PyTorch-based)
- **Vision:** OpenCV (grayscale, denoising, deskew, threshold)
- **ML:** TensorFlow 2.16 + PyTorch 2.3
- **DB:** SQLite via SQLAlchemy
- **PDF:** fpdf2
- **Frontend:** HTML5, CSS3, Bootstrap 5, Vanilla JS

---

## 📄 License

MIT License © 2024 Handwritten AI
