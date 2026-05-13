"""
Handwritten Notes AI — app.py
Main Flask application
"""

import os
import uuid
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, send_from_directory, abort)
from werkzeug.utils import secure_filename
from fpdf import FPDF
from database import db, Note
from models.predict import predict_text

# ─────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR  = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}

app = Flask(__name__)
app.config.update(
    SECRET_KEY         = os.environ.get("SECRET_KEY", "handwritten-secret-key-2024"),
    SQLALCHEMY_DATABASE_URI = "sqlite:///handwritten_notes.db",
    SQLALCHEMY_TRACK_MODIFICATIONS = False,
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024,   # 16 MB upload limit
    UPLOAD_FOLDER      = UPLOAD_DIR,
)

db.init_app(app)
os.makedirs(UPLOAD_DIR, exist_ok=True)

with app.app_context():
    db.create_all()


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def unique_filename(filename: str) -> str:
    ext  = filename.rsplit(".", 1)[1].lower() if "." in filename else "png"
    return f"{uuid.uuid4().hex}.{ext}"


# ─────────────────────────────────────────────────────────────────
# Routes — Pages
# ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result/<int:note_id>")
def result(note_id: int):
    note = Note.query.get_or_404(note_id)
    return render_template("result.html", note=note)


@app.route("/history")
def history():
    notes = Note.query.order_by(Note.created_at.desc()).all()
    return render_template("history.html", notes=notes)


# ─────────────────────────────────────────────────────────────────
# Routes — API
# ─────────────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    Accepts multipart/form-data with:
        file     — the image file
        language — optional (default 'en')
    Returns JSON with note_id and extracted text.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(f.filename):
        return jsonify({"error": "File type not allowed. Use PNG, JPG, BMP or TIFF."}), 415

    language = request.form.get("language", "en")

    # Save upload
    safe_name  = unique_filename(secure_filename(f.filename))
    save_path  = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    f.save(save_path)

    # Run OCR
    try:
        result_data = predict_text(save_path, language=language)
    except Exception as exc:
        return jsonify({"error": f"OCR failed: {str(exc)}"}), 500

    # Persist to DB
    note = Note(
        filename       = safe_name,
        original_path  = save_path,
        extracted_text = result_data["text"],
        confidence     = result_data["confidence"],
        language       = language,
    )
    db.session.add(note)
    db.session.commit()

    return jsonify({
        "note_id":    note.id,
        "text":       result_data["text"],
        "confidence": round(result_data["confidence"] * 100, 2),
        "blocks":     result_data["blocks"],
        "redirect":   url_for("result", note_id=note.id),
    })


@app.route("/api/download/txt/<int:note_id>")
def download_txt(note_id: int):
    note = Note.query.get_or_404(note_id)
    txt_path = os.path.join(UPLOAD_DIR, f"note_{note_id}.txt")
    with open(txt_path, "w", encoding="utf-8") as fp:
        fp.write(note.extracted_text)
    return send_from_directory(UPLOAD_DIR, f"note_{note_id}.txt",
                               as_attachment=True,
                               download_name=f"handwritten_note_{note_id}.txt")


@app.route("/api/download/pdf/<int:note_id>")
def download_pdf(note_id: int):
    note = Note.query.get_or_404(note_id)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.set_title(f"AntiGravity Note #{note_id}")
    pdf.multi_cell(0, 8, note.extracted_text)
    pdf_path = os.path.join(UPLOAD_DIR, f"note_{note_id}.pdf")
    pdf.output(pdf_path)
    return send_from_directory(UPLOAD_DIR, f"note_{note_id}.pdf",
                               as_attachment=True,
                               download_name=f"handwritten_note_{note_id}.pdf")


@app.route("/api/delete/<int:note_id>", methods=["DELETE"])
def delete_note(note_id: int):
    note = Note.query.get_or_404(note_id)
    # Remove image file
    try:
        if os.path.exists(note.original_path):
            os.remove(note.original_path)
    except OSError:
        pass
    db.session.delete(note)
    db.session.commit()
    return jsonify({"message": "Note deleted successfully."})


@app.route("/api/history")
def api_history():
    notes = Note.query.order_by(Note.created_at.desc()).limit(50).all()
    return jsonify([n.to_dict() for n in notes])


# ─────────────────────────────────────────────────────────────────
# Serve uploaded images
# ─────────────────────────────────────────────────────────────────

@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


# ─────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
