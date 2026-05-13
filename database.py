"""
Handwritten Notes AI — database.py
SQLite database models using Flask-SQLAlchemy
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class Note(db.Model):
    """Stores each OCR conversion result."""
    __tablename__ = "notes"

    id            = db.Column(db.Integer, primary_key=True)
    filename      = db.Column(db.String(256), nullable=False)
    original_path = db.Column(db.String(512), nullable=False)
    extracted_text = db.Column(db.Text, nullable=False, default="")
    confidence    = db.Column(db.Float, nullable=False, default=0.0)
    language      = db.Column(db.String(64), nullable=False, default="en")
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id":             self.id,
            "filename":       self.filename,
            "original_path":  self.original_path,
            "extracted_text": self.extracted_text,
            "confidence":     round(self.confidence * 100, 2),
            "language":       self.language,
            "created_at":     self.created_at.strftime("%d %b %Y, %H:%M"),
        }
