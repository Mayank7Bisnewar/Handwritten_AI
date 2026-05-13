import sys
from models.predict import predict_text

def test():
    img_path = "static/uploads/5db7db12d4254fc8ad43049b35b1bcb4.jpg"
    print(f"Testing TrOCR pipeline on: {img_path}")
    
    result = predict_text(img_path)
    
    print("\n--- Final Recognized Text ---")
    print(result["text"])

if __name__ == "__main__":
    test()
