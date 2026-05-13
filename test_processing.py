import cv2
import easyocr
import numpy as np

img_path = "static/uploads/5db7db12d4254fc8ad43049b35b1bcb4.jpg"
reader = easyocr.Reader(['en'], gpu=False)

def test_img(img, label):
    print(f"\n--- {label} ---")
    results = reader.readtext(img, detail=0, paragraph=True)
    print(" ".join(results))

# 1. Original Grayscale
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
test_img(gray, "Grayscale")

# 2. Thresholding + Dilation (make text thicker)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((2,2), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=1)
inverted = cv2.bitwise_not(dilated)
test_img(inverted, "Thicker Text (Otsu + Dilate)")

# 3. Increase Contrast (CLAHE)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl_img = clahe.apply(gray)
test_img(cl_img, "CLAHE Contrast")

# 4. Adaptive Thresholding with large block
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 15)
test_img(adaptive, "Adaptive Threshold")

