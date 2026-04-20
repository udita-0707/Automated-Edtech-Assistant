import io
from PIL import Image, ImageFilter, ImageOps
import pytesseract

class TesseractOCR:
    """
    Standard OCR engine for Phase 1.
    Uses classical image processing followed by the Tesseract legacy (non-neural) engine.
    """
    
    def __init__(self, tesseract_cmd=None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def preprocess(self, image: Image.Image) -> Image.Image:
        """
        Enhancement pipeline for handwriting:
        1. Grayscale
        2. Auto-contrast
        3. Sharpen
        4. Binarize
        """
        gray = image.convert("L")
        ac = ImageOps.autocontrast(gray, cutoff=1)
        sharp = ac.filter(ImageFilter.SHARPEN)
        # Threshold at half-intensity
        binary = sharp.point(lambda p: 255 if p > 128 else 0, "1")
        return binary

    def transcribe(self, image: Image.Image) -> str:
        """
        Performs OCR using legacy engine (--oem 0) which is often more stable
        on highly cleaned/binarized text than the LSTM engine.
        """
        processed = self.preprocess(image)
        # --oem 0: Legacy engine only.
        # --psm 6: Assume a single uniform block of text.
        custom_config = "--oem 0 --psm 6"
        text = pytesseract.image_to_string(processed, config=custom_config)
        return " ".join(text.split())

if __name__ == "__main__":
    # Smoke test logic
    print("TesseractOCR engine initialized.")
