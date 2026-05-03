import os
import sys
from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
_PH3 = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PH3)

# Add root to path so we can import from phase3.ocr
sys.path.insert(0, _ROOT)

from phase3.ocr.trocr_engine import HandwritingOCR

def create_dummy_image(text_lines, filename):
    # Create a white image
    img = Image.new('RGB', (600, 100 * len(text_lines) + 50), color = 'white')
    d = ImageDraw.Draw(img)
    
    # Draw simple text to simulate lines (though TrOCR expects handwriting, this tests the pipeline)
    for i, line in enumerate(text_lines):
        d.text((50, 50 + i * 100), line, fill='black')
        
    path = os.path.join(_HERE, filename)
    img.save(path)
    return path

def run_tests():
    print("Loading OCR Engine...")
    engine = HandwritingOCR()
    
    print("\n" + "="*50)
    print("Test 1 — Single line image")
    img1 = create_dummy_image(["Hello world this is a test"], "test1.png")
    pil_img1 = Image.open(img1)
    res1 = engine.transcribe(pil_img1)
    print(f"Final joined: {res1}")
    
    print("\n" + "="*50)
    print("Test 2 — Multi-line paragraph")
    img2 = create_dummy_image([
        "First line of the paragraph.",
        "Second line of the paragraph.",
        "Third line of the paragraph."
    ], "test2.png")
    pil_img2 = Image.open(img2)
    res2 = engine.transcribe(pil_img2)
    print(f"Final joined: {res2}")
    
    print("\n" + "="*50)
    print("Test 4 — Edge case (blank image)")
    img4 = create_dummy_image([], "test4.png")
    pil_img4 = Image.open(img4)
    res4 = engine.transcribe(pil_img4)
    print(f"Final joined: '{res4}'")

if __name__ == "__main__":
    run_tests()
