import torch
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from phase3.ocr.line_segmenter import LineSegmenter

class HandwritingOCR:
    """
    Phase 2 Neural OCR engine.
    Utilizes microsoft/trocr-base-handwritten — a transformer-based
    encoder-decoder model (ViT + RoBERTa).
    
    Theoretical Context:
    The Vision Transformer (ViT) encoder segments the image into patches, 
    while the RoBERTa decoder generates the text token-by-token using 
    cross-attention over the visual embeddings.
    """
    
    def __init__(self, model_name="microsoft/trocr-base-handwritten"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading TrOCR on {self.device}...")
        
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.segmenter = LineSegmenter()

    def transcribe_single_line(self, pil_image: Image.Image) -> str:
        """
        Transcribes a single PIL Image crop (one line).
        This is the original TrOCR logic, now extracted
        as a helper method.
        """
        # Convert to RGB if necessary
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
            
        pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_new_tokens=128
            )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()

    def transcribe(self, image: Image.Image) -> str:
        """
        Full paragraph transcription.

        Pipeline:
        1. Segment image into individual text lines
        2. Run TrOCR on each line independently
        3. Filter empty results
        4. Join with spaces

        Theory:
        TrOCR encoder (ViT) was trained on single-line
        IAM crops of shape ~(384, variable_width).
        Feeding a full paragraph (e.g. 1200x800) causes
        the ViT patch tokenizer to create too many tokens,
        overwhelming the RoBERTa decoder's attention span.
        Line segmentation restores the expected input
        distribution the model was trained on.
        """
        import os
        import tempfile
        
        # Save PIL Image to temporary file for segmenter
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        try:
            line_images = self.segmenter.segment(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        results = []
        for i, line_img in enumerate(line_images):
            try:
                text = self.transcribe_single_line(line_img)
                if text and len(text.strip()) > 0:
                    results.append(text.strip())
            except Exception as e:
                print(f"Line {i+1} transcription failed: {e}")
                continue

        if not results:
            return ""

        return " ".join(results)

if __name__ == "__main__":
    print("HandwritingOCR (TrOCR) initialized. Ready for Phase 2.")
