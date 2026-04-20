import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

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

    def transcribe(self, image: Image.Image) -> str:
        """
        Inference pipeline:
        1. Rescale and normalize image via TrOCRProcessor
        2. Autoregressive generation via VisionEncoderDecoderModel
        3. Decode token IDs to string
        """
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        
        # Greedy search for generation (standard for TrOCR)
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text.strip()

if __name__ == "__main__":
    print("HandwritingOCR (TrOCR) initialized. Ready for Phase 2.")
