import cv2
import numpy as np
from PIL import Image

class LineSegmenter:
    """
    Segments a paragraph image into individual
    text line crops using horizontal projection
    profile analysis.

    Theory:
    A scanned text page has alternating bands of:
      - High pixel density rows (text lines)
      - Low pixel density rows (whitespace between lines)

    Horizontal projection profile = sum of dark pixels
    per row. Plotting this gives peaks at text lines
    and valleys at whitespace.

    We find the valleys (row indices where projection
    falls below threshold) to determine line boundaries.

    This is a classical document analysis technique
    that requires no neural network and works reliably
    on clean scanned/photographed text.
    """

    def preprocess(self, image_path: str):
        """
        Convert to grayscale and binarize for
        clean projection profile computation.
        """
        img = cv2.imread(image_path)
        if img is None:
            pil_img = Image.open(image_path).convert("RGB")
            img = np.array(pil_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Otsu binarization — automatic threshold
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return img, binary

    def get_line_boundaries(self, binary, min_line_height=10, padding=4):
        """
        Computes horizontal projection profile and
        finds text line boundaries.

        Returns list of (y_start, y_end) tuples,
        one per detected text line.
        """
        h_proj = np.sum(binary, axis=1)
        threshold = np.max(h_proj) * 0.05

        in_line = h_proj > threshold
        boundaries = []
        start = None

        for i, val in enumerate(in_line):
            if val and start is None:
                start = i
            elif not val and start is not None:
                if i - start >= min_line_height:
                    y1 = max(0, start - padding)
                    y2 = min(binary.shape[0], i + padding)
                    boundaries.append((y1, y2))
                start = None

        if start is not None:
            y1 = max(0, start - padding)
            y2 = binary.shape[0]
            if y2 - y1 >= min_line_height:
                boundaries.append((y1, y2))

        return boundaries

    def segment(self, image_path: str):
        """
        Returns list of PIL Image crops, one per line.
        Falls back to full image if segmentation fails
        or produces fewer than 2 lines.
        """
        try:
            img, binary = self.preprocess(image_path)
            boundaries = self.get_line_boundaries(binary)

            if len(boundaries) < 2:
                # Single line or failed segmentation
                # Return full image as-is
                return [Image.open(image_path).convert("RGB")]

            h, w = img.shape[:2]
            crops = []
            for (y1, y2) in boundaries:
                crop_bgr = img[y1:y2, 0:w]
                crop_rgb = cv2.cvtColor(
                    crop_bgr, cv2.COLOR_BGR2RGB
                )
                crops.append(Image.fromarray(crop_rgb))

            return crops

        except Exception as e:
            print(f"Line segmentation failed: {e}")
            return [Image.open(image_path).convert("RGB")]
