"""
Image Text Replacement Engine
- OCR text detection with PaddleOCR
- Precise text removal using pixel-level masks
- High-quality text re-rendering with matched fonts
"""

import os
import warnings
warnings.filterwarnings('ignore')

os.environ['GLOG_minloglevel'] = '3'
os.environ['FLAGS_print_model_net_proto'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class TextStyle:
    font_size: int
    color: tuple  # (R, G, B)
    bold: bool
    italic: bool
    alignment: str
    letter_spacing: float
    pixel_mask: Optional[np.ndarray] = None  # binary mask of text pixels (full image size)
    mask_bbox: tuple = (0, 0, 0, 0)  # (x1, y1, x2, y2) of mask region


@dataclass
class TextRegion:
    bbox: list
    text: str
    confidence: float
    style: Optional[TextStyle] = None


class TextReplacer:
    _ocr = None
    _ready = False
    _load_error = None
    languages = ["ch", "en"]

    def __init__(self, languages=None):
        if languages is None:
            languages = ["ch", "en"]
        self.languages = languages
        self._ocr = None
        self._ready = False
        self._load_error = None

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _ensure_ocr(self, progress_cb: Callable | None = None):
        if self._ocr is not None:
            return

        def log(msg):
            if progress_cb:
                progress_cb(msg)

        try:
            import warnings as _w
            _w.filterwarnings('ignore')
            import paddle
            paddle.set_device('cpu')
            from paddleocr import PaddleOCR

            log("Loading PaddleOCR models...")

            lang = "ch" if "ch" in str(self.languages) else "en"

            self._ocr = PaddleOCR(
                lang=lang,
                use_angle_cls=True,
                use_gpu=False,
                show_log=False,
                use_space_char=True,
            )
            self._ready = True
            log("PaddleOCR models ready")
        except Exception as e:
            self._load_error = f"Model load failed: {e}"
            raise RuntimeError(self._load_error) from e

    def detect_text(
        self, image: np.ndarray,
        progress_cb: Callable | None = None,
        min_confidence: float = 0.6,
    ) -> list[TextRegion]:
        self._ensure_ocr(progress_cb)
        if progress_cb:
            progress_cb("Detecting text...")

        image = self._normalize_image(image)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._ocr.ocr(img_rgb)
        regions = []

        if results and results[0]:
            for item in results[0]:
                bbox = item[0]
                text_info = item[1]
                text = text_info[0]
                conf = text_info[1]
                if text and text.strip() and conf >= min_confidence:
                    regions.append(TextRegion(bbox=bbox, text=text.strip(), confidence=conf))
        return regions

    def analyze_style(self, image: np.ndarray, region: TextRegion) -> TextStyle:
        """Analyze style AND extract pixel-precise text mask."""
        image = self._normalize_image(image)
        bbox = region.bbox
        pts = np.array(bbox, dtype=np.int32)

        x, y, w, h = cv2.boundingRect(pts)
        x = max(0, x)
        y = max(0, y)
        w = max(1, min(w, image.shape[1] - x))
        h = max(1, min(h, image.shape[0] - y))

        pad = 4
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return TextStyle(font_size=12, color=(0, 0, 0), bold=False,
                             italic=False, alignment="center", letter_spacing=0.0)

        # --- Extract text color using K-Means (for color only) ---
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        flat_roi = roi.reshape(-1, 3).astype(np.float32)

        # Simple 2-cluster K-Means on color to find text vs background
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv2.kmeans(flat_roi, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()

        # Text cluster = minority (fewer pixels)
        cluster_counts = np.bincount(labels, minlength=2)
        text_cluster = np.argmin(cluster_counts)

        # Get text color (BGR) from minority cluster center
        text_color_bgr = flat_roi[labels == text_cluster].mean(axis=0)
        color = tuple(int(c) for c in text_color_bgr[::-1])  # BGR -> RGB

        # --- Build mask using color distance ---
        color_dist = np.sqrt(np.sum((roi.astype(np.float32) - text_color_bgr.reshape(1, 1, 3)) ** 2, axis=2))
        # Adaptive threshold: inclusive to catch anti-aliased edge pixels too
        bg_color_bgr = flat_roi[labels != text_cluster].mean(axis=0)
        max_dist = np.sqrt(np.sum((text_color_bgr - bg_color_bgr) ** 2))
        threshold = max(20, max_dist * 0.55)  # More inclusive than before

        text_mask_roi = (color_dist < threshold).astype(np.uint8) * 255
        # Clean up noise
        text_mask_roi = cv2.morphologyEx(text_mask_roi, cv2.MORPH_OPEN,
                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        # Close small gaps within text strokes
        text_mask_roi = cv2.morphologyEx(text_mask_roi, cv2.MORPH_CLOSE,
                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        # Fallback if mask is empty
        if np.sum(text_mask_roi) < 10:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text_mask_roi = (binary == 0).astype(np.uint8) * 255
            if np.sum(text_mask_roi) > np.sum(~(text_mask_roi > 0)):
                text_mask_roi = (binary == 255).astype(np.uint8) * 255

        # Measure actual text height from mask
        text_rows = np.any(text_mask_roi > 0, axis=1)
        if np.any(text_rows):
            text_row_indices = np.where(text_rows)[0]
            actual_text_height = text_row_indices[-1] - text_row_indices[0] + 1
        else:
            actual_text_height = h

        font_size = max(8, int(actual_text_height * 0.85))

        # --- Bold detection ---
        edges = cv2.Canny(gray, 50, 150)
        stroke_width = 0
        if np.sum(edges) > 0:
            dist = cv2.distanceTransform(edges.astype(np.uint8), cv2.DIST_L2, 3)
            stroke_width = np.median(dist[dist > 0]) * 2
        bold = stroke_width > 3.0

        # --- Italic detection ---
        italic = False
        if len(region.text) > 1:
            top_edge = bbox[1][0] - bbox[0][0]
            bottom_edge = bbox[2][0] - bbox[3][0]
            if abs(top_edge) > w * 0.08 or abs(bottom_edge) > w * 0.08:
                italic = True

        # --- Alignment ---
        img_w = image.shape[1]
        center_x = x + w / 2
        if center_x < img_w * 0.35:
            alignment = "left"
        elif center_x > img_w * 0.65:
            alignment = "right"
        else:
            alignment = "center"

        # --- Letter spacing ---
        letter_spacing = 0.0
        if len(region.text) > 1:
            letter_spacing = max(0, (w / len(region.text) - font_size * 0.55) / font_size)

        # --- Build full-image pixel mask (with 1px expansion for edge coverage) ---
        text_mask_roi = cv2.dilate(text_mask_roi,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                    iterations=1)
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = text_mask_roi

        return TextStyle(
            font_size=font_size,
            color=color,
            bold=bold,
            italic=italic,
            alignment=alignment,
            letter_spacing=max(0, letter_spacing),
            pixel_mask=full_mask,
            mask_bbox=(x1, y1, x2, y2),
        )

    def remove_text(self, image: np.ndarray, region: TextRegion) -> tuple[np.ndarray, np.ndarray]:
        """Remove text using OCR polygon mask — guaranteed to cover all text pixels."""
        bbox = region.bbox
        pts = np.array(bbox, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)

        # Expand the polygon to ensure full text coverage including anti-aliased edges
        pad = max(2, min(w, h) // 10)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        shifted_pts = pts.copy()
        shifted_pts[:, 0] -= x1
        shifted_pts[:, 1] -= y1
        sub_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.fillPoly(sub_mask, [shifted_pts], 255)

        # Also merge pixel mask if available (for precise edge coverage)
        if region.style is not None and region.style.pixel_mask is not None:
            pixel_part = region.style.pixel_mask[y1:y2, x1:x2]
            sub_mask = np.maximum(sub_mask, pixel_part)

        mask[y1:y2, x1:x2] = sub_mask

        # Inpaint with radius proportional to text size for full coverage
        text_height = abs(region.bbox[0][1] - region.bbox[2][1])
        inpaint_radius = max(3, int(text_height // 3))
        result = cv2.inpaint(image, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
        return result, mask

    def render_text(self, image: np.ndarray, text: str,
                    region: TextRegion, style: TextStyle) -> np.ndarray:
        """Render sharp, properly-sized replacement text."""
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        bbox = region.bbox
        x, y, w, h = cv2.boundingRect(np.array(bbox, dtype=np.int32))

        # --- Find the best matching font ---
        font, actual_size = self._match_font(text, w, h, style)

        overlay = Image.new("RGBA", image_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        text_bbox = draw.textbbox((0, 0), text, font=font,
                                   spacing=int(style.letter_spacing * actual_size))
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # Scale down if text is too wide, but never scale UP
        if text_w > w - 4 and w > 0:
            scale = (w - 4) / text_w
            scaled_size = max(8, int(actual_size * scale))
            font = self._resize_font(font, scaled_size)
            actual_size = scaled_size
            draw = ImageDraw.Draw(overlay)
            text_bbox = draw.textbbox((0, 0), text, font=font,
                                       spacing=int(style.letter_spacing * actual_size))
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

        # Position: center in original bounding box
        if style.alignment == "center":
            text_x = x + (w - text_w) / 2
        elif style.alignment == "right":
            text_x = x + w - text_w - 2
        else:
            text_x = x + 2

        # Center vertically, compensating for font baseline offset
        text_y = y + (h - text_h) / 2 - text_bbox[1]

        draw.text((text_x, text_y), text, font=font,
                  fill=style.color + (255,),
                  spacing=int(style.letter_spacing * actual_size))

        # Composite without blur — PIL already anti-aliases
        image_pil = Image.alpha_composite(image_pil.convert("RGBA"), overlay)
        result = cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        return result

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Image is None")
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.shape[2] == 3:
            return image.copy()
        raise ValueError(f"Unexpected image shape: {image.shape}")

    def process(self, image: np.ndarray, replacements: dict[str, str],
                progress_callback=None) -> np.ndarray:
        result = self._normalize_image(image)

        # Step 1: Detect
        regions = self.detect_text(
            result,
            progress_cb=lambda msg: progress_callback("model_status", 0, 0, msg)
                if progress_callback else None,
        )
        if progress_callback:
            progress_callback("detect", len(regions), 0)

        # Step 2: Analyze style + extract pixel masks
        for i, region in enumerate(regions):
            region.style = self.analyze_style(result, region)
            if progress_callback:
                progress_callback("analyze", len(regions), i + 1)

        # Step 3: Match replacements
        to_replace = []
        for region in regions:
            if region.text in replacements:
                to_replace.append((region, replacements[region.text]))

        if not to_replace:
            return result, [r.text for r in regions], []

        # Step 4: Remove text (pixel-precise)
        for i, (region, _) in enumerate(to_replace):
            result, _ = self.remove_text(result, region)
            if progress_callback:
                progress_callback("remove", len(to_replace), i + 1)

        # Step 5: Render new text
        for i, (region, new_text) in enumerate(to_replace):
            result = self.render_text(result, new_text, region, region.style)
            if progress_callback:
                progress_callback("render", len(to_replace), i + 1)

        return result, [r.text for r in regions], to_replace

    def _match_font(self, text: str, target_w: int, target_h: int,
                    style: TextStyle) -> tuple[ImageFont.ImageFont, int]:
        """Find the font whose rendering best matches the original text metrics."""
        font_candidates = self._list_fonts(style.bold)

        best_font = None
        best_size = style.font_size
        best_score = float('inf')

        for size in [max(8, style.font_size - 2), style.font_size, style.font_size + 2]:
            for font_path in font_candidates:
                try:
                    font = ImageFont.truetype(font_path, size=size)
                    # Verify font has glyphs for this text
                    if not self._font_has_glyphs(font, text):
                        continue
                    tmp_img = Image.new("RGB", (1, 1))
                    tmp_draw = ImageDraw.Draw(tmp_img)
                    tbox = tmp_draw.textbbox((0, 0), text, font=font)
                    tw = tbox[2] - tbox[0]
                    th = tbox[3] - tbox[1]
                    if tw <= 0 or th <= 0:
                        continue
                    score = abs(tw - target_w) * 1.5 + abs(th - target_h)
                    if score < best_score:
                        best_score = score
                        best_font = font
                        best_size = size
                except Exception:
                    continue

        if best_font is not None:
            return best_font, best_size
        return ImageFont.load_default(), style.font_size

    def _font_has_glyphs(self, font: ImageFont.ImageFont, text: str) -> bool:
        """Check if font can render all characters in text."""
        try:
            mask = font.getmask(text)
            if mask is None:
                return False
            # Check that non-whitespace chars have non-zero width
            bbox = mask.getbbox()
            return bbox is not None and bbox[2] > 0 and bbox[3] > 0
        except Exception:
            return False

    def _list_fonts(self, bold: bool) -> list[str]:
        """List available font paths, bold-preferring if requested."""
        paths = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/msyhbd.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/times.ttf",
            "C:/Windows/Fonts/timesbd.ttf",
            "C:/Windows/Fonts/timesi.ttf",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
        ]
        # Filter to existing files only
        existing = [p for p in paths if os.path.exists(p)]
        if bold:
            bold_first = [p for p in existing if 'bd' in p.lower() or 'bold' in p.lower() or 'hei' in p.lower()]
            other = [p for p in existing if p not in bold_first]
            return bold_first + other
        return existing

    def _resize_font(self, font: ImageFont.ImageFont, size: int) -> ImageFont.ImageFont:
        try:
            if hasattr(font, 'path') and font.path:
                return ImageFont.truetype(font.path, size=size)
        except Exception:
            pass
        return ImageFont.load_default()
