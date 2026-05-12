# -*- coding: utf-8 -*-
"""
PaddleOCR Model Pre-loader
Pre-downloads PaddleOCR models to avoid first-run delay.
Models are hosted on Baidu CDN.
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np


def main():
    print("=" * 50)
    print("PaddleOCR Model Pre-loader")
    print("=" * 50)
    print()
    print("Downloading OCR models from Baidu CDN (~50MB)...")
    print()

    try:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(
            lang='ch',
            use_angle_cls=True,
            use_gpu=False,
            show_log=False,
            use_space_char=True,
        )

        print("Warming up...")
        dummy = np.ones((320, 320, 3), dtype=np.uint8) * 255
        try:
            ocr.ocr(dummy)
        except Exception:
            pass

        print()
        print("=" * 50)
        print("Models ready. Run: python app.py")
        print("=" * 50)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease check your network connection.")
        sys.exit(1)


if __name__ == "__main__":
    main()
