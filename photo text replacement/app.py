"""
Photo Text Replacement Tool - Gradio Web Interface
"""

import socket
import gradio as gr
import cv2
import numpy as np
from text_replacer import TextReplacer
from PIL import Image
import traceback

_replacer = None


def _get_replacer():
    global _replacer
    if _replacer is None:
        _replacer = TextReplacer(languages=["ch", "en"])
    return _replacer


def detect_and_show(image, progress=gr.Progress()):
    if image is None:
        return "请上传一张图片", []

    try:
        arr = np.array(image)
        if arr.ndim == 2:
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.shape[2] == 4:
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        progress(0.2, desc="正在识别文字...")
        r = _get_replacer()
        regions = r.detect_text(img_bgr)

        progress(1.0, desc="完成")
        if not regions:
            return "未检测到文字", []

        lines = []
        text_list = []
        for region in regions:
            text = region.text.strip()
            if text:
                lines.append(f"[{region.confidence:.0%}] {text}")
                text_list.append(text)

        info = "检测到以下文字：\n" + "\n".join(lines)
        return info, text_list

    except Exception as e:
        traceback.print_exc()
        return f"识别出错：{e}", []


def process_image(image, replacements_text, detected_texts, progress=gr.Progress()):
    if image is None:
        return None, "请先上传图片"

    if not replacements_text or not replacements_text.strip():
        return image, "请输入替换规则，格式：原文字 -> 新文字（每行一条）"

    try:
        replacements = {}
        for line in replacements_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            for sep in ("->", "→"):
                if sep in line:
                    parts = line.split(sep, 1)
                    old = parts[0].strip()
                    new = parts[1].strip() if len(parts) > 1 else ""
                    if old:
                        replacements[old] = new
                    break

        if not replacements:
            return image, "未解析到有效的替换规则"

        arr = np.array(image)
        if arr.ndim == 2:
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.shape[2] == 4:
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        progress(0.0, desc="加载中...")
        r = _get_replacer()

        def progress_callback(stage, total, current, extra=None):
            if stage == "detect":
                progress(0.2, desc=f"发现 {total} 处文字")
            elif stage == "analyze":
                progress(0.3 + 0.15 * (current / max(1, total)),
                         desc=f"分析文字样式 ({current}/{total})")
            elif stage == "remove":
                progress(0.5 + 0.15 * (current / max(1, total)),
                         desc=f"去除原文字 ({current}/{total})")
            elif stage == "render":
                progress(0.7 + 0.25 * (current / max(1, total)),
                         desc=f"渲染新文字 ({current}/{total})")

        progress(0.05, desc="处理中...")
        result, all_texts, replaced = r.process(img_bgr, replacements, progress_callback)

        progress(1.0, desc="完成")
        info_lines = [
            f"检测文字：{len(all_texts)} 处  |  已替换：{len(replaced)} 处",
            "",
        ]
        if replaced:
            info_lines.append("替换详情：")
            for region, new_text in replaced:
                info_lines.append(f"  {region.text}  →  {new_text}")
        else:
            info_lines.append("未匹配到替换文字，请检查原文字是否完全一致")

        result_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        return result_img, "\n".join(info_lines)

    except Exception as e:
        traceback.print_exc()
        return image, f"处理出错：{e}"


def find_free_port(start=7860, end=7870):
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start


with gr.Blocks(title="图片文字替换工具", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 图片文字替换工具
    1. **上传图片** → 自动识别文字
    2. **输入替换规则** → 格式：`原文字 -> 新文字`（每行一条）
    3. **点击处理** → 下载结果
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="上传图片", type="pil", height=350)
            detected_text = gr.Textbox(
                label="检测到的文字",
                lines=8,
                interactive=False,
                placeholder="上传图片后自动检测...",
            )
        with gr.Column(scale=1):
            output_image = gr.Image(label="处理结果", type="pil", height=350)
            result_info = gr.Textbox(label="处理日志", lines=8, interactive=False)

    replacements_input = gr.Textbox(
        label="替换规则（每行一条：原文字 -> 新文字）",
        placeholder="旧标题 -> 新标题\nhello -> 你好\n2024 -> 2025",
        lines=4,
    )

    with gr.Row():
        process_btn = gr.Button("开始替换", variant="primary", size="lg")
        clear_btn = gr.Button("清空", variant="secondary")

    hidden_texts = gr.State([])

    input_image.upload(
        fn=detect_and_show,
        inputs=[input_image],
        outputs=[detected_text, hidden_texts],
    )
    input_image.clear(
        fn=lambda: ("", []),
        inputs=[],
        outputs=[detected_text, hidden_texts],
    )
    process_btn.click(
        fn=process_image,
        inputs=[input_image, replacements_input, hidden_texts],
        outputs=[output_image, result_info],
    )
    clear_btn.click(
        fn=lambda: (None, None, "", [], None),
        inputs=[],
        outputs=[input_image, output_image, detected_text, hidden_texts, result_info],
    )


if __name__ == "__main__":
    print("Loading OCR model...")
    _get_replacer()._ensure_ocr()
    print("OCR model ready.")

    port = find_free_port(7860, 7870)
    print(f"Starting on http://127.0.0.1:{port}")
    demo.queue(max_size=5).launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        inbrowser=True,
    )
