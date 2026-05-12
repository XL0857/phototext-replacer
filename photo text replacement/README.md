# 图片文字替换工具

本地图片文字批量替换工具。上传图片 → 自动 OCR 识别文字 → 输入替换规则 → 无痕替换。

## 功能

- 自动识别图片中所有文字（中文 + 英文）
- 像素级精准去除原文字，保留背景纹理
- 匹配原文字字体、颜色、大小、对齐方式
- 支持批量替换（每行一条规则）

## 安装

```bash
# 1. 创建虚拟环境（Python 3.8+）
python -m venv venv
venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt
```

## 使用

```bash
python app.py
```

浏览器打开 `http://127.0.0.1:7860`

### 替换规则格式

```
原文字 -> 新文字
```

每行一条，支持中文和英文。示例：

```
旧标题 -> 新标题
Hello -> 你好
2024 -> 2025
```

## 技术栈

- **OCR**: PaddleOCR（PP-OCRv4 模型，百度 CDN）
- **文字去除**: OpenCV Inpainting（像素级遮罩）
- **文字渲染**: PIL/Pillow（字体匹配）
- **界面**: Gradio

## 依赖

- PaddlePaddle 2.6.2 + PaddleOCR 2.7.3
- OpenCV 4.6+
- Pillow 10+
- Gradio 6+
