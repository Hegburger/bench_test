# 将label-studio导出json文件转换成训练所需的格式
import json
import os
import struct
import urllib.parse
from pathlib import Path


def get_image_size(image_path: str) -> tuple[int, int]:
    """获取图片尺寸 (width, height)。只读文件头，不加载完整图片。"""
    with open(image_path, "rb") as f:
        header = f.read(24)
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            # PNG: width at byte 16, height at byte 20 (big-endian)
            w, h = struct.unpack(">II", header[16:24])
            return w, h
        # For JPEG
        if header[:2] == b"\xff\xd8":
            f.seek(2)
            while True:
                marker = f.read(2)
                if marker[:1] != b"\xff":
                    break
                seg_len = struct.unpack(">H", f.read(2))[0]
                if marker[1:2] in (b"\xc0", b"\xc2"):
                    h, w = struct.unpack(">HH", f.read(4))
                    return w, h
                f.seek(seg_len - 2, 1)
    print(f"Warning: cannot determine size for {image_path}, using 100x100 fallback")
    return 100, 100


def main():
    json_path = "project-5-at-2026-05-15-11-36-e991b6c8.json"
    # 图片目录（相对于此脚本）
    image_dir = Path(__file__).resolve().parent.parent / "image"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for task_idx, task in enumerate(data):
        # 获取图片路径及实际尺寸
        image_url = task.get("data", {}).get("ocr_image", "")
        if image_url:
            image_name = os.path.basename(image_url)
            image_name = urllib.parse.unquote(image_name)
            if "-" in image_name:
                image_name = image_name.split("-", 1)[-1]
            image_path = image_dir / image_name
            # 若精确匹配失败，尝试模糊匹配（替换 - 为 _ 等常见变体）
            if not image_path.exists():
                for variant in [image_name.replace("-", "_"), image_name.replace("_", "-")]:
                    candidate = image_dir / variant
                    if candidate.exists():
                        image_path = candidate
                        break
            if image_path.exists():
                actual_w, actual_h = get_image_size(str(image_path))
            else:
                print(f"Warning: image not found: {image_path}")
                actual_w, actual_h = 100, 100
        else:
            actual_w, actual_h = 100, 100

        results = task.get("annotations", [{}])[0].get("result", [])

        boxes = {}
        for r in results:
            rid = r.get("id")
            if rid not in boxes:
                boxes[rid] = {}

            val = r.get("value", {})
            if r.get("type") == "rectanglelabels":
                # Label-Studio 默认坐标格式是相对总体宽度高度的百分比 (0-100)
                # original_width/original_height 可能为 None，则用图片实际尺寸
                orig_w = val.get("original_width") or actual_w
                orig_h = val.get("original_height") or actual_h

                # 转换出适合OCR训练的绝对坐标点 (absolute pixels)
                x_pct = val.get("x", 0)
                y_pct = val.get("y", 0)
                w_pct = val.get("width", 0)
                h_pct = val.get("height", 0)

                x = x_pct * orig_w / 100.0
                y = y_pct * orig_h / 100.0
                w = w_pct * orig_w / 100.0
                h = h_pct * orig_h / 100.0

                # 构建标注框四点坐标 [ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ]
                boxes[rid]["points"] = [
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ]

                labels = val.get("rectanglelabels", [])
                boxes[rid]["label"] = labels[0] if labels else "text"

            elif r.get("type") == "textarea":
                texts = val.get("text", [])
                # 拼合转录文本
                boxes[rid]["transcription"] = "\n".join(texts)

        # 聚合成列表格式
        image_data = []

        for rid, item in boxes.items():
            if "points" in item:
                ocr_item = {
                    "transcription": item.get("transcription", ""),
                    "points": item["points"],
                    "label": item.get("label", "text")
                }
                image_data.append(ocr_item)

        # 构造导出文件名
        if image_url:
            base_name = os.path.splitext(image_name)[0]
        else:
            base_name = f"image_{task_idx}"

        out_name = f"{base_name}_gt.json"

        # 每个Task代表一张图的标注，将其分别输出到独立的JSON中
        with open(out_name, "w", encoding="utf-8") as f_out:
            json.dump(image_data, f_out, ensure_ascii=False, indent=2)

        print(f"Exported {len(image_data)} regions to {out_name}")


if __name__ == "__main__":
    main()