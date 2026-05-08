# 将label-studio导出json文件转换成训练所需的格式
import json
import os
import urllib.parse

def main():
    json_path = "project-5-at-2026-05-08-23-23-83ac239c.json"
    
    # 避免直接读取整个大文件，使用 json.load() 流式加载或者按需解析
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for task_idx, task in enumerate(data):
        results = task.get("annotations", [{}])[0].get("result", [])
        
        boxes = {}
        for r in results:
            rid = r.get("id")
            if rid not in boxes:
                boxes[rid] = {}
                
            val = r.get("value", {})
            if r.get("type") == "rectanglelabels":
                # Label-Studio 默认坐标格式是相对总体宽度高度的百分比
                orig_w = val.get("original_width", 100)
                orig_h = val.get("original_height", 100)
                
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
        image_url = task.get("data", {}).get("ocr_image", "")
        if image_url:
            image_name = os.path.basename(image_url)
            image_name = urllib.parse.unquote(image_name)
            if "-" in image_name:
                image_name = image_name.split("-", 1)[-1]
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