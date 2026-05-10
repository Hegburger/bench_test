# Oracle Bone Script Document Parsing Benchmark

古文字（甲骨文）文档解析评测基准。评估模型在混合内容文档页面上的表现：正文、隶定字(IDS序列)、甲骨文字图、插图。

## 目录结构

```
bench_test/
├── data/
│   ├── GT/                          # 标注数据与预处理脚本
│   │   ├── process.py               # Label-Studio 导出 → GT JSON 转换
│   │   ├── project-5-*.json         # Label-Studio 原始导出
│   │   └── *_gt.json                # 单图标注文件（绝对像素坐标）
│   └── image/                       # 原始文档图片
│       ├── 古文字导论_165.png
│       └── 小屯南地甲骨_2118.png
├── benchmark/                       # 评测核心包
│   ├── __init__.py                  # 公开 API
│   ├── schema.py                    # Region / ImageAnnotation 数据结构
│   ├── config.py                    # EvalConfig 评测配置
│   ├── data_loader.py               # GT JSON 加载
│   ├── ids_parser.py                # IDS 字符串 → 树解析器
│   ├── tree_edit.py                 # Zhang-Shasha 树编辑距离
│   ├── matcher.py                   # IOU + Hungarian 匹配（含小区域中心距离回退）
│   ├── metrics.py                   # CER / NED / liding评分 / 综合评分
│   ├── evaluator.py                 # 单图 & 数据集评测流水线
│   ├── reporter.py                  # 控制台输出 & JSON 报告
│   └── adapters/                    # 模型输出适配器
│       ├── base.py                  # 适配器基类
│       ├── paddle_ocr.py            # PaddleOCR 适配器
│       └── gemini.py                # Gemini 适配器
└── run_benchmark.py                 # CLI 入口
```

## 标签类型

| 标签 | 含义 | 检测评价 | 识别评价 |
|------|------|----------|----------|
| `text` | 现代中文正文 | IOU + F1 | CER (字符错误率) |
| `liding` | 隶定字 IDS 序列（如 `⿰至⿱⿰先先貝`） | IOU + F1 | 字符串 NED + 树编辑距离 |
| `oracle` | 甲骨文字图（嵌入图片） | IOU + F1 | 仅检测 |
| `picture` | 插图/照片 | IOU + F1 | 仅检测 |

## 评测架构（四层）

```
第1层: Detection（检测）
  ├── IOU 计算（两个区域框的交并比）
  ├── 小区域回退：面积 < 100px² 时用中心距离替代 IOU
  ├── Hungarian 最优 1-to-1 匹配
  └── 指标：Precision / Recall / F1（按标签分开）

第2层: Classification（分类）
  └── 匹配区域对中标签一致的准确率

第3层: Recognition（识别）
  ├── text: 字符编辑距离 CER = Levenshtein / max(|ref|, |hyp|)
  └── liding: (字符串 NED + 树编辑距离) / 2
       ├── 字符串 NED: 归一化编辑距离
       └── 树编辑距离: 基于 IDS 解析树的 Zhang-Shasha 算法
            ├── cost_rename(相同) = 0
            ├── cost_rename(两者皆 IDS 操作符) = 0.5
            ├── cost_rename(其他) = 1.0
            └── cost_delete / cost_insert = 1.0

第4层: Composite（综合）
  └── weighted_sum(label_scores) × classification_accuracy
       权重: text=0.35, liding=0.40, oracle=0.15, picture=0.10
```

## 坐标系统

所有坐标使用**绝对像素**（absolute pixels）。
- GT JSON 每条记录包含 `points` 字段：四点矩形 `[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]`
- 标注框 = `[min(x), min(y), max(x), max(y)]`
- `process.py` 从 Label-Studio 百分比坐标转换为绝对像素（需读取图片获取实际尺寸）

## IDS 解析

IDS (Ideographic Description Sequences) 使用 Unicode U+2FF0-U+2FFF 范围的操作符描述汉字结构：

| 操作符 | 含义 | 元数 |
|--------|------|------|
| ⿰ | 左右结构 | 2 |
| ⿱ | 上下结构 | 2 |
| ⿲ | 左中右结构 | 3 |
| ⿳ | 上中下结构 | 3 |
| ⿴ | 全包围 | 2 |
| ⿵ | 上三包围 | 2 |
| ⿶ | 下三包围 | 2 |
| ⿷ | 左三包围 | 2 |
| ⿸ | 左上包围 | 2 |
| ⿹ | 右上包围 | 2 |
| ⿺ | 左下包围 | 2 |
| ⿻ | 重叠 | 2 |

**已知局限**：同一汉字可能存在多种合法 IDS 表示（如 `⿰至⿱⿰先先貝` 与 `⿱⿰至先貝`），当前仅依靠树编辑距离的 `cost_rename(两者皆操作符)=0.5` 策略缓解。理想方案需引入 IDS 标准化或基于目标字符级别的比较。

## 使用方式

### 1. 生成 GT 文件

从 Label-Studio 导出的 JSON 生成单图标注：

```bash
cd data/GT
python process.py
# 输出: *_gt.json 文件（绝对像素坐标）
```

### 2. 运行评测

```bash
# 使用合成数据测试（从 GT 加噪声生成模拟预测）
python run_benchmark.py

# 自定义参数
python run_benchmark.py --gt-dir data/GT --output results.json --iou 0.5

# 禁用树编辑距离（仅用字符串 NED 评价 liding）
python run_benchmark.py --no-tree-edit

# 使用真实模型输出
python run_benchmark.py --model-dir predictions/ --model gemini
```

### 3. 编程接口

```python
from benchmark import (
    EvalConfig,
    load_gt_directory,
    evaluate_dataset,
    print_summary,
)

# 配置
config = EvalConfig(iou_threshold=0.5, liding_use_tree_edit=True)

# 加载 GT
gt_list = load_gt_directory("data/GT")
gt_dict = {ann.image_id: ann for ann in gt_list}

# 加载模型预测（需实现适配器）
from benchmark.adapters import GeminiAdapter
adapter = GeminiAdapter()
predictions = {}  # {image_id: ImageAnnotation from adapter.parse(...)}

# 评测
results = evaluate_dataset(predictions, gt_dict, config)
print_summary(results)
```

### 4. 添加新模型适配器

```python
from benchmark.adapters import ModelAdapter
from benchmark.schema import ImageAnnotation, Region

class MyModelAdapter(ModelAdapter):
    @property
    def model_name(self) -> str:
        return "MyModel"

    def parse(self, raw_output, image_id: str, image_path: str) -> ImageAnnotation:
        regions = []
        for item in raw_output:
            regions.append(Region(
                bbox=[item["x1"], item["y1"], item["x2"], item["y2"]],
                label=item.get("label", "text"),
                transcription=item.get("text", ""),
                confidence=item.get("confidence", 1.0),
            ))
        return ImageAnnotation(
            image_id=image_id,
            image_path=image_path,
            regions=regions,
        )
```

## 匹配策略

- **主策略**：IOU ≥ 阈值 (默认 0.5) 的预测框可匹配
- **小区域回退**：面积 < 100px² 的极小区域（如单字甲骨文字图），若 IOU 不足但中心距离 < 20px 或 2×平均边长，允许匹配
- **匈牙利算法**：在可匹配的候选对上求全局最优 1-to-1 匹配
- **按标签分评**：per-label 匹配仅在同标签区域内进行

## 配置项

所有配置见 [benchmark/config.py](benchmark/config.py)：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `iou_threshold` | 0.5 | IOU 匹配阈值 |
| `text_case_sensitive` | False | 文本 CER 是否区分大小写 |
| `liding_use_tree_edit` | True | 是否启用树编辑距离 |
| `weight_text` | 0.35 | text 综合权重 |
| `weight_liding` | 0.40 | liding 综合权重 |
| `weight_oracle` | 0.15 | oracle 综合权重 |
| `weight_picture` | 0.10 | picture 综合权重 |
| `cost_rename_both_operators` | 0.5 | IDS 操作符互变的代价 |
| `cost_rename_otherwise` | 1.0 | 不同字符的改名代价 |

## 依赖

- Python 3.10+
- 无第三方依赖（核心评测仅用标准库）
- `Pillow`（可选，用于 process.py 获取图片尺寸）
