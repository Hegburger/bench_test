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
│   ├── image/                       # 原始文档图片
│   │   ├── 古文字导论_165.png
│   │   └── 小屯南地甲骨_2118.png
│   └── prediction/
│       └── paddleV1.5/              # PaddleOCR-VL-1.5 模型预测输出
├── benchmark/                       # 评测核心包
│   ├── __init__.py                  # 公开 API
│   ├── schema.py                    # Region / ImageAnnotation 数据结构
│   ├── config.py                    # EvalConfig 评测配置
│   ├── data_loader.py               # GT JSON 加载
│   ├── ids_parser.py                # IDS 字符串 → 树解析器
│   ├── tree_edit.py                 # Zhang-Shasha 树编辑距离
│   ├── matcher.py                   # IOU + Containment + Hungarian 匹配
│   ├── metrics.py                   # CER / 子串CER / NED / liding评分 / 综合评分
│   ├── evaluator.py                 # 单图 & 数据集评测流水线
│   ├── reporter.py                  # 控制台输出 / JSON 报告 / Debug 详情
│   └── adapters/                    # 模型输出适配器
│       ├── base.py                  # 适配器基类
│       ├── paddle_ocr.py            # PaddleOCR / PaddleOCR-VL 适配器
│       └── gemini.py                # Gemini 适配器
├── run_benchmark.py                 # CLI 入口
├── benchmark_results.json           # 评测结果（JSON）
└── CLAUDE.md                        # 开发文档
```

## 标签类型

| 标签 | 含义 | 检测评价 | 识别评价 |
|------|------|----------|----------|
| `text` | 现代中文正文 | IOU/Containment + F1 | CER (字符错误率) |
| `liding` | 隶定字 IDS 序列（如 `⿰至⿱⿰先先貝`） | IOU/Containment + F1 | 字符串 NED + 树编辑距离 |
| `oracle` | 甲骨文字图（嵌入图片） | IOU/Containment + F1 | 仅检测 |
| `picture` | 插图/照片 | IOU/Containment + F1 | 仅检测 |

## 评测架构（四层）

```
第1层: Detection（检测）
  ├── 主策略: IOU ≥ 阈值 (默认 0.5)
  ├── 回退1 (Containment): 若 IOU 不足但 max(gt⊂pred, pred⊂gt) ≥ 0.7
  │     解决 GT 字符级标注 vs 模型布局级预测之间的粒度失配
  ├── 回退2 (中心距离): 面积 < 100px² 时用中心距离替代
  ├── Hungarian 最优 1-to-1 匹配
  └── 指标：Precision / Recall / F1（按标签分开）

第2层: Classification（分类）
  └── 匹配区域对中标签一致的准确率

第3层: Recognition（识别）
  ├── 直接匹配对: 同标签 matched pairs 的标准 CER/liding_score
  ├── 覆盖识别 (Coverage): 对未匹配的 GT 区域，找最佳覆盖的预测框
  │     ├── text: 子串 CER —— 滑动窗口在长预测文本中找 GT 子串的最佳匹配
  │     └── liding: 同标准 liding_score（需同标签预测）
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

## 匹配策略详解

### 三层回退机制

评测在匹配预测框与 GT 框时使用三层回退：

| 层级 | 条件 | 适用场景 |
|------|------|----------|
| **IOU** | IOU ≥ 0.5 | 预测框与 GT 框大小相近、位置对齐 |
| **Containment** | max(gt⊂pred, pred⊂gt) ≥ 0.7 | 模型输出 layout 级大框、GT 为字符级小框 |
| **中心距离** | 两框面积均 < 100px² 且中心距离近 | 极小甲骨文字图 |

其中 Containment 的两类包含关系：

```
gt⊂pred (gt_in_pred): GT 区域有多少比例在预测框内
  → 模型大框包含 GT 小文本区域（最常见）
pred⊂gt (pred_in_gt): 预测框有多少比例在 GT 框内
  → 模型预测是 GT 标注的子集
```

### 覆盖识别 (Coverage Recognition)

解决"一个大预测框包含多个 GT 区域"的问题：

1. 对未被匹配的 GT text 区域，找到覆盖比例最高的同标签预测框
2. 若覆盖 ≥ `recognition_cover_threshold`（默认 0.5），用子串 CER 评估
3. 子串 CER：在长预测文本上滑动窗口，找 GT 文本的最佳匹配位置

```
例：GT "國文字裡最古的" (7字) ⊆ Pred 整列文本 (200+字)
  → 子串 CER = 0.0 (在预测文本中精确定位到匹配子串)
```

**注意**：识别评测仅在同标签对上进行。若模型只能输出 `text` 标签，则不会错误地为 `liding`/`oracle` 计算识别分。

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

# 评测 PaddleOCR-VL-1.5
python run_benchmark.py --model-dir data/prediction/paddleV1.5 --model paddle_ocr_vl

# Debug 模式：显示每对预测与 GT 的匹配详情、识别得分、覆盖关系
python run_benchmark.py --model-dir data/prediction/paddleV1.5 --model paddle_ocr_vl --debug

# 自定义参数
python run_benchmark.py --iou 0.6 --no-tree-edit --output results.json
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
config = EvalConfig(
    iou_threshold=0.5,
    containment_threshold=0.7,
    recognition_cover_threshold=0.5,
    liding_use_tree_edit=True,
    debug=True,  # 启用逐对详情
)

# 加载 GT
gt_list = load_gt_directory("data/GT")
gt_dict = {ann.image_id: ann for ann in gt_list}

# 加载模型预测（需实现适配器）
from benchmark.adapters import PaddleOCRVAdapter
adapter = PaddleOCRVAdapter()
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

在 `run_benchmark.py` 的 `adapter_map` 中注册：

```python
from benchmark.adapters import MyModelAdapter

adapter_map = {
    ...
    "my_model": MyModelAdapter,
}
```

## 已知局限

### 粒度失配
GT 标注在字符/词组级（一张图 75 个 region），版面分析模型输出 block 级大框（一张图 17 个 prediction）。当前通过 Containment 匹配 + 覆盖识别缓解，但检测 Recall 仍受 1-to-1 匹配限制。

### 标签体系与通用 OCR 不兼容
通用 OCR 模型无法区分 `liding` vs `text`（外观上都是嵌在正文中的字符）。oracle 字符图需要专门的视觉检测能力。评测通过在识别层限制同标签对来避免误判。

### IDS 多义性
同一汉字可有多种合法 IDS 表示，树编辑距离只能缓解不能根治。

### 样本量
仅 2 张图，页面类型迥异（古籍竖排+穿插甲骨字图 vs 著录目录），结果缺乏统计显著性。

## 当前数据集

| 图片 | GT regions | text | liding | oracle | 版式 |
|------|-----------|------|--------|--------|------|
| 古文字导论_165 | 10 | 5 | 0 | 5 | 古籍竖排，正文穿插甲骨字图 |
| 小屯南地甲骨-2118 | 75 | 52 | 11 | 12 | 卜辞著录目录，横排 |

### 最新评测结果 (PaddleOCR-VL-1.5)

| 指标 | 得分 |
|------|------|
| text detection F1 | 0.496 |
| text recognition CER | 0.122 (52/52 GT 覆盖) |
| liding detection F1 | 0.000 (模型无 liding 分类能力) |
| oracle detection F1 | 0.000 (模型无 oracle 检测能力) |
| 综合得分 | 0.087 |

## 配置项

所有配置见 [benchmark/config.py](benchmark/config.py)：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `iou_threshold` | 0.5 | IOU 匹配阈值 |
| `containment_threshold` | 0.7 | 包含关系匹配阈值 (max(gt⊂pred, pred⊂gt)) |
| `recognition_cover_threshold` | 0.5 | 覆盖识别最低覆盖比例 |
| `text_case_sensitive` | False | 文本 CER 是否区分大小写 |
| `liding_use_tree_edit` | True | 是否启用树编辑距离 |
| `debug` | False | 是否收集逐对匹配/识别详情 |
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
