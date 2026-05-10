# Oracle Bone Script Document Parsing Benchmark

古文字（甲骨文）文档解析评测基准。评估模型在混合内容文档上的：正文 OCR、隶定字(IDS序列)、甲骨文字图检测、插图检测。

## 常用命令

```bash
# 从 Label-Studio 导出生成 GT
cd data/GT && python process.py

# 用合成数据测试 pipeline（GT 加噪声模拟预测）
python run_benchmark.py

# 评测 PaddleOCR-VL-1.5 模型
python run_benchmark.py --model-dir data/prediction/paddleV1.5 --model paddle_ocr_vl --output benchmark_results_paddle.json

# 自定义 IOU 阈值 / 禁用树编辑距离
python run_benchmark.py --iou 0.6 --no-tree-edit
```

## 架构要点

### 四层评测流水线
1. **Detection** — IOU + Hungarian 1-to-1 匹配，小区域（<100px²）用中心距离回退
2. **Classification** — 匹配对中标签一致的准确率
3. **Recognition** — text 用 CER (Levenshtein)，liding 用 (字符串NED + 树编辑距离)/2
4. **Composite** — weighted_sum(label_scores) × classification_accuracy

### 标签体系
| 标签 | 含义 | 评测方式 |
|------|------|----------|
| `text` | 现代中文正文 | IOU + CER |
| `liding` | 隶定字 IDS 序列 | IOU + 字符串NED + 树编辑距离 |
| `oracle` | 甲骨文字图 | 仅检测(IOU) |
| `picture` | 插图/照片 | 仅检测(IOU) |

### IDS 树编辑距离
- Zhang-Shasha 算法，解析 IDS 字符串为树
- cost_rename(相同)=0, cost_rename(两者皆操作符)=0.5, cost_rename(其他)=1.0
- 已知局限：同一汉字可有多种合法 IDS 表示，树编辑距离只能缓解不能根治

### 坐标系统
- 所有坐标使用**绝对像素**（absolute pixels）
- GT 存四点 `[[x1,y1],...,[x4,y4]]`，处理时转为 bbox `[min(x), min(y), max(x), max(y)]`

### 适配器模式
模型预测通过适配器归一化为 `ImageAnnotation`：
- `PaddleOCRAdapter` — 标准 PaddleOCR 格式（flat list with points/text）
- `PaddleOCRVAdapter` — PaddleOCR-VL 格式（嵌套 `prunedResult.parsing_res_list`，block_label→label 映射）
- `GeminiAdapter` — Gemini 模型输出

## 已知问题与局限

### 粒度失配（核心问题）
GT 标注在字符/词组级（一张图 75 个 region），版面分析模型输出 block 级大框（一张图 17 个 prediction）。导致：
- 大量 GT region 因 IOU < 0.5 无法匹配（如古文字导论_165 整张图 0 匹配）
- 评测实际衡量的是"标注粒度一致性"而非"模型版面理解能力"

### 标签体系与通用 OCR 不兼容
- 通用 OCR 模型无法区分 `liding` vs `text`（外观上都是嵌在正文中的字符）
- oracle 字符图需要专门的视觉检测能力，OCR 模型不具备
- 当前没有跨标签 fallback 机制

### 样本量
- 仅 2 张图：古文字导论_165（古籍竖排+穿插甲骨字图）、小屯南地甲骨-2118（著录目录）
- 页面类型迥异但评测未区分，结果缺乏统计显著性

### Composite Score 设计
- `composite *= classification_acc` 是乘法惩罚——分类全错则综合分归零
- liding 权重 0.40（最高），对不具备 liding 检测的模型上限仅 0.60

## 当前数据集状态

### GT 文件
- `data/GT/古文字导论_165_gt.json` — 10 regions (5 text + 5 oracle)，竖排古籍页面
- `data/GT/小屯南地甲骨-2118_gt.json` — 75 regions (52 text + 11 liding + 12 oracle)，著录目录

### 模型预测
- `data/prediction/paddleV1.5/` — PaddleOCR-VL-1.5 输出，需用 `paddle_ocr_vl` 适配器

### 最新评测结果 (PaddleOCR-VL-1.5)
- 综合分 0.059，文本 CER 10.6%
- 图1 零匹配（粒度失配），图2 text F1=0.377
- liding/oracle 全覆盖失败
