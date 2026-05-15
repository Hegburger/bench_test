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

# Debug 模式：生成逐对匹配详情和识别细节
python run_benchmark.py --model-dir data/prediction/paddleV1.5 --model paddle_ocr_vl --debug --output debug_results.json

# 生成交互式可视化 HTML
python visualize_benchmark.py --debug debug_results.json --image-dir data/image --output benchmark_viz.html

# 自定义 IOU 阈值 / 禁用树编辑距离
python run_benchmark.py --iou 0.6 --no-tree-edit
```

## 架构要点

### 四层评测流水线
1. **Detection** — 三层级联匹配（IOU → 包含关系 → 中心距离）+ Hungarian 1-to-1 最优匹配
2. **Classification** — 匹配对中标签一致的准确率
3. **Recognition** — 直接匹配对 + 覆盖回退（覆盖识别），text 用 CER/substring_CER，liding 用 (字符串NED + 树编辑距离)/2
4. **Composite** — weighted_sum(label_scores) × classification_accuracy

### 匹配策略：三层级联回退

| 优先级 | 方法 | 条件 | 说明 |
|--------|------|------|------|
| 1 | **IOU 直接匹配** | IOU ≥ 0.5 | 标准交并比 |
| 2 | **包含关系回退** | max(gt_in_pred, pred_in_gt) ≥ 0.7 | 缓解粒度失配，大框套小框也能匹配 |
| 3 | **中心距离回退** | 双方面积 < 100px² 且中心距 < 阈值 | 极小区域（如单个甲骨字图）用位置近似 |

### 覆盖识别 (Coverage Recognition)

解决 1-to-1 matching 的固有限制。对于未被匹配的 GT 文本区域：
- 找最佳覆盖的预测框（coverage ≥ 0.5）
- text: 用 **子串 CER**（滑动窗口在长文本中找最佳对齐）
- liding: 标准 liding_score（需同标签预测）

这意味着即使 GT region 未被 bbox-matching 选中，其文本识别质量仍被评测。例如小屯南地甲骨-2118 的 52 个 text GT 全部获得了 OCR 评分（17 通过 matching + 41 通过 coverage）。

### 标签体系
| 标签 | 含义 | 评测方式 |
|------|------|----------|
| `text` | 现代中文正文 | IOU + CER / 子串CER |
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

### 可视化工具

`visualize_benchmark.py` 从 debug 结果生成自包含的交互式 HTML：
- 图层：全局 Pred/GT 开关、匹配/覆盖/未匹配关系、连线、文字标签
- 交互：点击框选 → 侧栏显示完整详情、悬浮预览、滚轮缩放、拖拽平移
- 颜色体系：蓝 = Pred，绿/紫/琥珀 = GT；实线 = 已匹配，虚线 = 覆盖，点线 = 未匹配

## 已知问题与局限

### 粒度失配（已缓解，未根治）
GT 标注在字符/词组级（一张图 75 个 region），版面分析模型输出 block 级大框（一张图 17 个 prediction）。当前通过**包含关系匹配 + 覆盖识别**缓解：
- 包含关系匹配让大框套小框的 pair 能被匹配到（解决 detection）
- 覆盖识别让未匹配的 GT text 也能被评测（解决 recognition）
- 但 detection recall 仍受 1-to-1 matching 限制：block 输出下 F1 上限 = 17/75 ≈ 0.37

### 标签体系与通用 OCR 不兼容
- 通用 OCR 模型无法区分 `liding` vs `text`（外观上都是嵌在正文中的字符）
- oracle 字符图需要专门的视觉检测能力，OCR 模型不具备
- 评测仅在同标签对上进行识别，避免误判

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

### 最新评测结果 (PaddleOCR-VL-1.5, IOU=0.5, containment=0.7)

| 指标 | 得分 | 说明 |
|------|------|------|
| text detection F1 | 0.496 | 17 pred (全text) vs 57 text GT |
| text recognition CER | 0.122 | 57/57 GT text 覆盖（包含关系 + 覆盖回退） |
| liding detection F1 | 0.000 | 模型无 liding 分类能力 |
| oracle detection F1 | 0.000 | 模型无 oracle 检测能力 |
| classification accuracy | 0.574 | 17 个匹配中有 6 个 mislabel |
| **综合分** | **0.087** | 受 liding/oracle F1=0 和 class_acc 惩罚影响 |

逐图：
- **古文字导论_165**: 4 pred vs 10 GT → 2 matched + 4 coverage, composite=0.077
- **小屯南地甲骨-2118**: 17 pred vs 75 GT → 17 matched + 41 coverage, composite=0.098
