#!/usr/bin/env python3
"""Generate an interactive HTML visualization of benchmark debug results.

Usage:
    # Generate debug results first:
    python run_benchmark.py --model-dir data/prediction/paddleV1.5 --model paddle_ocr_vl --debug --output debug_results.json

    # Then visualize:
    python visualize_benchmark.py

    # Or with custom paths:
    python visualize_benchmark.py --debug debug_results.json --image-dir data/image --output viz.html
"""

import argparse
import base64
import json
import sys
from pathlib import Path

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>甲骨文文档解析 — Benchmark 可视化</title>
<style>
:root {
    --bg: #0d1117;
    --panel-bg: #161b22;
    --card-bg: #1c2333;
    --border: #30363d;
    --text: #c9d1d9;
    --text-dim: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --red: #f85149;
    --amber: #d2991d;
    --purple: #a371f7;
    --cyan: #39d2c0;
}

*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI','Microsoft YaHei',sans-serif;background:var(--bg);color:var(--text);display:flex;height:100vh;overflow:hidden}

/* ── Sidebar ── */
#sidebar{width:350px;min-width:350px;background:var(--panel-bg);border-right:1px solid var(--border);display:flex;flex-direction:column;overflow-y:auto}
#sidebar section{padding:14px 16px;border-bottom:1px solid var(--border)}
#sidebar h2{font-size:15px;color:var(--accent);margin-bottom:10px;display:flex;align-items:center;gap:6px}
#sidebar h3{font-size:12px;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px}

#image-select{width:100%;padding:7px 10px;background:var(--card-bg);color:var(--text);border:1px solid var(--border);border-radius:6px;font-size:13px;cursor:pointer}
#image-select:focus{outline:none;border-color:var(--accent)}

/* ── Stats ── */
.stat-row{display:flex;gap:6px;margin-bottom:6px}
.stat-item{flex:1;background:var(--card-bg);border-radius:6px;padding:8px 10px;text-align:center}
.stat-item .val{font-size:20px;font-weight:700;color:var(--accent);line-height:1.2}
.stat-item .val.good{color:var(--green)}
.stat-item .val.warn{color:var(--amber)}
.stat-item .val.bad{color:var(--red)}
.stat-item .lbl{font-size:10px;color:var(--text-dim);margin-top:2px}

/* ── Layer toggles ── */
.layer-group{margin-bottom:6px}
.layer-group h4{font-size:11px;color:var(--text-dim);margin-bottom:4px;text-transform:uppercase}
.layer-row{display:flex;flex-wrap:wrap;gap:4px}
.layer-btn{display:flex;align-items:center;gap:5px;padding:5px 8px;font-size:11px;cursor:pointer;border-radius:4px;border:1px solid var(--border);background:var(--card-bg);color:var(--text);transition:all 0.15s;user-select:none}
.layer-btn:hover{border-color:var(--accent)}
.layer-btn.active{background:#1f3a5f;border-color:var(--accent)}
.layer-btn input{display:none}
.layer-btn .dot{width:8px;height:8px;border-radius:1.5px;flex-shrink:0}

/* ── Color legend ── */
.legend-table{width:100%;border-collapse:collapse;font-size:10px}
.legend-table td{padding:3px 4px;border-bottom:1px solid rgba(48,54,61,0.5);vertical-align:middle}
.legend-table .swatch{width:22px;height:14px;border-radius:2px;display:inline-block}

/* ── Detail panel ── */
#detail-panel{min-height:100px;max-height:300px;overflow-y:auto}
#detail-panel table{width:100%;border-collapse:collapse;font-size:12px}
#detail-panel td{padding:3px 8px;border-bottom:1px solid rgba(48,54,61,0.5);vertical-align:top}
#detail-panel td.key{color:var(--text-dim);white-space:nowrap;width:68px}
#detail-panel td.val{word-break:break-all}
#detail-panel .match-iou{color:#2196f3;font-weight:600}
#detail-panel .match-contain{color:var(--amber);font-weight:600}
#detail-panel .match-cover{color:var(--purple);font-weight:600}
#detail-panel .pass{color:var(--green);font-weight:600}
#detail-panel .fail{color:var(--red);font-weight:600}
#detail-panel .corner-badge{display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600}

/* ── Canvas area ── */
#main-canvas{flex:1;position:relative;overflow:hidden;cursor:crosshair}
canvas{position:absolute;top:0;left:0}
#tooltip{position:fixed;pointer-events:none;background:#1c2333;color:var(--text);padding:6px 10px;border-radius:6px;font-size:11px;display:none;z-index:999;max-width:340px;white-space:pre-wrap;border:1px solid var(--border);box-shadow:0 4px 12px rgba(0,0,0,0.5)}
#tooltip .tt-header{font-weight:600;margin-bottom:2px}
#status-bar{position:absolute;bottom:10px;left:16px;font-size:11px;color:var(--text-dim);pointer-events:none;background:rgba(13,17,23,0.7);padding:4px 10px;border-radius:4px}
.select-flash{animation:flash 0.4s ease-out}
@keyframes flash{0%{filter:brightness(2)}100%{filter:brightness(1)}}
</style>
</head>
<body>

<!-- ═══════════ SIDEBAR ═══════════ -->
<div id="sidebar">

    <section>
        <h2>甲骨文文档解析评测</h2>
        <select id="image-select"></select>
    </section>

    <section>
        <h3>评测摘要</h3>
        <div id="stats"></div>
    </section>

    <section>
        <h3>图层控制</h3>

        <div class="layer-group">
            <h4>全局</h4>
            <div class="layer-row">
                <label class="layer-btn active" id="lbl-show-pred"><span class="dot" style="background:#58a6ff"></span> Predictions</label>
                <label class="layer-btn active" id="lbl-show-gt"><span class="dot" style="background:#3fb950"></span> GT 标注</label>
            </div>
        </div>

        <div class="layer-group">
            <h4>匹配关系</h4>
            <div class="layer-row">
                <label class="layer-btn active" id="lbl-show-matched"><span class="dot" style="background:#3fb950"></span> 已匹配 (IOU / 包含)</label>
                <label class="layer-btn active" id="lbl-show-coverage"><span class="dot" style="background:#a371f7"></span> 覆盖识别</label>
                <label class="layer-btn active" id="lbl-show-lines"><span class="dot" style="background:#8b949e"></span> 匹配连线</label>
            </div>
        </div>

        <div class="layer-group">
            <h4>未匹配</h4>
            <div class="layer-row">
                <label class="layer-btn active" id="lbl-show-umpred"><span class="dot" style="background:#f85149"></span> 未匹配 Pred</label>
                <label class="layer-btn active" id="lbl-show-umgt"><span class="dot" style="background:#d2991d"></span> 未匹配 GT</label>
            </div>
        </div>

        <div class="layer-group">
            <h4>标注</h4>
            <div class="layer-row">
                <label class="layer-btn active" id="lbl-show-labels"><span class="dot" style="background:#c9d1d9"></span> 文字标签</label>
            </div>
        </div>
    </section>

    <section>
        <h3>颜色图例</h3>
        <table class="legend-table">
        <tr><td><span class="swatch" style="background:rgba(88,166,255,0.2);border:2px solid #58a6ff;border-radius:2px"></span></td><td><b>Pred 框</b> (蓝色系)</td><td>模型预测区域</td></tr>
        <tr><td><span class="swatch" style="background:rgba(63,185,80,0.2);border:2px solid #3fb950"></span></td><td><b>GT 框</b> (绿色系)</td><td>真实标注区域</td></tr>
        <tr><td style="padding-top:6px" colspan="3"><b>框线样式 = 匹配状态</b></td></tr>
        <tr><td><span class="swatch" style="border:2px solid #3fb950;background:transparent"></span></td><td>实线框</td><td>已匹配 (IOU / 包含关系)</td></tr>
        <tr><td><span class="swatch" style="border:2px dashed #a371f7;background:transparent"></span></td><td>虚线框</td><td>覆盖识别 (未匹配但覆盖 GT)</td></tr>
        <tr><td><span class="swatch" style="border:1.5px dashed #f85149;background:transparent"></span></td><td>点线框</td><td>未匹配 (孤立区域)</td></tr>
        <tr><td style="padding-top:6px" colspan="3"><b>连线颜色 = 匹配方式</b></td></tr>
        <tr><td><span class="swatch" style="background:#2196f3"></span></td><td>蓝色连线</td><td>IOU 直接匹配</td></tr>
        <tr><td><span class="swatch" style="background:#d2991d"></span></td><td>琥珀连线</td><td>包含关系匹配</td></tr>
        <tr><td><span class="swatch" style="background:#a371f7"></span></td><td>紫色连线</td><td>覆盖识别</td></tr>
        <tr><td style="padding-top:6px" colspan="3"><b>选中 = 青蓝色角标 + 发光</b></td></tr>
        <tr><td><span class="swatch" style="background:#39d2c0"></span></td><td>角标 / 脉冲</td><td>当前点击选中的区域</td></tr>
        </table>
    </section>

    <section>
        <h3>选中区域详情</h3>
        <div id="detail-panel"><i style="color:var(--text-dim)">← 点击画布上的框查看详情</i></div>
    </section>

</div>

<!-- ═══════════ CANVAS ═══════════ -->
<div id="main-canvas">
    <canvas id="canvas"></canvas>
    <div id="status-bar">滚轮缩放 | 拖拽平移 | 点击框选 | 悬停预览</div>
</div>
<div id="tooltip"></div>

<script type="application/json" id="benchmark-data">
__DATA_PLACEHOLDER__
</script>
<script>
// ── Data ──
const DATA = JSON.parse(document.getElementById('benchmark-data').textContent);

// ── Color palette ──
const C = {
    // Pred fills (blue tints)
    predFill:      'rgba(88,166,255,0.20)',
    predFillCov:   'rgba(88,166,255,0.12)',
    predFillUnm:   'rgba(150,170,200,0.25)',
    // Pred strokes
    predStroke:    '#58a6ff',
    predStrokeCov: '#7cb8ff',
    predStrokeUnm: '#f85149',

    // GT fills (green/amber tints)
    gtFill:        'rgba(63,185,80,0.20)',
    gtFillCov:     'rgba(163,113,247,0.18)',
    gtFillUnm:     'rgba(210,153,29,0.22)',
    // GT strokes
    gtStroke:      '#3fb950',
    gtStrokeCov:   '#a371f7',
    gtStrokeUnm:   '#d2991d',

    // Lines
    lineIOU:       'rgba(33,150,243,0.55)',
    lineContain:   'rgba(210,153,29,0.55)',
    lineCover:     'rgba(163,113,247,0.45)',

    // Selection
    selColor:      '#39d2c0',
    selGlow:       'rgba(57,210,192,0.35)',

    // Label text
    lblMatch:      '#c9d1d9',
    lblCover:      '#d2a8ff',
    lblUnmatch:    '#ffa198',
};

// ── State ──
let currentImageId = null;
let currentImage = null;
let selectedBox = null;
let zoom = 1, panX = 0, panY = 0;
let dragging = false, dragStart = {};

// ── DOM refs ──
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const imageSelect = document.getElementById('image-select');
const statsDiv = document.getElementById('stats');
const detailPanel = document.getElementById('detail-panel');
const tooltip = document.getElementById('tooltip');
const statusBar = document.getElementById('status-bar');
const mainCanvas = document.getElementById('main-canvas');

// ── Layer state from toggle buttons ──
const layers = {
    showPred: true,
    showGT: true,
    showMatched: true,
    showCoverage: true,
    showUnmatchedPred: true,
    showUnmatchedGT: true,
    showLines: true,
    showLabels: true,
};

function bindLayer(id, key) {
    const lbl = document.getElementById(id);
    const cb = lbl.querySelector('input[type=checkbox]');
    lbl.addEventListener('click', () => {
        layers[key] = !layers[key];
        lbl.classList.toggle('active', layers[key]);
        if (currentImage) draw();
    });
    layers[key] = lbl.classList.contains('active');
}

bindLayer('lbl-show-pred',    'showPred');
bindLayer('lbl-show-gt',      'showGT');
bindLayer('lbl-show-matched', 'showMatched');
bindLayer('lbl-show-coverage','showCoverage');
bindLayer('lbl-show-lines',   'showLines');
bindLayer('lbl-show-umpred',  'showUnmatchedPred');
bindLayer('lbl-show-umgt',    'showUnmatchedGT');
bindLayer('lbl-show-labels',  'showLabels');

// ── Init ──
function init() {
    let ids = Object.keys(DATA.images);
    if (ids.length === 0) return;
    ids.forEach(id => {
        let opt = document.createElement('option');
        opt.value = id;
        opt.textContent = DATA.images[id].name || id;
        imageSelect.appendChild(opt);
    });
    imageSelect.addEventListener('change', () => loadImage(imageSelect.value));

    // Mouse wheel zoom
    mainCanvas.addEventListener('wheel', e => {
        e.preventDefault();
        let factor = e.deltaY < 0 ? 1.1 : 0.9;
        let rect = mainCanvas.getBoundingClientRect();
        let mx = e.clientX - rect.left;
        let my = e.clientY - rect.top;
        let newZoom = Math.max(0.1, Math.min(10, zoom * factor));
        panX = mx - (mx - panX) * (newZoom / zoom);
        panY = my - (my - panY) * (newZoom / zoom);
        zoom = newZoom;
        draw();
    });

    // Pan
    mainCanvas.addEventListener('mousedown', e => {
        if (e.button === 0) {
            dragging = true;
            dragStart = { x: e.clientX - panX, y: e.clientY - panY };
        }
    });
    window.addEventListener('mouseup', () => { dragging = false; });

    // Move: pan or hover
    window.addEventListener('mousemove', e => {
        if (dragging) {
            panX = e.clientX - dragStart.x;
            panY = e.clientY - dragStart.y;
            draw();
            tooltip.style.display = 'none';
            return;
        }
        if (!currentImage) return;
        let rect = mainCanvas.getBoundingClientRect();
        let mx = e.clientX - rect.left;
        let my = e.clientY - rect.top;
        let boxes = hitTest(mx, my);
        if (boxes.length > 0) {
            showTooltip(e.clientX + 16, e.clientY + 16, boxes[0]);
            mainCanvas.style.cursor = 'pointer';
        } else {
            tooltip.style.display = 'none';
            mainCanvas.style.cursor = selectedBox ? 'default' : 'crosshair';
        }
    });

    // Click to select / deselect
    mainCanvas.addEventListener('click', e => {
        if (dragging) return;
        let rect = mainCanvas.getBoundingClientRect();
        let mx = e.clientX - rect.left;
        let my = e.clientY - rect.top;
        let boxes = hitTest(mx, my);
        if (boxes.length > 0 && boxes[0] === selectedBox) {
            // Click same box: deselect
            selectedBox = null;
        } else {
            selectedBox = boxes.length > 0 ? boxes[0] : null;
        }
        updateDetail();
        draw();
        // Auto-scroll detail panel to top
        detailPanel.scrollTop = 0;
    });

    // Keyboard: Esc to deselect
    window.addEventListener('keydown', e => {
        if (e.key === 'Escape') { selectedBox = null; updateDetail(); draw(); }
    });

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    loadImage(ids[0]);
}

function resizeCanvas() {
    canvas.width = mainCanvas.clientWidth;
    canvas.height = mainCanvas.clientHeight;
    if (currentImage) draw();
}

// ── Image loading ──
function loadImage(imageId) {
    currentImageId = imageId;
    selectedBox = null;
    let info = DATA.images[imageId];
    if (!info) return;

    let img = new Image();
    img.onload = () => {
        currentImage = { img, info };
        let cw = mainCanvas.clientWidth;
        let ch = mainCanvas.clientHeight;
        let scale = Math.min(cw / img.naturalWidth, ch / img.naturalHeight, 0.95);
        zoom = scale;
        panX = (cw - img.naturalWidth * scale) / 2;
        panY = (ch - img.naturalHeight * scale) / 2;
        updateStats();
        updateDetail();
        draw();
    };
    img.src = info.dataUri;
}

// ── Hit testing ──
function hitTest(mx, my) {
    if (!currentImage) return [];
    let { info } = currentImage;

    function boxContains(bx, by, bw, bh) {
        let ix = (mx - panX) / zoom;
        let iy = (my - panY) / zoom;
        let margin = 4 / zoom; // small tolerance for thin boxes
        return ix >= bx - margin && ix <= bx + bw + margin
            && iy >= by - margin && iy <= by + bh + margin;
    }

    function bboxArea(bb) {
        if (!bb || bb.length < 4) return Infinity;
        return (bb[2] - bb[0]) * (bb[3] - bb[1]);
    }

    let allRegions = [];

    for (let p of info.pairs) {
        if ((p.type === 'matched' && !layers.showMatched) ||
            (p.type === 'coverage' && !layers.showCoverage)) continue;

        let pbb = p.pred_bbox, gbb = p.gt_bbox;
        if (layers.showPred && boxContains(pbb[0], pbb[1], pbb[2]-pbb[0], pbb[3]-pbb[1]))
            allRegions.push({...p, hitOn: 'pred'});
        if (layers.showGT && boxContains(gbb[0], gbb[1], gbb[2]-gbb[0], gbb[3]-gbb[1]))
            allRegions.push({...p, hitOn: 'gt'});
    }

    if (layers.showUnmatchedPred) {
        for (let up of (info.unmatched_pred || [])) {
            if (!layers.showPred) continue;
            let bb = up.bbox;
            if (boxContains(bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1]))
                allRegions.push({...up, hitOn: 'pred', type: 'unmatched_pred'});
        }
    }

    if (layers.showUnmatchedGT) {
        for (let ug of (info.unmatched_gt || [])) {
            if (!layers.showGT) continue;
            let bb = ug.bbox;
            if (boxContains(bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1]))
                allRegions.push({...ug, hitOn: 'gt', type: 'unmatched_gt'});
        }
    }

    // Smallest first = most precise
    allRegions.sort((a, b) => {
        let ab = a.hitOn === 'pred' ? (a.pred_bbox || a.bbox) : (a.gt_bbox || a.bbox);
        let bb = b.hitOn === 'pred' ? (b.pred_bbox || b.bbox) : (b.gt_bbox || b.bbox);
        return bboxArea(ab) - bboxArea(bb);
    });

    return allRegions;
}

// ── Tooltip ──
function showTooltip(x, y, item) {
    let lines = [];
    let header = '';

    if (item.type === 'matched') {
        let method = item.match_method === 'iou' ? 'IOU匹配' : '包含关系匹配';
        header = `已匹配 · ${method}`;
        lines.push(`命中: ${item.hitOn === 'pred' ? 'Pred框' : 'GT框'}`);
        lines.push(`Pred: ${item.pred_label}  →  GT: ${item.gt_label}`);
        lines.push(`匹配分: ${item.match_score?.toFixed(4)}`);
        if (item.rec_score !== undefined && item.rec_score !== null) {
            let rs = typeof item.rec_score === 'object'
                ? `strNED=${item.rec_score.string_ned} comb=${item.rec_score.combined}`
                : item.rec_score;
            lines.push(`识别错误率: ${rs}`);
        }
    } else if (item.type === 'coverage') {
        header = '覆盖识别';
        lines.push(`命中: ${item.hitOn === 'pred' ? 'Pred框' : 'GT框'}`);
        lines.push(`覆盖率: ${item.coverage?.toFixed(3)}`);
        lines.push(`Pred: ${item.pred_label}  →  GT: ${item.gt_label}`);
        if (item.rec_score !== undefined && item.rec_score !== null) {
            lines.push(`识别错误率: ${item.rec_score}`);
        }
    } else if (item.type === 'unmatched_pred') {
        header = '未匹配 Prediction';
        lines.push(`标签: ${item.label}`);
    } else if (item.type === 'unmatched_gt') {
        header = '未匹配 GT';
        lines.push(`标签: ${item.label}`);
    }

    tooltip.innerHTML = `<div class="tt-header">${header}</div>${lines.join('<br>')}`;
    tooltip.style.display = 'block';
    // Keep tooltip within viewport
    let maxX = window.innerWidth - 350;
    let maxY = window.innerHeight - 100;
    tooltip.style.left = Math.min(x, maxX) + 'px';
    tooltip.style.top = Math.min(y, maxY) + 'px';
}

// ── Detail panel ──
function updateDetail() {
    if (!selectedBox) {
        detailPanel.innerHTML = '<i style="color:var(--text-dim)">← 点击画布上的框查看详情<br><br>提示: 再次点击同一框取消选中<br>按 Esc 也可取消</i>';
        return;
    }
    let s = selectedBox;
    let html = '<table>';

    // Type + match info
    if (s.type === 'matched') {
        let method = s.match_method === 'iou' ? 'IOU 直接匹配' : '包含关系匹配';
        let cls = s.match_method === 'iou' ? 'match-iou' : 'match-contain';
        html += `<tr><td class="key">类型</td><td class="val"><span class="corner-badge" style="background:${s.match_method==='iou'?'#1a3a5c':'#3d2e0a'};color:${s.match_method==='iou'?'#58a6ff':'#d2991d'}">${method}</span> <span style="font-size:11px;color:var(--text-dim)">命中:${s.hitOn==='pred'?'Pred':'GT'}</span></td></tr>`;
        html += `<tr><td class="key">匹配分数</td><td class="val">${s.match_score?.toFixed(4)}</td></tr>`;
    } else if (s.type === 'coverage') {
        html += `<tr><td class="key">类型</td><td class="val"><span class="corner-badge" style="background:#2a1a3c;color:#a371f7">覆盖识别</span> <span style="font-size:11px;color:var(--text-dim)">命中:${s.hitOn==='pred'?'Pred':'GT'}</span></td></tr>`;
        html += `<tr><td class="key">覆盖率</td><td class="val match-cover">${s.coverage?.toFixed(4)}</td></tr>`;
    } else if (s.type === 'unmatched_pred') {
        html += `<tr><td class="key">类型</td><td class="val"><span class="corner-badge" style="background:#3d1a1a;color:#f85149">未匹配 Pred</span></td></tr>`;
    } else if (s.type === 'unmatched_gt') {
        html += `<tr><td class="key">类型</td><td class="val"><span class="corner-badge" style="background:#3d2e0a;color:#d2991d">未匹配 GT</span></td></tr>`;
    }

    // Labels
    let predLabel = s.pred_label || (s.label && s.type === 'unmatched_pred' ? s.label : '');
    let gtLabel = s.gt_label || (s.label && s.type === 'unmatched_gt' ? s.label : '');
    if (predLabel || gtLabel) {
        let labelMatch = predLabel === gtLabel;
        if (predLabel) {
            html += `<tr><td class="key">Pred 标签</td><td class="val">${predLabel}</td></tr>`;
        }
        if (gtLabel) {
            let cls = (s.type === 'matched' || s.type === 'coverage') ? (labelMatch ? 'pass' : 'fail') : '';
            html += `<tr><td class="key">GT 标签</td><td class="val ${cls}">${gtLabel}${!labelMatch && cls ? ' ✗ 不匹配' : ''}</td></tr>`;
        }
    }

    // Bboxes
    if (s.pred_bbox) {
        let bb = s.pred_bbox.map(v => Math.round(v));
        html += `<tr><td class="key">Pred 坐标</td><td class="val">[${bb.join(', ')}]  ${bb[2]-bb[0]}×${bb[3]-bb[1]}px</td></tr>`;
    }
    if (s.gt_bbox) {
        let bb = s.gt_bbox.map(v => Math.round(v));
        html += `<tr><td class="key">GT 坐标</td><td class="val">[${bb.join(', ')}]  ${bb[2]-bb[0]}×${bb[3]-bb[1]}px</td></tr>`;
    } else if (s.bbox) {
        let bb = s.bbox.map(v => Math.round(v));
        html += `<tr><td class="key">坐标</td><td class="val">[${bb.join(', ')}]  ${bb[2]-bb[0]}×${bb[3]-bb[1]}px</td></tr>`;
    }

    // Recognition score
    if (s.rec_score !== undefined && s.rec_score !== null) {
        let rs = s.rec_score;
        let rsStr, rsClass;
        if (typeof rs === 'object') {
            rsStr = `字符串NED=${rs.string_ned?.toFixed(4)}  `;
            if (rs.tree_ted !== null && rs.tree_ted !== undefined) rsStr += `树编辑=${rs.tree_ted.toFixed(4)}  `;
            rsStr += `综合=${rs.combined?.toFixed(4)}`;
            rsClass = rs.combined < 0.1 ? 'pass' : (rs.combined > 0.5 ? 'fail' : '');
        } else {
            rsStr = `${rs}`;
            rsClass = rs < 0.1 ? 'pass' : (rs > 0.5 ? 'fail' : '');
        }
        html += `<tr><td class="key">识别错误率</td><td class="val ${rsClass}">${rsStr} <span style="font-size:10px;color:var(--text-dim)">(0=完美, 越低越好)</span></td></tr>`;
    }

    // Text content
    if (s.pred_text) {
        html += `<tr><td class="key">Pred 文本</td><td class="val" style="font-size:11px">${escapeHtml(s.pred_text)}</td></tr>`;
    }
    if (s.gt_text) {
        html += `<tr><td class="key">GT 文本</td><td class="val" style="font-size:11px">${escapeHtml(s.gt_text)}</td></tr>`;
    }
    if ((!s.pred_text && !s.gt_text) && s.text) {
        html += `<tr><td class="key">文本</td><td class="val" style="font-size:11px">${escapeHtml(s.text)}</td></tr>`;
    }

    // Indices
    if (s.pred_idx !== undefined) html += `<tr><td class="key">Pred #</td><td class="val">${s.pred_idx}</td></tr>`;
    if (s.gt_idx !== undefined) html += `<tr><td class="key">GT #</td><td class="val">${s.gt_idx}</td></tr>`;

    html += '</table>';
    detailPanel.innerHTML = html;
}

function escapeHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Stats grid ──
function updateStats() {
    let info = currentImage?.info;
    if (!info || !info.summary) { statsDiv.innerHTML = ''; return; }
    let s = info.summary;

    function item(val, lbl, cls='') {
        return `<div class="stat-item"><div class="val ${cls}">${val}</div><div class="lbl">${lbl}</div></div>`;
    }

    let html = '<div class="stat-row">';
    html += item(s.num_pred, 'Pred', '');
    html += item(s.num_gt, 'GT', '');
    html += item(s.matched_pairs, '已匹配', 'good');
    html += item(s.coverage_pairs, '覆盖识别', '');
    html += '</div><div class="stat-row">';
    html += item(s.composite.toFixed(4), '综合分', s.composite > 0.3 ? 'good' : s.composite > 0.1 ? 'warn' : 'bad');
    html += item(s.class_acc.toFixed(3), '分类准确率', s.class_acc > 0.7 ? 'good' : s.class_acc > 0.4 ? 'warn' : 'bad');
    let det = s.detection || {};
    let rec = s.recognition || {};
    for (let [lbl, d] of Object.entries(det)) {
        if (lbl === 'overall') continue;
        html += item(d.f1.toFixed(3), `${lbl} F1`, d.f1 > 0.5 ? 'good' : d.f1 > 0.2 ? 'warn' : '');
    }
    for (let [lbl, r] of Object.entries(rec)) {
        html += item(r.mean_error.toFixed(4), `${lbl} CER`, r.mean_error < 0.15 ? 'good' : r.mean_error < 0.4 ? 'warn' : 'bad');
    }
    html += '</div>';
    statsDiv.innerHTML = html;
}

// ── Drawing helpers ──
function drawBox(bbox, fill, stroke, lw, dash) {
    let [x, y, x2, y2] = bbox;
    let w = x2 - x, h = y2 - y;
    ctx.save();
    if (fill) { ctx.fillStyle = fill; ctx.fillRect(x, y, w, h); }
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lw / zoom;
    if (dash) ctx.setLineDash(dash.map(d => d / zoom));
    ctx.strokeRect(x, y, w, h);
    ctx.restore();
}

function drawCorners(bbox, color, size) {
    let [x, y, x2, y2] = bbox;
    let s = size / zoom;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 3 / zoom;
    ctx.lineCap = 'round';
    ctx.setLineDash([]);
    // top-left
    ctx.beginPath(); ctx.moveTo(x, y+s); ctx.lineTo(x, y); ctx.lineTo(x+s, y); ctx.stroke();
    // top-right
    ctx.beginPath(); ctx.moveTo(x2-s, y); ctx.lineTo(x2, y); ctx.lineTo(x2, y+s); ctx.stroke();
    // bottom-right
    ctx.beginPath(); ctx.moveTo(x2, y2-s); ctx.lineTo(x2, y2); ctx.lineTo(x2-s, y2); ctx.stroke();
    // bottom-left
    ctx.beginPath(); ctx.moveTo(x+s, y2); ctx.lineTo(x, y2); ctx.lineTo(x, y2-s); ctx.stroke();
    ctx.restore();
}

function drawLine(bboxA, bboxB, stroke, lw) {
    ctx.save();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lw / zoom;
    ctx.setLineDash([5 / zoom, 5 / zoom]);
    ctx.beginPath();
    ctx.moveTo((bboxA[0] + bboxA[2]) / 2, (bboxA[1] + bboxA[3]) / 2);
    ctx.lineTo((bboxB[0] + bboxB[2]) / 2, (bboxB[1] + bboxB[3]) / 2);
    ctx.stroke();
    ctx.restore();
}

function drawLabel(x, y, text, color, size) {
    ctx.save();
    let fs = Math.max(8, size) / zoom;
    ctx.font = `600 ${fs}px "Segoe UI","Microsoft YaHei",sans-serif`;
    // Text shadow for readability
    ctx.shadowColor = 'rgba(0,0,0,0.85)';
    ctx.shadowBlur = 3 / zoom;
    ctx.fillStyle = color;
    ctx.fillText(text, x, y - 3 / zoom);
    ctx.restore();
}

// ── Main draw ──
function draw() {
    if (!currentImage) return;
    let { img, info } = currentImage;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.save();
    ctx.translate(panX, panY);
    ctx.scale(zoom, zoom);

    // 1. Base image
    ctx.drawImage(img, 0, 0);

    // 2. Lines (drawn under boxes)
    if (layers.showLines) {
        for (let p of info.pairs) {
            if (p.type === 'matched' && !layers.showMatched) continue;
            if (p.type === 'coverage' && !layers.showCoverage) continue;
            if (!layers.showPred && !layers.showGT) continue;

            let lc;
            if (p.type === 'matched') {
                lc = p.match_method === 'iou' ? C.lineIOU : C.lineContain;
            } else {
                lc = C.lineCover;
            }
            drawLine(p.pred_bbox, p.gt_bbox, lc, 1.2);
        }
    }

    // Helper: is a box selected?
    function isSelected(p) {
        if (!selectedBox) return false;
        if (p.type !== selectedBox.type) return false;
        if (p.type === 'matched' || p.type === 'coverage')
            return p.pred_idx === selectedBox.pred_idx && p.gt_idx === selectedBox.gt_idx;
        if (p.type === 'unmatched_pred') return p.idx === selectedBox.idx;
        if (p.type === 'unmatched_gt') return p.idx === selectedBox.idx;
        return false;
    }

    // 3. GT boxes
    if (layers.showGT) {
        // Matched GT
        if (layers.showMatched) {
            for (let p of info.pairs) {
                if (p.type !== 'matched') continue;
                let sel = isSelected(p);
                drawBox(p.gt_bbox, C.gtFill, sel ? C.selColor : C.gtStroke, sel ? 3.5 : 2, []);
                if (sel) drawCorners(p.gt_bbox, C.selColor, 12);
            }
        }
        // Coverage GT
        if (layers.showCoverage) {
            for (let p of info.pairs) {
                if (p.type !== 'coverage') continue;
                let sel = isSelected(p);
                drawBox(p.gt_bbox, C.gtFillCov, sel ? C.selColor : C.gtStrokeCov, sel ? 3.5 : 2, [4, 3]);
                if (sel) drawCorners(p.gt_bbox, C.selColor, 12);
            }
        }
        // Unmatched GT
        if (layers.showUnmatchedGT) {
            for (let ug of (info.unmatched_gt || [])) {
                let sel = isSelected(ug);
                drawBox(ug.bbox, C.gtFillUnm, sel ? C.selColor : C.gtStrokeUnm, sel ? 3.5 : 1.5, [3, 3]);
                if (sel) drawCorners(ug.bbox, C.selColor, 12);
            }
        }
    }

    // 4. Pred boxes
    if (layers.showPred) {
        // Matched pred
        if (layers.showMatched) {
            for (let p of info.pairs) {
                if (p.type !== 'matched') continue;
                let sel = isSelected(p);
                drawBox(p.pred_bbox, C.predFill, sel ? C.selColor : C.predStroke, sel ? 3.5 : 2, []);
                if (sel) drawCorners(p.pred_bbox, C.selColor, 12);
            }
        }
        // Coverage pred
        if (layers.showCoverage) {
            for (let p of info.pairs) {
                if (p.type !== 'coverage') continue;
                let sel = isSelected(p);
                drawBox(p.pred_bbox, C.predFillCov, sel ? C.selColor : C.predStrokeCov, sel ? 3.5 : 2, [4, 3]);
                if (sel) drawCorners(p.pred_bbox, C.selColor, 12);
            }
        }
        // Unmatched pred
        if (layers.showUnmatchedPred) {
            for (let up of (info.unmatched_pred || [])) {
                let sel = isSelected(up);
                drawBox(up.bbox, C.predFillUnm, sel ? C.selColor : C.predStrokeUnm, sel ? 3.5 : 1.5, [3, 3]);
                if (sel) drawCorners(up.bbox, C.selColor, 12);
            }
        }
    }

    // 5. Labels (on top of everything)
    if (layers.showLabels) {
        // Matched labels
        if (layers.showMatched && layers.showPred) {
            for (let p of info.pairs) {
                if (p.type !== 'matched') continue;
                let sel = isSelected(p);
                let parts = [];
                if (p.gt_label === p.pred_label) {
                    parts.push(p.gt_label);
                } else {
                    parts.push(`Pred:${p.pred_label}→GT:${p.gt_label}`);
                }
                if (p.rec_score !== undefined && p.rec_score !== null && p.gt_label === p.pred_label) {
                    let rs = typeof p.rec_score === 'object' ? p.rec_score.combined : p.rec_score;
                    let errStr = rs === 0 ? '✓' : rs.toFixed(2);
                    parts.push(`e=${errStr}`);
                }
                drawLabel(p.pred_bbox[0], p.pred_bbox[1], parts.join(' '),
                    sel ? '#fff' : C.lblMatch, 10);
            }
        }
        // Coverage labels
        if (layers.showCoverage && layers.showGT) {
            for (let p of info.pairs) {
                if (p.type !== 'coverage') continue;
                let sel = isSelected(p);
                let parts = [`cov=${p.coverage?.toFixed(2)}`];
                if (p.rec_score !== undefined && p.rec_score !== null) {
                    parts.push(`e=${p.rec_score.toFixed(2)}`);
                }
                drawLabel(p.gt_bbox[0], p.gt_bbox[1], parts.join(' '),
                    sel ? '#fff' : C.lblCover, 9);
            }
        }
        // Unmatched labels
        if (layers.showUnmatchedPred && layers.showPred) {
            for (let up of (info.unmatched_pred || [])) {
                let sel = isSelected(up);
                let labelColor = up.label === 'picture' ? '#ffa198' : C.lblUnmatch;
                drawLabel(up.bbox[0], up.bbox[1], `✗ ${up.label || '?'}`,
                    sel ? '#fff' : labelColor, 10);
            }
        }
        if (layers.showUnmatchedGT && layers.showGT) {
            for (let ug of (info.unmatched_gt || [])) {
                let sel = isSelected(ug);
                drawLabel(ug.bbox[0], ug.bbox[1], `✗ GT ${ug.label || '?'}`,
                    sel ? '#fff' : '#f0c060', 10);
            }
        }
    }

    // 6. Selected glow
    if (selectedBox) {
        let bb;
        if (selectedBox.type === 'matched' || selectedBox.type === 'coverage') {
            bb = selectedBox.hitOn === 'pred' ? selectedBox.pred_bbox : selectedBox.gt_bbox;
        } else {
            bb = selectedBox.bbox;
        }
        if (bb) {
            // Faint outer glow
            ctx.save();
            ctx.strokeStyle = C.selGlow;
            ctx.lineWidth = 8 / zoom;
            ctx.setLineDash([]);
            ctx.strokeRect(bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1]);
            ctx.restore();
        }
    }

    ctx.restore();

    // Status bar
    let z = Math.round(zoom * 100);
    let selInfo = selectedBox ? ` | 已选中: ${selectedBox.type}` : '';
    statusBar.textContent = `缩放: ${z}% | 滚轮缩放 | 拖拽平移 | 点击框选 | Esc 取消选择${selInfo}`;
}

init();
</script>
</body>
</html>"""


def load_image_as_datauri(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".tif": "image/tiff", ".tiff": "image/tiff", ".bmp": "image/bmp"}
    mime = mime_map.get(ext, "image/png")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_image_data(debug_results: dict, image_dir: Path) -> dict:
    images = {}
    per_image = debug_results.get("per_image", {})

    for image_id, img_result in per_image.items():
        if "error" in img_result:
            continue
        debug = img_result.get("debug")
        if not debug:
            continue

        image_name = image_id + ".png"
        image_path = image_dir / image_name
        if not image_path.exists():
            for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                candidate = image_dir / (image_id + ext)
                if candidate.exists():
                    image_path = candidate
                    break

        if not image_path.exists():
            print(f"Warning: image not found for {image_id}, tried {image_path}")
            continue

        data_uri = load_image_as_datauri(image_path)

        matched_pairs = [p for p in debug["pairs"] if p["type"] == "matched"]
        coverage_pairs = [p for p in debug["pairs"] if p["type"] == "coverage"]

        class_acc = img_result.get("classification", {}).get("accuracy", 0)
        composite = img_result.get("composite", 0)

        detection = {}
        for label, d in img_result.get("detection", {}).items():
            if label == "overall":
                continue
            detection[label] = {
                "f1": d.get("f1", 0),
                "precision": d.get("precision", 0),
                "recall": d.get("recall", 0),
            }
        detection["overall"] = img_result.get("detection", {}).get("overall",
                                    {"f1": 0, "precision": 0, "recall": 0})

        recognition = {}
        for label, r in img_result.get("recognition", {}).items():
            recognition[label] = {
                "mean_error": r.get("mean", 0),
                "num_pairs": r.get("num_pairs", 0),
            }

        images[image_id] = {
            "name": image_id,
            "dataUri": data_uri,
            "pairs": debug["pairs"],
            "unmatched_pred": debug["unmatched_pred"],
            "unmatched_gt": debug["unmatched_gt"],
            "summary": {
                "num_pred": debug["num_pred"],
                "num_gt": debug["num_gt"],
                "matched_pairs": len(matched_pairs),
                "coverage_pairs": len(coverage_pairs),
                "composite": composite,
                "class_acc": class_acc,
                "detection": detection,
                "recognition": recognition,
            },
        }

    return images


def main():
    parser = argparse.ArgumentParser(description="Generate interactive benchmark visualization")
    parser.add_argument("--debug", default=None,
                       help="Path to debug results JSON (from run_benchmark.py --debug)")
    parser.add_argument("--image-dir", default="data/image",
                       help="Directory containing document images")
    parser.add_argument("--output", default="benchmark_viz.html",
                       help="Output HTML file path")
    parser.add_argument("--benchmark-json", default="benchmark_results.json",
                       help="Path to standard benchmark_results.json (used if no --debug)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    image_dir = script_dir / args.image_dir

    if args.debug:
        debug_path = Path(args.debug)
        if not debug_path.is_absolute():
            debug_path = script_dir / args.debug
    else:
        debug_path = script_dir / "debug_results.json"
        if not debug_path.exists():
            debug_path = script_dir / args.benchmark_json

    if not debug_path.exists():
        print(f"Error: debug results not found at {debug_path}")
        print("Run: python run_benchmark.py --model-dir data/prediction/paddleV1.5 "
              "--model paddle_ocr_vl --debug --output debug_results.json")
        sys.exit(1)

    print(f"Loading debug results from {debug_path}...")
    with open(debug_path, "r", encoding="utf-8") as f:
        debug_results = json.load(f)

    if not image_dir.exists():
        print(f"Error: image directory not found: {image_dir}")
        sys.exit(1)

    print("Building visualization data...")
    images = build_image_data(debug_results, image_dir)

    if not images:
        print("Error: no valid images with debug data found")
        sys.exit(1)

    print(f"  Loaded {len(images)} image(s) with debug overlays")

    data_json = json.dumps({"images": images}, ensure_ascii=False)
    data_json = data_json.replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)

    output_path = script_dir / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nVisualization saved to {output_path}")
    print(f"Open it in your browser: file:///{output_path.as_posix()}")


if __name__ == "__main__":
    main()
