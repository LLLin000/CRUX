# D0 — 标注规范 + Kaggle→COCO 转换

- **Blocking edges**: none
- **Runs on**: Windows（本机，Python 3.14 / RTX 5060 8GB）
- **Outputs**: `docs/annotation-spec.md`（两套标注规范）；`tools/kaggle_to_coco.py`（转换脚本，含自检）

## Acceptance
- [x] 训练标注规范定稿：instance_id, class=hold|volume, polygon（COCO 格式）
- [x] benchmark 标注规范定稿：instance_id, route_id, route_color, is_start, is_finish
- [x] Kaggle VIA JSON → COCO JSON 转换脚本跑通（真实数据样本），输出自检通过
- [x] RF-DETR 训练环境可安装（rfdetr pip 包 + GPU 可用性确认）

## 状态
- **完成**：`docs/annotation-spec.md`（两套规范）+ `tools/kaggle_to_coco.py`（含自检）就位；rfdetr 训练环境可用

## Blocking
- blocks: D0.5, D1
