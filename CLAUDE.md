# CRUX workspace conventions

## 工作纪律（最高优先级）
- **任何不清楚的东西先查官方文档**，不要无限制试错。来源优先级：
  1. 官方文档/官方示例（仓库内 docs/、cookbook、README，优先于网页转述）
  2. 官方源码（pip 安装包内源码 = 唯一权威的本地真相）
  3. 确认行为靠"读文档+读源码"，试错最多 1-2 次，仍不通就回到文档
- 第三方库的配置/API 一律以官方 cookbook 结构为准（例：RF-DETR 训练用 Python API `RFDETRSegSmall + SegmentationTrainConfig + build_trainer`，不用 CLI 扁平参数）
- 产出脚本必须带 `--selfcheck` 自检
- **iOS 版本纪律**：deployment target = iOS 17.0。任何 iOS 18+ 的 API（如 `MKMapItem.identifier`）必须包 `#available(iOS 18, *)` 检查并提供降级路径，否则低版本编译失败

## Structure
- `PLAN.md` — 方案 v1.2.1（冻结的 spec，单一事实来源）
- `climb_crux_minimal_v4.html` — 视觉 prototype（深色 #0B0C0E + accent #C9FF45）
- `docs/` — 规范文档（标注规范等）
- `tools/` — 训练/转换脚本（Python，Windows 可跑）
- `.scratch/crux/issues/` — 本地 ticket（ask-matt 流程），每个 ticket 声明 blocking edges
- `app/` — iOS 工程（Mac 侧，M0 起）

## License discipline（PLAN §2）
- 全链只用 permissive（Apache-2.0/MIT/BSD）框架，禁 Ultralytics（AGPL）
- Kaggle 数据 = research only，绝不进 production weights
- production weights 只用自有实拍数据

## 双轨
- Track A（App，Mac 需要 Xcode）：M0 → M1 → M2 → M3
- Track B（数据/模型，Windows 可推进）：D0 → D0.5 → D1 → D2 → D3
