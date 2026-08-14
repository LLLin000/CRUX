# D1 — 数据收集 + RouteBenchmark

- **Blocking edges**: D0
- **Runs on**: Windows（本机）+ 实拍（iPhone）
- **Scope**: production/research 数据严格分离（§2 规则 2）

## Acceptance
- [ ] production 自有实拍：**≥100 images AND ≥1000 instances**（初始目标，不死板）；统计 gym/wall/lighting/device diversity
- [ ] research 数据（Kaggle）：仅调参，许可未核实前不进 production pipeline
- [ ] climbnet 预标注 + 人工修正
- [ ] **RouteBenchmark**：100-200 条真实路线、跨多岩馆；验证集按岩馆隔离

## 状态
- **进行中**：7 张真实照片已到位，待评估/标注
- 拍摄规范（角度/距离/光照）D1 开工前定稿

## Blocking
- blocks: D2（正式训练 production weights）
