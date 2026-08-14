# M1 — 加记录全流程 + 伪分割渲染 + 日历/图鉴

- **Blocking edges**: M0
- **Scope**: 加记录全流程（相机/相册 → fixOrientation+sRGB → 表单：gym 地图选 POI → 保存）+ 路线详情伪分割渲染（演示照片手摆 maskRLE 写入 schema + 发光 + route spine）+ 日历/图鉴列表

## Acceptance
- [ ] 完整走通"拍照 → 保存 → 图鉴 → 详情"
- [ ] 渲染效果对齐 prototype（#0B0C0E 深色 + 发光 mask + route spine，Glow only / Glow+line 两档）
- [ ] 点选对齐（canonical 规则）用真机照片验证
- [ ] gym 用地图 API（MKLocalSearch）选 POI，存快照

## 状态
- **待办**（未开工，等 M0 完成后启动）

## Blocking
- blocks: M2（接真模型）
