# M0 — iOS 工程骨架

- **Blocking edges**: none
- **Requires**: macOS + Xcode 15.3+（Windows 不可执行；交付包见 `app/README.md` 与 `docs/m0-delivery.md`）
- **Target**: 底部 2 主入口（今日|图鉴）+ 独立 ＋；SwiftData schema v1；设计 token；状态驱动导航

## Acceptance
- [ ] schema v1 编译通过（ClimbRoute/Hold/RouteUnionMask，含 maskRLE/maskWidth/maskHeight/provenance 字段）
- [ ] 2 主 tab 可切；Add flow 可开可关；二级导航（设置）完整返回
- [ ] 设计 token（Theme）与 prototype 对齐（#0B0C0E / #C9FF45 / route colors）

## 依赖
- 无（M0 不依赖 D 轨）
