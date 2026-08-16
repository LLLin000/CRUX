# CRUX iOS 方案 v1.2.1

> 攀岩记录 + 路线可视化 App（iOS 原生，SwiftUI）。定位：拍照 → 识别岩点 → 点选路线 → 发光渲染。
> v1.2.1（合同定稿）：修正 RLE raster space 矛盾、真漏检 fallback、START/FINISH 输入、sRGB canonicalization、双层 benchmark；gym 改为地图 API 选地址（不维护 gym 列表，移除 Gym 实体）。**此后不再迭代方案，进入 M0 + D0 + D0.5。**

## 0. 产品定义

**MVP 包含**：拍照/相册 → 岩点实例分割（hold/volume）→ 每点 Lab 取色 → 用户点一个岩点（seed）→ 同色候选 → 空间分组 → 人工 ± 修正（含"补岩点"）→ 压暗背景 + 发光 mask + route spine 渲染 → 记录保存（等级/结果/次数/用时/备注）→ 今日/图鉴浏览。

**产品核心行为（永远只有三个）**：看记录 → 加记录 → 看路线。设置是二级功能；地图浏览页 post-MVP。

**gym 策略（v1.2.1 定）**：**不维护 gym 列表**。加记录时接入地图 API（Apple `MKLocalSearch`），用户输入店名/搜索"攀岩"，选中的 POI 存为快照（名称 + 坐标 + mapItem 标识）。只要店在地图里有就能选。岩馆浏览页从历史记录聚合（distinct gym 快照）。

**MVP 明确不含**：难度自动估计、攀爬顺序 AI、多用户/社区/云端同步、路线自动完整分组（只做"seed 点选后按颜色+空间筛选候选"）。

**MVP 的核心价值主张**：AI/算法只负责把人工操作从"逐个点 10 个岩点"降低到"改 1-2 个"，**不追求 100% 自动**。**第一 KPI 即此价值主张的直接度量**（§7 D2）。

**分割策略（v1.2.1 定）：优先 recall，不优先 precision**——宁可多 3 个假阳性（用户可删），不要漏 2 个真岩点（用户无法恢复）。配套"补岩点"fallback（§6）。

**连线语义**：自下而上连线只是**视觉引导**（route spine / visual guide），程序不知道 beta。代码与 UI 一律不得叫"攀爬路线顺序/climbing sequence"。渲染支持 **Glow only / Glow+line** 两档，用户可关。

**视觉基线**：`climb_crux_minimal_v4.html` prototype（深色主题 `#0B0C0E` + 荧光绿 accent `#C9FF45` + 路线色体系）。

## 1. 技术决策

| # | 决策 | 选择 | 理由 |
|---|---|---|---|
| D1 | 语言/框架 | SwiftUI + SwiftData，iOS 17+，iPhone only | 单用户记录类 App 的 2026 标准组合 |
| D2 | 导航 | **底部 2 主入口：`今日 \| 图鉴` + 独立 `＋`**。设置从今日左上角进；加记录时 gym 用地图选。状态驱动导航（sheet/跳转是 VM 里的 enum，syncups 模式） | 产品气质：核心行为只有看/加/看；相机用 fullScreenCover |
| D3 | 模型运行时 | **首选候选：RF-DETR-Seg**（roboflow，Apache-2.0）→ **ONNX 导出 + iOS 端 ONNX Runtime（CoreML EP 优先，CPU 兜底）**（v1.3 修正：无 Mac 环境，Core ML .mlpackage 导出需 macOS，改为 ONNX 全跨平台路线）→ 自写 Swift decode/mask 重建。**D0.5 Gate A/B/C 通过后冻结，否则切 RTMDet-Ins** | ONNX 导出纯跨平台（torch.onnx，Windows 验证可行）；onnxruntime-swift 是微软官方 iOS 推理库；DINOv2 transformer 在 ANE 上的落位仍是最大未知（CoreML EP 不友好则 CPU/GPU） |
| D4 | 分割类别 | 只训 `hold` + `volume`，**颜色不进模型** | 泛化性最好；颜色是后处理，架构解耦 |
| D5 | 颜色 | mask 内像素 Lab median（**算法事实**）；seed 点选后 CIEDE2000（ΔE00）匹配。**阈值是 initial heuristic（~5-10），不是 invariant**，D0.5/D1 baseline 校准。**所有 Lab 一律从 canonical sRGB 图像计算**（v1.2.1） | Lab 抗光照优于 RGB；ΔE00 优于 CIE76 欧氏；统一色彩空间消除 P3/sRGB 偏差 |
| D6 | 空间拆分 | **seeded 流程**：seed → ΔE 同色候选 → 空间算法生成若干 group → 默认选"包含 seed 的 group" → 用户 ± 修正。DBSCAN(eps≈2×岩点直径, min_samples=2) 是 initial heuristic；**参数不得当产品 invariant** | DBSCAN min_samples=1 ≡ 连通分量（数学事实）；失败时用户手动修正兜底 |
| D7 | 架构解耦 | `HoldSegmenter` 协议隔离模型层（`RFDETRSegmenter` → future fallback） | 降低换模型成本；**协议解耦不隔离许可证义务**，合规靠 §2 |
| D8 | 存储 | SwiftData + 照片存 `Documents/Photos/<UUID>.jpg`（路径入库，不存 blob）；schema 首日版本化迁移 | SwiftData 对单用户中型模型已生产级 |
| D9 | 相册/相机/地图 | PhotosPicker（免权限）+ UIImagePickerController 包装（相机）+ **MKLocalSearch 选 gym POI**（v1.2.1） | AVFoundation 自定义相机仅在需要实时检测 UI 时引入 |
| D10 | 数据起点 | Kaggle 12GB 数据集（VIA polygon 标注）→ 转 COCO 格式；climbnet 权重（Apache）预标注 | 不用从零标 300-500 张。research/production 分离见 §2 |

**推理规格（全链统一，v1.3 修正）**：`RF-DETR-Seg → ONNX → iOS onnxruntime（CoreML EP / CPU）`。
- **FP16/INT8**：ONNX 导出的量化选项（rfdetr `export(format="onnx")` 支持 fp16/int8 优化），Gate C 实测决定；不再依赖 coremltools。
- **分辨率不锁 640px**：官方 Nano/S/M 标准 312/384/432；实际取 384/432/512，由 D0.5 Gate C 真机 benchmark 决定。

**构建与发布（v1.3 更新，无本地 Mac）**：
- **当前阶段（发布推迟，功能/UI 测试优先）**：Codemagic（免费 500 分钟/月 macOS M2）跑 **Xcode 编译 + iOS 模拟器测试**（UI 测试/单元测试不需要 developer 账号，不需要签名）。
- **发布阶段（用户购买 Apple Developer Program $99/年 后）**：Codemagic 自动签名 → TestFlight → 真机安装。全程无需本地 Mac。
- 唯一硬门槛：Apple Developer Program（账号非 Mac），购买前不做任何发布工作。

## 2. 许可证与数据策略（自始干净，无 AGPL）

**决策**：只用 permissive（Apache-2.0/MIT/BSD）训练框架与权重，**不引入 Ultralytics**（官方仍为 AGPL-3.0 / Enterprise 双许可）。许可合规不靠架构，靠两条硬规则：

**规则 1（框架）**：训练/导出/推理全链只用 permissive 组件。

| 框架 | 许可 | 实例分割性能（COCO mask AP） | iOS 导出 | verdict |
|---|---|---|---|---|
| **RF-DETR-Seg**（roboflow） | Apache-2.0（代码+权重） | Nano 40.3 / S 43.1 / M 45.3 | ONNX 导出 + iOS ONNX Runtime；CoreML EP optional | ⭐ 首选 |
| **RTMDet-Ins**（mmdetection） | Apache-2.0 | tiny 35.4 / s 38.7 / m 42.1 | ONNX/PyTorch adapter 需独立 decode | 备选（D0.5 失败时） |
| SparseInst（mmdet projects/） | Apache-2.0 | R-50 33.6 / vd-DCN 37.4 | 无 NMS 全卷积，导出需原型 | 备选 |
| Mask-RT-DETR-S（PaddleDetection） | Apache-2.0 | 41.0（box 46.1） | coremltools v8 已移除 ONNX 前端，导出链断 | 排除 |
| YOLOv6-Seg / PaddleYOLO | GPL-3.0 | 44.8 / — | GPL 分发受限 | 排除 |
| Ultralytics 全家 | AGPL-3.0 | 40-47 | SDK 成熟但 AGPL | 排除 |

**规则 2（数据）——research weights / production weights 严格分离**：
- **research weights**：Kaggle（CC BY-SA，论文另有 NC-SA 记载）只用于 D0/D0.5 技术验证，**绝不随 App 分发**。**Kaggle 数据仅在其实际许可允许的范围内用于 research；许可未核实前不得进入 production pipeline**。
- **production weights**：任何随 App 发布的 checkpoint，**只用自有实拍数据或明确授权数据训练**。
- 预标注工具：climbnet 权重（Apache-2.0）可用。

## 3. 系统架构

```
SwiftUI screens
        ↓ 只传 command / DTO
Feature coordinators（只编排，不拥有领域数据）
        ├─ Today / Calendar
        ├─ Route Archive / Gym / Map
        ├─ Add Record
        ├─ Route Detail
        └─ Settings / Profile
        ↓
On-device domain modules（互不越权）
        ├─ Capture      → CanonicalImage
        ├─ Analysis     → RouteAnalysis / RouteDraft
        ├─ RouteStore   → ClimbRoute / Hold / RouteUnionMask
        ├─ RouteQuery   → screen projections（只读）
        └─ Renderer     → preview image（纯函数）
```

### 3.1 Prototype 分区与后端模块归属

| Prototype 区域 | 后端模块 | 唯一职责 | 明确不负责 |
|---|---|---|---|
| 今天 / 日历 | `TodayQuery` | 从已保存路线生成月历、次数、岩馆数量摘要 | 不推理、不写路线、不维护 Gym |
| 路线图鉴 / 筛选 | `RouteQuery` | 返回路线卡片、筛选、排序 | 不改模型结果、不保存照片 |
| 地图 / 岩馆详情 | `GymQuery` | 按 `gymNameSnapshot + 经纬度` 聚合路线 | 不创建 Gym entity、不拥有路线 |
| 添加路线照片 | `Capture` | 相机/相册输入、EXIF 修正、sRGB、canonical JPEG | 不解码模型、不写 SwiftData |
| 添加路线表单 | `RouteMetadata` | 日期、岩馆 snapshot、等级、结果、尝试、用时、备注 | 不负责分割、不直接操作模型 session |
| 路线识别 / 高亮 | `Analysis` | segmentation、Lab、seed selection、manual correction | 不依赖 SwiftData、不知道 SwiftUI |
| 路线详情 / 渲染 | `Renderer` | 读取 `RouteSnapshot`，生成 Glow / Glow+line 预览 | 不持久化、不重新推理 |
| 设置 / 个人档案 | `Preferences` | 用户偏好、展示选项、默认值 | 不修改历史路线、不参与识别算法 |
| 保存 / 编辑 / 删除 | `RouteStore` | 唯一允许写 `ClimbRoute`、`Hold`、`RouteUnionMask` 的模块 | 不包含 ONNX、Lab、布局或导航逻辑 |

`AddRecordFlow` 是 coordinator：只按顺序调用 `Capture → Analysis → RouteMetadata → RouteStore`，
不把这四块重新实现一遍。`RouteDetailView` 默认只读；点击编辑时重新进入
`Analysis` 的 draft，不在 View 内直接修改 SwiftData。

### 3.2 M2 后端接口设计（冻结）

**模块位置**

- `CRUXCore` 只放 DTO、协议、颜色/路线纯逻辑；不得依赖 SwiftUI、SwiftData、UIKit 或 ONNX Runtime。
- `CRUXClient` 提供 `ONNXHoldSegmenter` adapter；模型加载、tensor 名称、输出解码和 execution provider 只存在这里。
- UI 只依赖 `RouteAnalysis`、`RouteDraft` 和 `RouteSnapshot`，不读取 ONNX 输出。

**外部接口**

```swift
public protocol HoldSegmenter: Sendable {
    var modelVersion: String { get }
    var inputSize: Int { get }
    func segment(_ image: CanonicalImage) async throws -> SegmentationResult
}

public struct SegmentationResult: Sendable {
    public let modelVersion: String
    public let inputSize: Int
    public let detections: [DetectedHold]
}

public struct DetectedHold: Sendable, Identifiable {
    public let id: Int                 // stable only within one analysis
    public let kind: HoldKind          // hold | volume
    public let confidence: Float
    public let geometry: HoldGeometry
}
```

**坐标与数据不变量**

1. 输入已经 `fixOrientation`、转 sRGB，并保存为 canonical JPEG；宽高随请求传入。
2. 模型内部可 stretch/resize 到方形输入，但输出必须 inverse-transform 回原图 canonical normalized `[0,1]`。
3. `HoldGeometry` 的 bbox 是 canonical normalized；mask 是 bbox-local COCO RLE，宽高显式保存，RLE 从零段开始、`Int32 little-endian`。
4. decoder 在返回前把 mask clip 到自己的 bbox；UI 不再做 mask 修补。
5. Lab 只从 canonical sRGB 原图 mask 内取样；模型不预测颜色。
6. `DetectedHold.id` 只在一次 `SegmentationResult` 内稳定；保存为 SwiftData `Hold` 后由模型对象关系负责持久身份。

**错误、生命周期与测试缝**

- `ONNXHoldSegmenter` 加载一次并复用 session；禁止每张照片重新创建 session。
- 错误使用 typed failures：model unavailable、invalid image、invalid output、inference failed；不返回空结果伪装成功。
- `SegmentationResult` 不保存 logits、tensor 或 provider 细节，只保存后续分析和 provenance 所需结果。
- 测试通过 `FakeHoldSegmenter` 注入固定结果；Core 测试不链接 ONNX Runtime。

**唯一数据流**

`Capture → ONNXHoldSegmenter → HoldColorAnalyzer → SeededRouteSelector → RouteDraft
→ RouteStore`。
Today、Archive、Gym、Map、Renderer 都只能从 `RouteStore` 读取 projection；
它们不能反向调用 Analysis。

## 4. 坐标系与图片变换（architecture invariant）

**Canonical image space 是唯一真相坐标系**：

```
UIImage（含 EXIF orientation，可能 P3/sRGB）
  → fixOrientation
  → 颜色空间转换 → sRGB（v1.2.1 定：消除 P3/sRGB 偏差）
  → canonical upright sRGB image    ← 立即保存（Documents/Photos/<UUID>.jpg）
所有 Lab 一律从 canonical sRGB image 计算

模型推理只是 inference transform，不进数据库：
  canonical sRGB image
  → modelTransform（letterbox/resize 到 384/432/512）
  → prediction（model-input-space）
  → inverseTransform（含 letterbox padLeft/padTop 还原）
  → canonical normalized coordinates

SwiftUI 渲染：canonical sRGB image + scaledToFit；touch 坐标 → 逆变换回 canonical normalized
```

**三种坐标空间（v1.2.1 名词定稿）**：

| 空间 | 定义 | 用于 |
|---|---|---|
| **canonical normalized space** | 相对 canonical sRGB image 的 [0,1] 坐标 | `centroidX/Y`、`bboxX/Y/W/H`（数据库） |
| **bbox-local raster space** | bbox 内部矩形栅格（maskWidth × maskHeight 像素） | `maskRLE`（数据库） |
| **model-input space** | 模型输入张量坐标（含 letterbox） | 只存在于分割器内部，**绝不入库** |

**硬规则**：
1. 数据库/模型层接口**绝不保存 model-input-space 坐标**。
2. **maskRLE 不是归一化坐标**（v1.2.1 修正）：`bbox/centroid` 用 canonical normalized；`maskRLE` 用与 canonical bbox 对齐的 **bbox-local raster**，显式保存 `maskWidth/maskHeight`。
3. letterbox padding 在 inverseTransform 中正确还原（长边缩放 + 短边居中，记录 padLeft/padTop）。

## 5. 数据模型（SwiftData，schema v1 定稿）

**v1.2.1：移除 `Gym` 实体**——gym 信息以 POI 快照存于 ClimbRoute（地图 API 选择的产物，无级联/删除规则问题）。

```swift
@Model final class ClimbRoute {
    id, name,
    // gym = 地图 POI 快照（MKLocalSearch 选择，无 Gym 实体）
    gymNameSnapshot: String,
    gymLatitude: Double?, gymLongitude: Double?,
    gymMapItemID: String?,
    gradeSystem: GradeSystem, gradeValue: String,   // 如 VScale + "V4"
    referenceL: Double, referenceA: Double, referenceB: Double,  // Lab 算法事实（canonical sRGB 计算）
    paletteColor: PaletteColor,    // blue|red|yellow|green|purple|black|gray|orange，UI 标签
    result: Result,                // flash|top|project，用户直接选择，不推导
    attempts: Int, durationSeconds: Int,
    note, date, photoFilename,
    // model provenance
    segmenterModelVersion: String?, routeSelectorVersion: String?,
    inferenceInputSize: Int?,
    manualAdditions: Int, manualRemovals: Int
}

@Model final class Hold {
    centroidX: Double, centroidY: Double,        // canonical normalized [0,1]
    bboxX, bboxY, bboxWidth, bboxHeight: Double, // canonical normalized，原子字段
    maskWidth: Int, maskHeight: Int,             // bbox-local raster 尺寸（v1.2.1）
    maskRLE: Data,                               // bbox-local 二值 mask，COCO-style RLE → Codable → Data
    isStart: Bool, isFinish: Bool,               // 用户显式标记（长按菜单），可多个 START；不自动推导
    route → ClimbRoute
}

@Model final class RouteUnionMask {
    route → ClimbRoute, maskData,                // 可选缓存：详情页快速渲染
    rasterWidth: Int, rasterHeight: Int          // full-image raster（canonical 等比缩放，如宽 512）
}
```

**关键定稿**：
- **maskRLE = bbox-local raster**（v1.2.1 修正）：decode local mask → 映射到 normalized bbox → 映射到 canonical image → SwiftUI display。比 full-image RLE 省空间（12MP 图不逐 hold 存全图 mask）。
- **bbox 原子字段**，SwiftData 不存 tuple。
- **颜色两维分离**：`referenceL/A/B`（算法事实）+ `paletteColor`（UI 标签），不混成一个 RouteColor。
- **isStart/isFinish 是用户语义**（v1.2.1 定）：Manual Correction 页长按岩点弹出"设为 START / 设为 FINISH"；可多个 START（起步手点/脚点）；不设置则 route spine 纯视觉 bottom→top，**不自动把最低/最高点写成 isStart/isFinish**。
- mask 从 schema v1 就预留：M2 后重开历史记录无需重跑模型即可精确高亮；换模型不影响历史记录。
- result 直接存用户选择，单一事实来源，不搞 attempts==1 推导 Flash。

## 6. 核心流程

```
拍照/相册 → fixOrientation + 转 sRGB → canonical sRGB image（保存）
  → RF-DETR-Seg（ONNX + iOS ONNX Runtime，FP16/INT8，recall 优先阈值）
      [inference transform 仅在 adapter 内部，入库前 inverseTransform]
  → 每 hold: maskRLE(bbox-local) + bbox + centroid + Lab median
  → 用户点任一岩点（seed）
  → ΔE00 匹配全图同色候选（阈值 heuristic）
  → 空间分组生成若干 group → 默认选中"包含 seed 的 group"
  → Manual Correction：
      普通点击  添加 / 删除路线岩点（已检测的 hold）
      长按未识别位置  → "补一个岩点" → 简单圆形/椭圆 mask 框住（v1.2.1：真漏检 fallback）
      长按岩点   → 设为 START / 设为 FINISH
  → 确认 → 保存 route（holds + maskRLE + referenceLab + paletteColor + provenance + 修正计数）
  → 渲染：背景压暗 35% + 目标 mask 发光（blur+加亮）+ route spine（自下而上 Catmull-Rom）
  → 用户可切 Glow only / Glow+line
```

**漏检语义（v1.2.1 定，两种"漏"严格区分）**：
- **A. 已检测但未选中**：RouteSelector 没选 → 用户点一下即加入（现有交互）。
- **B. 真漏检**：RF-DETR 没检测到 → 页面无此 mask → 长按补岩点（圆形/椭圆框住）。MVP 不做自由画笔。
- 由此定义 **route recoverability KPI**：仅靠现有检测 + 简单人工修正能还原完整路线的比例（§7）。

## 7. 里程碑（双轨并行）

### Track A — App（SwiftUI）

| 里程碑 | 内容 | 验收 |
|---|---|---|
| **M0**（第 1 周） | Xcode 工程 + 设计 token + **底部 2 主入口（今日\|图鉴）+ 独立 ＋** + 二级导航（设置）+ SwiftData schema v1 | 2 主 tab 可切、Add flow 可开可关、二级导航完整返回、schema v1 编译通过（含 maskRLE/maskWidth/maskHeight/provenance 字段） |
| **M1**（第 2-3 周） | 加记录全流程（相机/相册→fixOrientation+sRGB→表单：gym 地图选 POI→保存）+ 路线详情**伪分割渲染**（演示照片手摆 maskRLE 写入 schema + 发光 + route spine）+ 日历/图鉴列表 | 完整走通"拍照→保存→图鉴→详情"；渲染效果对齐 prototype；点选对齐（canonical 规则）用真机照片验证 |
| **M2**（第 4-5 周） | 后端：`HoldSegmenter` + ONNX Runtime adapter + inverseTransform/decode；前端只消费 `RouteAnalysis` 完成 seed 点选、±修正、补岩点、START·FINISH；另加岩馆聚合页 | 真实照片可分割；UI 不感知 tensor/provider；修正计数落库；**端到端延迟达标（§7 D2）** |
| **M3**（第 6 周） | 打磨：空态、动画、性能、无障碍、真机回归 | 日常可用，TestFlight 打包 |

### Track B — 数据与模型

| 里程碑 | 内容 | 验收 |
|---|---|---|
| **D0**（第 1 周） | **两套标注规范**：① 训练标注 `instance_id, class=hold\|volume, polygon`（COCO 格式）；② benchmark 标注 `instance_id, route_id, route_color, is_start, is_finish`。Kaggle 数据转 COCO 脚本 | 规范文档 + 转换脚本跑通，样本可视化 |
| **D0.5**（第 1-2 周，关键 spike） | **RF-DETR-Seg 链路验证，只 gate runtime/export（v1.3：ONNX 路线）**：10-20 张小数据微调（research weights）→ **ONNX 导出（Windows 可做）** → iPhone 真机（onnxruntime） | **Gate A**：PyTorch→ONNX 完整导出 + 推理数值一致性（Windows 端 torch 对比）；**Gate B**：输出正确（mask shape/bbox 正确、无 crash、数值无异常）；**Gate C**：384/432/512 三档 p50/p95/peak memory（真机）。**只有 A/B/C runtime 问题才触发切 RTMDet**；hold recall 不足归 D1/D2 |
| **D1**（第 2-3 周） | 数据收集：**自有实拍（production）** + Kaggle（research，仅调参）+ climbnet 预标注 + 人工修正；**建立 RouteBenchmark**（100-200 条真实路线、跨多岩馆） | production：**≥100 images AND ≥1000 instances**（初始目标，不死板）；统计 gym/wall/lighting/device diversity；验证集**按岩馆隔离**；RouteBenchmark 完成 |
| **D2**（第 4 周） | 正式训练 RF-DETR-Seg-S/M（production weights）+ ONNX Runtime-compatible FP16/INT8 export + 真机基准 + RouteBenchmark 评测（双层，见下） | **产品 KPI 达标** |
| **D3**（第 5 周） | 错误案例分析 + 补标迭代 + 模型版本迭代 | 集成 M2；**用 provenance 数据验证新模型把平均修正数降下来** |

**D2 验收指标（产品 KPI，双层 benchmark）**。不写含糊的"mask mAP ≥ 0.8"（RF-DETR-Seg-S 官方 COCO AP50:95 仅 43.1）。阈值在 D0.5/D1 baseline 后冻结：

**Benchmark 双层拆解（v1.2.1 定）**——D3 才能定位"该补数据还是该调算法"：
- **selector-only**：GT hold masks → Lab → SeededRouteSelector → candidate recall/precision（评价 ΔE+空间分组本身）
- **end-to-end**：photo → RF-DETR → Lab → SeededRouteSelector → correction count（评价用户实际体验）

KPI 清单：
1. **第一 KPI：平均每条路线从 seed 到正确路线所需用户操作次数（修正数）**——直接度量"从点 10 个到改 1-2 个"，目标 ≤ 2
2. **route recoverability**（v1.2.1 新增）：仅靠现有检测 + 简单修正可还原完整路线的比例
3. candidate recall / precision（selector-only 与 end-to-end 各报一组）
4. hold recall / 小岩点漏检率（分割质量，训练标注评测）
5. manual additions / removals 分布（total correction count）

**延迟验收（端到端）**：`fixOrientation + sRGB + resize/letterbox + CoreML inference + decode + mask 重建 + Lab extraction` 全链，报告 **p50 / p95 / peak memory**。目标 p95 ≤ 1.5s，具体阈值 D0.5 Gate C 定。

**合流点**：第 4-5 周 M2 接入 D2 模型。

## 8. 风险与应对

| 风险 | 等级 | 应对 |
|---|---|---|
| **DINOv2 骨干在 iOS onnxruntime 上的性能/兼容未验证**（最大未知；CoreML EP 对 transformer 支持有限） | 高 | **D0.5 Gate A/B/C 先行**；CoreML EP 不友好则 CPU/GPU EP（单图场景可接受）；仍不过切 RTMDet-Ins |
| RF-DETR 自定义数据训练流程未验证 | 中 | D0.5 用 10-20 张小数据先跑通全链 |
| Kaggle 数据许可未厘清（NC-SA / BY-SA 冲突） | 中 | research/production 分离；Kaggle 仅在许可允许范围内 research |
| 老机型（A15/A16）seg 性能未知 | 中 | Gate C 起多机型基准；不达标降分辨率档（384 起）或换小模型 |
| 同色多路线误合并 / 空间参数切断路线 | 中 | seeded 流程 + 参数是 heuristic；用户手动修正兜底 |
| 真漏检导致路线无法还原 | 中 | recall 优先阈值 + **长按补岩点 fallback** + route recoverability KPI |
| 坐标/EXIF/色彩空间偏差导致点选偏移或 Lab 漂移 | 中 | canonical sRGB 硬规则（§4）；M1 伪分割阶段真机验证 |
| 标注一致性差 | 中 | climbnet 预标注 + 双人抽查 + 分馆验证集 |
| 路线渲染效果与 prototype 观感差 | 低 | M1 用伪 mask 先对齐视觉，再换真 mask |

## 9. 待定事项（不阻塞开工）

- [ ] 岩馆实拍照片的拍摄规范（角度/距离/光照）—— Track B D1 前定
- [ ] App 名称/图标（暂用 CRUX）
- [ ] Kaggle 数据集许可证核实（research 受限范围内可用；production 用自有数据）
- [ ] D0.5 后冻结：模型选型、输入分辨率、ΔE00 阈值、DBSCAN 参数
- [ ] 地图浏览页 post-MVP（选 POI 功能 MVP 有，浏览聚合页 M2）；难度估计明确不做

## 10. 参考仓库

- 模型：**roboflow/rf-detr**（RF-DETR-Seg，Apache-2.0；ONNX export + iOS ONNX Runtime）、open-mmlab/mmdetection（RTMDet-Ins/SparseInst 备选）、cydivision/climbnet（预标注权重，Apache）、xiaoxiae Kaggle 数据集（research only）、mkurc1/climbingcrux_model（同类验证）
- UI：luisarmada/climbfolio（日历/图鉴/会话流）、masonmill/climbinglog（SwiftUI）、IanTimchak/spraywalls（Board/Hold/Route 数据模型）、pointfreeco/syncups（状态驱动导航）、konotori/RecipeBox（SwiftData 蓝本）
- 算法：tillwf/climbing-holds-pathway-extractor（HSV 区间）、xiaoxiae std/（KMeans）、NeuralClimb（HLS mode 取色）
