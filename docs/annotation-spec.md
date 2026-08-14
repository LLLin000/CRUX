# CRUX 标注规范 v1（D0 交付）

对应 PLAN v1.2.1 §7 D0：**两套标注**，严格分离。

## 1. 训练标注（Train Annotation）—— 用于训练 RF-DETR-Seg

### 1.1 格式

COCO JSON 格式（RF-DETR 训练直接消费）：

```json
{
  "images": [{"id": 0, "file_name": "0000.jpg", "width": 4500, "height": 3000}],
  "categories": [{"id": 0, "name": "hold"}, {"id": 1, "name": "volume"}],
  "annotations": [
    {"id": 0, "image_id": 0, "category_id": 0,
     "segmentation": [[x1,y1, x2,y2, ...]],   // 单个外轮廓 polygon，闭合无需重复首点
     "area": 12345.6, "bbox": [x, y, w, h],
     "iscrowd": 0}
  ]
}
```

### 1.2 类别定义（语义必须单一）

| 类别 | 定义 | 示例 |
|---|---|---|
| `hold` | 任何可抓握/踩踏的**独立岩点**：常规造型点、不规则胶合板点、小体积点 | 蓝/红/黄点、异形点 |
| `volume` | **大型结构块**：木质/合成大体积、三角块、带螺栓孔的箱体。若 volume 上又装了 hold，两者**分别标注** | 大蓝色三角、木箱 |

**边界规则**：
- 视觉上连成一片但属于不同路线的同色点，仍各自一个 instance（训练只看形状，不管路线）
- 被部分遮挡的岩点：标注可见部分即可（训练用），benchmark 标注可补充
- 两个岩点重叠/接触：**分别标注**，polygon 各自闭合（允许 overlap）
- 墙边缘被裁掉的岩点：标注到图像边界
- 面积 < 图像 0.01% 的岩点不标（噪声）

### 1.3 polygon 规则

- 单个外轮廓；**不需要内孔**（hold 无孔洞语义）
- 顶点数：8-200；顶点点距尽量均匀（不要求逐像素）
- 顶点顺序：任意方向（COCO 无方向要求）
- 贴合轮廓 ±2px 内；凹多边形必须如实勾出凹陷（如 crimp 的缺口）

### 1.4 数据规范

- 原始图像不裁剪、不 resize 直接标注（RF-DETR 训练时自行 resize）
- 标完必须过一致性检查：`area > 0`、顶点数合法、bbox 与 polygon 一致（转换脚本自检）

## 2. Benchmark 标注（RouteBenchmark）—— 用于产品 KPI

### 2.1 目的

KPI（修正数、candidate recall、route recoverability）需要**路线级 GT**，训练标注算不出来。RouteBenchmark 是独立的小集（100-200 条路线、跨多岩馆），**只标注，不训练**。

### 2.2 格式

在训练标注（instance 级）之上**追加路线层**：

```json
{
  "instances": [{"instance_id": 0, "image_id": 0, "category_id": 0,
                 "segmentation": [...], "bbox": [...], "area": ...}],
  "routes": [
    {"route_id": "R-001", "image_id": 0, "route_color": "blue",
     "instances": [0, 3, 5, 7],            // 该路线包含的 instance_id 列表
     "is_start": [0, 3],                    // 起步岩点（可多个：双手/手+脚）
     "is_finish": [7]}                      // 终点（match 也算）
  ]
}
```

### 2.3 标注规则

- **route_id**：每条路线唯一；**同色不同路线必须拆开**（蓝色 V3 与蓝色 V5 是两条）
- **route_color**：以岩馆实际路线颜色为准（目测标签，不量化 Lab——Lab 是算法输出，GT 只要语义色）
- **is_start / is_finish**：显式标记；**允许缺失**（记录员不确定时留空，算法侧 route spine 纯视觉）
- 一条路线的岩点可以跨训练标注的多个 instance（含 volume 上装的 hold）
- 图片内**未属于任何路线**的岩点：instance 照常标注，不进 routes 数组

### 2.4 集合规模（初始目标，数字不死板）

- 100-200 条真实路线（≈ 20-40 面墙照片，跨 ≥ 3 家岩馆）
- 验证集**按岩馆隔离**（不随机切分）
- 统计登记：gym / wall / lighting（日光/顶光/暗）/ device / color 分布

### 2.5 双层评测（selector-only / end-to-end）

| 评测 | 输入 | 回答的问题 |
|---|---|---|
| selector-only | GT masks → Lab → SeededRouteSelector | ΔE + 空间分组本身好不好 |
| end-to-end | photo → RF-DETR → Lab → SeededRouteSelector | 用户实际体验 |

D3 迭代时据此定位：candidate recall 低是"补数据训模型"还是"调颜色/DBSCAN"。

## 3. 标注工具链

- 训练标注：Label Studio / VIA（Web，导出 COCO）
- benchmark 标注：Label Studio 自定义 label config（route_id 字段），或 CSV+验证脚本
- 转换：`tools/kaggle_to_coco.py`（Kaggle VIA → COCO）；自有数据直接标 COCO
- 预标注：climbnet 权重（Apache-2.0）先出 mask，人工修正——只用于训练标注

## 4. 许可红线（PLAN §2 规则 2）

- Kaggle 数据（CC BY-SA，未核实）→ **research only**，只进 D0/D0.5/方法验证
- production weights 只用**自有实拍**标注数据
- 转换脚本输出目录分 `research/` 与 `production/`，物理隔离
