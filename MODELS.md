# CRUX 模型版本表

命名规则：`crux-hold-seg-v<major>.<minor>.<patch>-<导出分辨率>-<精度>.onnx`

- major：训练数据规模变化（v0 = Kaggle bh 12 图；v1 = 含 realpic 的 production 数据）
- minor：训练配置变化（分辨率/增强/模型架构）
- patch：同一配置的修复性重训

| 版本 | 训练数据 | 分辨率 | 增强 | val mAP_50_95 | ONNX 文件 | 状态 |
|---|---|---|---|---|---|---|
| v0.1.0 | Kaggle bh (12 train) | 384 | 无 | ~0.29 | ~~`output/onnx/...`~~ | baseline（**已清理**） |
| v0.2.0 | Kaggle bh (12 train) | 640 | 无 | ~0.29 | ~~`output/onnx_640/...`~~ | 细掩码验证（**已清理**） |
| **v0.3.0** | Kaggle bh (12 train) | 640 | AUG_CONSERVATIVE, 130ep | **0.276** | `output/onnx_640_aug/crux-hold-seg-v0.3.0-648-fp16.onnx` | **当前默认** |
| v0.3.0-fp32 | 同上 | 640 | 同上 | 同上 | `output/onnx_640_aug_fp32/crux-hold-seg-v0.3.0-648-fp32.onnx` | INT8 量化源 |

## 训练 checkpoint 对应

| 版本 | ckpt 目录 | research pth |
|---|---|---|
| v0.1.0/v0.2.0 | ~~`output/rfdetr_seg_small/`~~（已清理） | — |
| v0.3.0 | `output/rfdetr_seg_small_aug/` | `crux-hold-seg-research.pth` |

## 已知限制（v0.x 系列）

- 数据仅 12 张标注图 → 泛化有限：蓝色体积置信度低（0.17，双阈值缓解）、训练集外的体积形状（红色大块）漏检
- **提升路径 = 数据**（D1 标注 realpic → v1.0.0 production 权重），不是再训

## 推理脚本默认指向

所有 tools 脚本默认 `--model output/onnx_640_aug/crux-hold-seg-v0.3.0-648-fp16.onnx`
