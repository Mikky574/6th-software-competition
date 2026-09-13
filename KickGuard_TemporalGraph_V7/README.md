# KickGuard Temporal-Graph V7

用于“超深油气井钻井过程溢流实时预警”赛题，本地目录版完整实现。

## 目录结构

```text
超深油气井钻井过程溢流实时预警/
├── train/
│   ├── WELL_000001/
│   ├── ...
│   ├── WELL_000009/
│   └── WELL_0000010/   # 自动映射为 WELL_000010
├── test/
│   └── data/
│       ├── test_001.csv
│       ├── ...
│       └── test_324.csv
└── base/
    └── KickGuard_TemporalGraph_V7/
        ├── kickguard_v7.py
        ├── requirements.txt
        └── run.sh
```

程序只认 `../train` 下的一级井目录，不依赖内部类似 `WELL_000009时序数据` 的子目录名称，因此 WELL_000006 下内部目录写错名字无需修改。

## 安装

```bash
pip install -r requirements.txt
```

## 第一步：只校验训练时间

第一次必须先运行：

```bash
rm -rf work_v7
python kickguard_v7.py \
  --train-dir ../train \
  --test-dir ../test \
  --work-dir ./work_v7 \
  --scan-only
```

V7 使用事件感知时间解析。每口井只接受能在已知溢流事件附近产生大量连续样本的时间解释，并只保留事件前 9 小时至事件后 2 小时。因此不应再出现 `2022 -> 2045` 这种错误时间跨度。

## 正式全量训练

```bash
python kickguard_v7.py \
  --train-dir ../train \
  --test-dir ../test \
  --work-dir ./work_v7 \
  --output ./result_v7.zip
```

或：

```bash
bash run.sh
```

## 输出

```text
result_v7.zip
result_v7_precision.zip
result_v7_recall.zip
result_v7_threshold.zip

work_v7/V7_summary.json
work_v7/V7_oof.csv
work_v7/V7_test_debug.csv
work_v7/V7_graph_edges.csv
```

优先提交 `result_v7.zip`。

## 强制重建缓存

如果修改了时间解析、特征或训练窗口，请使用：

```bash
rm -rf work_v7
# 或
python kickguard_v7.py ... --force-rebuild
```

## V7 核心

- 训练严格使用溢流事件之前的样本，避免 post-kick leakage
- 真实秒级窗口：60/180/300/600/900/1800 秒
- 10 口井 Leave-One-Well-Out 验证
- ExtraTrees + HistGradientBoosting + Logistic + prototype 融合
- drilling / circulating / tripping / static 工况门控
- 测试切片 hidden-domain 聚类
- 首尾状态 Temporal Graph 连续性重建
- 物理事件风险向前序切片传播
- 最终自动校准 main / precision / recall / threshold 四套提交
