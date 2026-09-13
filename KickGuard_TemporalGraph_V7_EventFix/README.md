# KickGuard Temporal-Graph V7 EventFix

本版本针对训练数据时间误解析问题重写时间轴：

1. 训练文件优先解析文件名中的 `YYYYMMDDTHHMMSS~YYYYMMDDTHHMMSS` 时间范围。
2. 只允许前 4 列作为显式时间候选，绝不把深度/流量等传感器列当时间。
3. 如果前 4 列没有可靠绝对时间，则使用文件名时间范围按行位置建立单调时间轴。
4. 训练井只保留事件前 9 小时至事件后 2 小时，避免错误远期日期污染。
5. 自动兼容 `WELL_0000010 -> WELL_000010`。
6. 不依赖训练井内部子目录名称。
7. 测试集要求正好 324 个 `test_001 ~ test_324` 文件。

## 目录

```text
超深油气井钻井过程溢流实时预警/
├── train/
├── test/
└── base/
    └── KickGuard_TemporalGraph_V7_EventFix/
        ├── kickguard_v7.py
        ├── requirements.txt
        └── run.sh
```

如果你把代码文件直接复制到 `base/`，默认 `../train`、`../test` 同样可用。

## 安装

```bash
pip install -r requirements.txt
```

## 先校验时间

一定先删除旧错误缓存：

```bash
rm -rf work_v7
python kickguard_v7.py \
  --train-dir ../train \
  --test-dir ../test \
  --work-dir ./work_v7 \
  --scan-only
```

正常日志应类似：

```text
[check] WELL_000001: ... event=2025-01-17 07:30:00, nearest=几秒
```

不能再出现 2031/2034/2045 等明显错误年份。

## 正式训练

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
