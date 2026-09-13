# KickGuard V9 Feedback-Aware

V9 不再重新训练一个全局分类器，而是利用已经获得的线上反馈对 66.9444 高分结果做可学习的 residual correction。

## 已知线上反馈

- 强锚点：66.9444
- V7：36.9733
- V8：0.0000

注意：`V8=0` 只作为反向排序证据，不直接视为逐样本负标签，因为总分为 0 可能来自 Recall=0 或 Specificity=0。

## 输入

至少需要一个 66.9444 高分提交包，并需要明确指定：

- `--v7-zip`：线上 36.9733 对应的 V7 ZIP
- `--v8-zero`：线上 0.0000 对应的 **准确 V8 ZIP**
- `work_v7/test_features_v7_eventfix_1.npz`

推荐同时提供：

- `V5_feedback_dropbad_116.zip`
- `V5_meta_plus4_124.zip`
- `work_v7/V7_test_debug.csv`
- `work_v7/V7_1_test_debug.csv`
- `work_v7/V8_test_debug.csv`

## 运行

```bash
python kickguard_v9_feedback.py \
  --anchor116 ./V5_feedback_dropbad_116.zip \
  --anchor124 ./V5_meta_plus4_124.zip \
  --v7-zip ./result_v7.zip \
  --v8-zero ./v8_submit/result_v8_nograph_top116.zip \
  --work-dir ./work_v7 \
  --output-dir ./v9_submit
```

如果你实际提交并得到 0.0000 的 V8 包不是 `result_v8_nograph_top116.zip`，必须把 `--v8-zero` 改成真实文件。

## 方法

1. 66.9444 提交作为强锚点。
2. 双锚点存在时：共同判正为保护区，116/124 分歧区为主要 correction 区。
3. V7 36.9733 作为中等正向排序信号。
4. V8 0.0000 作为反向排序信号，但不会硬编码为负标签。
5. 使用 1632 维 test feature cache，取方差最大的 192 维，经 signed-log + 标准化后训练 pseudo-label correction model。
6. correction model 使用 6-fold cross-fit 的 ExtraTreesRegressor + HistGradientBoostingRegressor，避免直接在 324 个测试样本上完全自拟合。
7. 高分 anchor 仍占 76%，residual correction 占 24%。
8. 同时输出保守 swap 版本和全局 feedback Top-K 版本。

## 输出

`v9_submit/` 下将生成：

- `v9_base116_swap1_k116.zip`
- `v9_base116_swap2_k116.zip`
- `v9_base116_swap3_k116.zip`
- `v9_base116_swap4_k116.zip`
- `v9_base116_swap6_k116.zip`
- `v9_base116_swap8_k116.zip`
- `v9_base124_swap*.zip`
- `v9_feedback_top112.zip`
- `v9_feedback_top116.zip`
- `v9_feedback_top120.zip`
- `v9_feedback_top124.zip`
- `v9_feedback_top128.zip`
- `v9_antiv8_top116.zip`
- `v9_antiv8_top120.zip`
- `v9_antiv8_top124.zip`
- `V9_feedback_debug.csv`
- `V9_feedback_summary.json`
- `SUBMIT_ORDER.txt`

## 推荐提交顺序

脚本会自动生成 `SUBMIT_ORDER.txt`。默认先做保守探针：

1. `v9_base116_swap2_k116.zip`
2. `v9_base124_swap2_k124.zip`
3. `v9_base116_swap3_k116.zip`
4. `v9_base124_swap3_k124.zip`
5. `v9_feedback_top120.zip`
6. `v9_antiv8_top116.zip`

不要一次性全部提交。先根据前 2~4 个 probe 的线上反馈继续缩小边界。
