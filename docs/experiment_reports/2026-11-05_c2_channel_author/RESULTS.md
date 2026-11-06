# Author-engine C2门控配对实验

主表：FP32、converted VisDrone val548、COCO maxDets100。

| run | AP | APs | AP75 | ΔAP | ΔAPs | GPU中位ms |
|---|---:|---:|---:|---:|---:|---:|
| screen_c2_150e_s0 | 25.606 | 17.322 | 25.840 | +0.000 | +0.000 | 7.133 |
| screen_channel_150e_s0 | 25.954 | 17.469 | 26.563 | +0.348 | +0.147 | 7.146 |
| screen_scalar_150e_s0 | 26.224 | 17.758 | 26.608 | +0.618 | +0.436 | 7.140 |

筛选决定：

```json
{
  "passed": false,
  "checks": {
    "AP_gain_at_least_0_3pp": true,
    "APs_gain_at_least_0_3pp": false,
    "AP75_APm_APl_no_drop_beyond_0_5pp": false,
    "latency_increase_at_most_10percent": true
  },
  "channel_minus_scalar_pp": {
    "AP": -0.2696077029895392,
    "APs": -0.28889872220781854,
    "AP75": -0.04526762358449332,
    "APm": 0.009133688716580934,
    "APl": -2.616356015551645
  },
  "next": "screen complete; preserve negative results; no automatic 350e",
  "caveat": "Engineering screening, not statistical significance. Historical C2 AP28.722 is context only."
}
```

历史28.722 AP不是本轮唯一基线；本轮所有比较使用同批次新训练的配对C2。
所有负面结果均保留。150轮筛选不等于证明收敛。
