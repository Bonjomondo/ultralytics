# 5090 C2 + O2M-C

训练期为原生C2 Encoder输出P3/P4/P5增加PP-YOLOE风格密集监督，原生RT-DETR Hungarian主损失不变。分支 `5090-c2-o2mc-20261031`。

[设计、状态及证据入口](../../docs/experiment_reports/2026-10-31_5090_c2_o2mc/README.md) · [官方参考身份](UPSTREAM.md) · [固定配置](recipe.yaml)。

从仓库根目录直接使用UAV_CV：

```bash
export PYTHONPATH="$PWD:$PWD/../detrscope/src"
UAV_PYTHON=/home/ubuntu/miniconda3/envs/UAV_CV/bin/python
"$UAV_PYTHON" -m experiments.c2_o2mc_5090.test_o2mc

# 完整流程须用独立tmux和文件日志，按主报告中的launch.sh启动。
# 每次选新的root，拒绝覆盖已存在的plan和run；加--smoke只做工程验收。
"$UAV_PYTHON" -u -m experiments.c2_o2mc_5090.run pipeline \
  --root "$PWD/runs/c2_o2mc_5090/5090-c2-o2mc-new"
```

先做数据审计和CPU门禁，再做65图/1轮训练、8图正式COCO工程验收；通过后自动从零训练6471图/150轮、评估best完整548图、严格移除辅助头导出原生C2、同环境复评已有C2 seed0基线，并生成紧凑报告。每轮第三个batch记录分配和主辅梯度，不使用DETRScope在线梯度功能。

默认无自动续训。中断时保留目录，精确核对last/resume.pt中的epoch、optimizer、EMA等状态后单独设计恢复，不传布尔resume，也不从best伪装连续训练。归档生成不触发训练。
