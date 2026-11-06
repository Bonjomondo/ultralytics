# 4090主机反复中断：2026-11-06诊断

更新：本报告记录续训前诊断。随后用户明确要求“继续训练”，已核验源码及完整98轮检查点后按原配方从第99轮恢复。此前等待诊断后再续训的条件被用户最新指示替代；根因仍未确认。见[续训授权及实际状态](../2026-11-05_c2_channel_author/user_resume_20261106.json)。

结论：已确认多处原生崩溃，但尚未定位唯一根因。不能把报错名称直接写成“PyTorch bug已确认”，也不能据此认定RTX 4090、CPU或内存损坏。目前必须同时排查共享软件环境中的原生库和CPU/内存稳定性。此前改workers或替换保存路径属于规避措施；成功续跑不等于根因解决。

本次只做状态、源码与日志核对、依赖检查、基础导入和CPU检查点读取；未重启训练、未改配方、未更新环境/驱动/BIOS，未停止其他任务。完整采集见[evidence.json](evidence.json)，检查点核验见[checkpoint_audit.json](checkpoint_audit.json)。

## 当前实验的两次故障

实验运行目录：`/home/zjnu/文档/ZHJ_WorkSpace/ultralytics/runs/c2_channel_author_20261105`。分支：`feat/r18-c2-channelgate-author`。冻结源码提交：`49f46e9bb5364d0463cd057b2fc0fbc9dd118016`，151个文件哈希全部匹配。

| 北京时间 | 位置 | 实际报错 | 处置与解释 |
|---|---|---|---|
| 11月6日02:28 | C2第51轮反向传播 | `t == DeviceType::CUDA INTERNAL ASSERT FAILED`，CUDAGuardImpl | 自动从完整第50轮检查点恢复；raw/EMA/优化器恢复验证通过。断言只说明内部收到的设备类型不满足CUDA检查，未指出是谁生成了错误状态。 |
| 11月6日04:59 | 恢复后的C2第99轮，约1406/1618批次，取下一批增强样本时 | `Fatal Python error: Segmentation fault`，退出码-11 | 训练已从开始统一workers=0，仍然崩溃。控制器达到两次尝试上限后退出。 |

第二次Python栈只落在`augment.py:70`的`data = t(data)`，没有保留下层原生调用栈和当批增强输入。因此不能点名OpenCV、某一种增强、某张图片或某个CPU核心为根因。第一次断言也没有足够原生调用证据定位设备类型错误的来源。

截至本次核查：队列failed、无训练子进程、GPU无计算任务。run/status.json仍写training属于进程崩溃后遗留的心跳，不能据此宣称在跑。完成98/150轮；Scalar和ChannelGate正式训练尚未开始。第99轮未计入已完成预算。完整98轮训练CSV已保留为失败时快照，尚无本轮正式COCO AP或筛选晋级结论。

last.pt与best.pt均为323167626字节，SHA256均为`3ce7f883929a5effbffdf09f2605c1fe1a7dfe9033fdc65227d16a6b3caf8e80`。已使用冻结后端、屏蔽GPU，在CPU加载检查点，确认epoch=97（零起算，即98轮完成），raw与EMA各533项全部有限、优化器存在，同时包含RNG、scheduler、scaler、stopper等字段。这证明检查点可读取且未发现权重NaN/Inf，不证明硬件稳定或能精确恢复同一采样轨迹。

## 历史故障为何相关，但不能直接合并根因

- 先前Scalar作者引擎350轮在第131轮遇到DataLoader worker SIGSEGV；从完整130轮状态恢复，workers8改0后完成。说明减小加载并发曾帮助完成实验，不能证明所有故障都来自多进程。
- 同一实验完成预测后的独立CPU COCOeval曾SIGSEGV，同一GT/预测重新评估成功。这说明排查范围必须包含CPU原生库，不能只盯CUDA反向传播；一次重试成功也不能排除数据触发的非确定性问题。
- 更早还出现保存路径相关段错误，改用标准库pickle与原子保存规避。不能把成功保存写成CPU/内存已通过稳定性检测。
- 内核历史保留了Python3.11段错误以及ROS/PCL程序段错误。ROS错误多次在相同库位置发生，可能是独立程序bug，不将它们当作硬件故障的直接证据。

## 已核对的环境与排除边界

| 项目 | 实际观察 | 能说明什么 |
|---|---|---|
| CPU/主机 | Intel i9-13900K / Dell Precision 3660 Tower | 该组合在Dell关于13/14代处理器稳定性问题的公告范围内；是重点排查线索，非故障诊断结果。 |
| BIOS/微码 | 2.38.0，DMI日期06/02/2026；微码0x133 | 不能简单说“没更新BIOS导致”；本次未确认最新版本、默认电压/功耗配置或硬件是否已老化。 |
| 软件 | Python3.11；torch2.1.2+cu121，torchvision0.16.2+cu121，NumPy1.26.4，OpenCV4.11.0，Pillow12.3.0，pycocotools2.0.11 | 同一共享环境有多个原生扩展；版本清单本身不能证明ABI或运行时稳定。 |
| NVIDIA | 驱动535.183.01；PyTorch CUDA runtime12.1 | 满足NVIDIA CUDA12.1最低驱动要求，不能把nvidia-smi显示12.2与torch12.1直接说成不兼容。满足最低要求也不保证无软件bug。 |
| 依赖/基础加载 | pip check通过；torch/NumPy/OpenCV/Pillow导入通过；检查点CPU加载通过 | 只排除了已声明依赖缺失和本次基础加载故障，未做长时间原生运算稳定性验证。 |
| 内核记录 | 保留的dmesg未匹配到NVIDIA Xid/NVRM计算故障、MCE/EDAC硬件报错；指定时间kernel journal无条目 | 日志范围有限；未记录不等于硬件健康。消费卡无可用ECC统计也不能充当内存检测。 |
| OOM | 10月29日23:58的OOM杀的是rviz；本次训练未找到对应OOM记录 | 是另一次事件，不能拿它解释11月6日训练中断。当前空闲内存与温度不能回推崩溃时峰值。 |

## 后续诊断与恢复条件

1. 保留原环境、冻结源码、完整日志和98轮检查点；本次不追加盲目重试。已有实验跟进继续用于诊断，恢复前须先满足下面条件。
2. 在隔离环境中先保持相同依赖版本、源码与输入，反复运行数据增强、原生COCOeval和保存/加载最小复现；尽可能记录输入、种子、原生调用栈。再逐项改变版本或并发设置。若同输入跨环境可稳定复现，优先定位具体原生代码；未复现不能视为排除。
3. 对i9-13900K平台做CPU与内存稳定性诊断，核对Dell支持的BIOS/默认电压功耗与内存配置。BIOS/硬件设置或离线内存诊断可能需要维护窗口，不在本次只读核查中直接更改。短时间正常不能排除间歇故障。
4. 针对性修复和短程验证有证据后，才审计是否恢复原98轮状态。若更改环境或训练执行协议，另记实验身份并保持C2/Scalar/Channel对照一致；已有中断对样本次序的影响须继续披露，不能称逐位等价配对。

## 官方参考

- [PyTorch v2.1.2 CUDAGuardImpl源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.1.2/c10/cuda/impl/CUDAGuardImpl.h)：构造函数内部断言检查设备类型为CUDA；该检查不诊断物理显卡故障。
- [NVIDIA CUDA12.1.1发布说明](https://docs.nvidia.com/cuda/archive/12.1.1/cuda-toolkit-release-notes/index.html)：Linux CUDA12.1最低兼容驱动525.60.13。
- [Dell公告000227933](https://www.dell.com/support/kbdoc/en-us/000227933/industry-wide-issue-with-intel-core-13th-and-14th-generation-i5-i7-and-i9-processors)：列出i9-13900K和Precision3660，建议更新BIOS后仍崩溃时继续由Dell支持诊断。本文据此提高平台稳定性排查优先级，不推断这颗CPU已经损坏。
