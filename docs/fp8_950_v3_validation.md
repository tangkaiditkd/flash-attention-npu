# 950 V3 FP8 实施与验收记录

E4M3FN Q/K/V → BF16 O、FP32 LSE。Dq/Dk/Dv固定128，Q/KV基块128/256。
覆盖现有定长、变长、已有cache读取、分页和split-KV；所有FP8 mask采用高精度DN。
复用已有FD，未扩展cache追加、batch重映射或split场景；V4、backward及其他FP8格式不在范围内。

## 2026-10-22：clean code前的已验收基线

CANN9.3、950 V3最终版本：原生ATK固定72例＋随机512例，**584/584通过**；O 584项、开启的LSE 297项全部通过。阈值仍为2/1.2/1.2，没有精度豁免。此前ID173的causal＋softcap LSE失败已修复，其最大相对误差比由2.34777降为0.62005。

FP16/BF16 FD 12例、窗口16例、metadata 9项、接口/图模式检查及可选descale 40组通过。最终二进制SHA256为`5018758c04ca26c477ccd225ed15b9f12fca880b306c7eff56dea95cfa13f878`；源码在本次提交前与验收快照逐文件校验一致，保存基线期间只修正文档。

性能共125个配置，其中103个主配置的attention几何平均加速比（BF16/FP8）：无mask 1.3472×、causal 1.4873×、window 1.3838×、FD 1.5158×。FD包含初始化和combine的设备kernel合计为1.6853×；这些是torch profiler记录，均排除host开销。

msprof op另复测12个相同代表配置：主kernel逐例加速比与同期torch profiler最大相对差异2.82%；6个FD的attention＋combine合计为1.7192×对1.6913×。combine单项的绝对时间存在稳定差异，不能混用两工具做优化前后比较。该子集不是103组全量重测。

完整原始证据和已验收二进制保留在仓外：
`/home/tangkaidi/flash_attention_npu_fp8_validation_archive/20261022/perf/`。
其中`round_20261021/README.md`是最终验收，`msprof_op_crosscheck_20261022/README.md`是工具对照。大型性能产物不随源码提交；下文为历史记录，不代表当前精度状态。

## 历史记录：2026-10-21 descale 连续存储回归

FP8 descale 不再向 kernel 传递 batch/head stride；按 `[batch, num_kv_heads]` 连续地址读取。
Python 公共入口先将非连续 descale 转为连续 Tensor，直接 C++ 后端要求连续输入。
950 V3 已在 CANN 9.3.0（`/home/tangkaidi/complete_1019/cann`）下完整强制重编译，包括 AICPU 对象。

当时独立 ATK 完整执行584例：**583通过、1例精度失败，运行异常0例**，耗时3930秒。
O为584/584通过；297例开启LSE，其中296通过、1失败；其余287例关闭LSE。共881项输出比较。
精度标准保持原生双标杆2/1.2/1.2。
失败用例为ID173（`fp8_0572_flash_attn_with_kvcache_causal`）：B4、Q739、KV630、Hq8/Hkv2、softcap=10。
只有LSE最大相对误差比2.347769超过2；平均和均方根误差比通过，O通过。
使用此前安装的、保留descale stride的9.3扩展独立重放此例，高/低标杆与NPU的O/LSE均与修改后逐位一致。
因此此例不是本次删除stride引入的回归；仍保留失败状态，未更改输入、标杆或阈值。
这是另一个用例，不属于此前用户接受的窄窗口LSE例外。

四种非连续/广播view的公共接口连续化结果与显式连续输入逐位一致，直接C++后端拒绝非连续descale。
非量化回归：12个FP16/BF16 FD constant-V用例及6项metadata测试通过。

[本轮汇总及完整报告索引](../../flash_attention_npu_fp8_atk/result/contiguous_descales/summary.json)、
[失败用例修改前后对比](../../flash_attention_npu_fp8_atk/result/contiguous_descales/case173_baseline_comparison.json)。

下文792例精度、stride覆盖和性能数据是此前CANN 9.2版本的历史验收记录，不替代本轮结果。

## 历史精度与接口（CANN 9.2）

**选定版本：208/208固定用例＋584/584新增随机配置全部通过，共792次执行、1584项O/LSE比较。**
固定集0/13配置重复；新增随机集中512个尺寸互异，均按batch1..10、Q/KV1..4096、KV head/group1..5、D128抽样。
其余64例补充短Q/FD，8例补充单轴广播。每例保存独立种子、实际FP8输入、descale、输出和调度。
新增集mask分布为195无mask、194causal、195窗口；包含151分页、88显式metadata、120softcap和88signed输入配置。
shape/dtype/NaN/无穷位置及全mask语义单独检查，有限值直接交给原生ATK双标杆；阈值固定2/1.2/1.2，没有附加容差。

|输出|最大相对误差比例上限|平均相对误差比例上限|均方根误差比例上限|
|---|---:|---:|---:|
|O|1.00513|1.0009|1.0003|
|LSE|1.24776|0.00926523|1|

高标杆以FP64一次返回O/LSE；低标杆以实际分块/遍历/舍入边界一次返回O/LSE。
低标杆的CPU Cube模型按16乘积组共同指数对齐/截断，再按顺序FP32累加；独立种子20矩阵的81920输出与CANN矩阵乘逐位一致。
case159暴露了原CPU普通matmul遗漏的Cube舍入；高标杆、输入及阈值均未修改，新完整矩阵重新执行。
窗口半区裁剪只省略exp(-inf)=0的贡献，不改变有效的四链FP32求和顺序。
接口、预分配输出、fake、backward拒绝、metadata图重放/compile检查通过；12个FP16/BF16 FD、6个metadata和16个FP16/BF16窗口回归通过。
新增随机集按实际planner核实：{'FA': 572, 'FD': 12}；窗口回退为FA的用例没有计作FD。
[实际调度覆盖统计](../../flash_attention_npu_precision_atk/validation/opt2/random_coverage.json)。

[精度汇总](../../flash_attention_npu_precision_atk/validation_results.json)、
[584例逐批原生报告与哈希](../../flash_attention_npu_precision_atk/artifacts/opt2_cube_random/summary.json)、
[固定集报告索引](../../flash_attention_npu_precision_atk/artifacts/opt2_cube_fixed/accuracy_result.json)、
[工具与重放方法](../../flash_attention_npu_precision_atk/README.md)。

## 性能

基准为非量化BF16输入/BF16输出，FP8为E4M3FN输入/BF16输出，均返回FP32 LSE。
输入反量化准备在计时之外。每配置每dtype每轮10次相邻256 MiB ArgMax→target，交换dtype顺序采集两轮；保留全部样本。
下面64例为attention单kernel统计，包含16 MHA、32 GQA、16 MQA，含非等长及未对齐尾块。

|Mask|配置数|FP8→BF16几何平均|BF16→FP8几何平均|两轮范围|
|---|---:|---:|---:|---:|
|none|32|1.320×|1.321×|1.225–2.064×|
|causal|32|1.425×|1.426×|1.287–2.166×|
|all|64|1.372×|1.372×|1.225–2.166×|

64例两轮均更快：64/64；最小attention中位耗时121.930μs，共2560条attention耗时。
相同64配置与上一轮已保存FP8数据的几何平均耗时比为1.118× / 1.120×；这是跨轮历史比较，不混入本轮FP8/BF16统计。
首测完整target波动超过10%的配置ID：[2, 4, 8, 10, 11, 14, 15, 26, 27, 36, 38, 46]。首测/追加复测/独立op数据分开保留，不删除或择优替换。

[64例全部样本与统计](../../flash_attention_npu_precision_atk/perf_dn/opt2_cube64/report.md)、
[逐例CSV](../../flash_attention_npu_precision_atk/perf_dn/opt2_cube64/comparison.csv)。

全场景另测39个放大配置：窗口、四种page size、定长/变长，以及短Q的auto/explicit/single split。
短Q窗口仍是已有普通FA回退，没有新增FD窗口支持。预生成metadata与动态metadata分会话采集，全部launch通过校验。

|Mask|配置数|attention几何平均（两轮）|完整target几何平均（两轮）|
|---|---:|---:|---:|
|none|10|1.350× / 1.347×|1.309× / 1.310×|
|causal|10|1.444× / 1.437×|1.401× / 1.394×|
|window|19|1.285× / 1.284×|1.278× / 1.277×|
|all|39|1.341× / 1.338×|1.317× / 1.315×|

完整target包含初始化、运行时metadata和FD combine；与attention单kernel分开比较。
39例两轮完整target均更快；最小attention中位耗时203.549μs。
全场景完整target首测高波动ID：[9, 15, 18, 22, 36, 37]；全部原始样本保留。
追加两轮复测的6个配置中位数仍全部更快；9号的极端值来自ComputeFAMetadata（约85.7/85.2ms），保留原样。
9号独立attention复核为1.305×，最大波动4.0%，FP8/BF16样本范围无重叠。
完整调用收益最小的配置为约1.084×；短Q、固定单split及现有FA回退的收益小于充分占满核心的DN长序列。
[39例完整结果](../../flash_attention_npu_precision_atk/perf_dn/opt2_cube_scenes/report.md)、
[逐例CSV](../../flash_attention_npu_precision_atk/perf_dn/opt2_cube_scenes/comparison.csv)、
[全场景波动项独立复测](../../flash_attention_npu_precision_atk/perf_dn/opt2_cube_scenes_recheck/report.md)。

## 选定优化与资源

- QFA式完整256-KV DN QK，分页128按页分段搬运后一次MMAD/Fixpipe；其余page size保持原寻址。
- BSA HIF8式FP32 max/sum/correction、四条寄存器链和固定P条带；结合QFA/FIA的E4M3FN Cast、Gather和StoreAlign。
- 窗口使用无符号区间判断；完全不可见的128列半区在VF前以模板选择，省去该半区max/exp，保持P布局及有效求和顺序。
- PV K尾部按32对齐，L1 V仅初始化padding，避免有效DMA与全区清零重叠；LSE采用max+log(sumScaled/256)避免额外消减。
- P指数位转换、额外OTmp buffer、Q驻留L0、八链展开、逐列窗口分支和压缩PV行容量等方案均有实验记录；收益不足或回归的改动已撤回。

每AIV UB233984 bytes、每AIC L1245760 bytes；S/P双buffer、LocalOut单buffer、FP32 O持久区、三个correction槽，预取深度2。
[资源地址与同步生命周期](fp8_950_v3.md)、[Cube参考模型与独立验证](fp8_950_v3_cube_reference.md)、[全部迭代证据](../../flash_attention_npu_precision_atk/perf_dn/opt2/iterations.md)。

## 环境与审查范围

tkd / CANN9.2.0-beta.2 / ATK26.9.24 / Ascend950PR物理NPU3；构建只选择950 V3。
核心频率由每轮npu-smi及采集记录保存；使用每次相邻ArgMax预热，没有宣称锁频。
Dao接口对齐本地提交e7a6402。最终固定/随机精度与正式性能使用相同源码和二进制哈希：

- binary SHA256：`fced969e3f22aae5b2e37f97972b47483472cd8d697f0be642ac0249e957dc33`
- source SHA256：`a6ab22967dfd609ed072ae98c480abb0fc01df1b424405d20bd775b12bc2aaae`
- precision tool SHA256：`e2bcffa6804f6564a7ad0789c7fd49aa75567f6ff08478d5c4635e09da76f93e`

最终七文件静态同步扫描保留24条红线、18条高级别及6条性能候选；图analyzer findings为0，但sync coverage为0。
全部候选保留，并按实际函数边界、初始化token、id/id+16、跨迭代轮转和独立UB区间补充人工核对；动态通过不代替静态完备证明。
[全部42条原文与区间说明](../../flash_attention_npu_precision_atk/review/opt2/selected_sync_review.md)、
[原始扫描](../../flash_attention_npu_precision_atk/review/opt2/selected_sync_audit.json)、
[图证据](../../flash_attention_npu_precision_atk/review/opt2/selected_flow_analysis.json)。

历史探索阶段的大型原始Tensor已迁移到/tmp/flash_attention_npu_historical_artifacts，原路径保留符号链接；
2026-10-21按用户要求清理了最终成功用例的大型Tensor转储；用例、种子、调度、原生报告和失败数据保留。
需要成功用例的原始Tensor时可按保存种子重建；[清理记录](../../flash_attention_npu_precision_atk/validation/storage_cleanup.json)。迁移原因及中断/恢复日志仍见迭代记录。
[历史数据位置清单](../../flash_attention_npu_precision_atk/perf_dn/opt2/historical_artifact_locations.json)。
