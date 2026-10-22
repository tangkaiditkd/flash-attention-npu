# 950 V3 FP8 Cube 低标杆模型

本模型最初在Ascend950PR / DAV_3510、CANN9.2.0-beta.2环境建立，使用E4M3FN输入和FP32 MMAD输出。
当前CANN9.3的完整ATK验收继续使用此模型；下文独立Cube验证记录属于原9.2环境。
它补齐CPU低标杆的Cube原语语义；没有改变生产算子的OnlineSoftmax精度、高标杆或ATK阈值。
16项分组描述的是Cube内部点积舍入，不引入按block的输入scale；Q/K/V descale仍为可选[batch, num_kv_heads]。

## 发现与定界

随机case159为B9、Q4074、KV3682、Hq4/Hkv1、窗口(0,0)，descale广播，seed20261180。
每个有效query只有一个有效K，因此LSE等于缩放后的QK score。
O通过，LSE最大相对误差比例7.381超过2；最大偏差点的LSE约2.8e-4，绝对差约1.26e-7。
同一输入重复五次，结果一致。

阶段导出显示：原LSE与内核max逐位一致；进一步导出原始QK，132552个有效score中2396个与普通CPU FP32点积不同。
独立CANN npu_quant_matmul在抽取的64个异常点复现了全部差异。
把独立矩阵乘的K128拆成K64或K32没有消除差异；逐项乘法矩阵检查未发现乘积错误。
常见CPU串行、分组、交错和树形FP32累加均不能模拟这些输出。

这些证据将问题限定为原CPU低标杆遗漏了实际FP8点积的对齐/截断行为，不能用修改log、P缩放或ATK容差解决。
临时max/QK导出已移除；生产代码保持既定FP8 MMAD和FP32 OnlineSoftmax流水。

## 经实验验证的计算模型

将点积沿K分为连续16项，令本组乘积为p_i=a_i*b_i。
对于非零组，取最大乘积的二进制指数E_g，定义：

\[
u_g=2^{E_g-18},\qquad
r_g=\sum_{i=0}^{15}u_g\,\operatorname{trunc}(p_i/u_g).
\]

按K顺序进行组间FP32累加：

\[
s_0=0,\qquad s_{g+1}=\mathrm{FP32}(s_g+r_g).
\]

全零组贡献零。每个对齐后的乘积最多19个有效位，16项组内和最多增加4位，所以组内和可由FP32精确表示。
CPU实现使用等价的FP32位掩码截断，避免在每个点积上构造大临时Tensor；组间仍按顺序执行FP32加法。

这是一项针对上述环境的实验验证模型，不将未查到的硬件内部实现细节当作官方保证。
API侧依据为安装的Ascend C MmadParams/Reg接口和实际FP8→FP32调用；具体对齐规则由下面的独立对照验证。

## 独立验证

- case159全部132552个有效score由模型准确复现。
- 使用独立seed20261225，测试K32/64/128/256和normal/small/large/mixed/PV五类输入，20个64×64矩阵、81920个输出全部与CANN npu_quant_matmul逐位一致。
- Python原型与正式CPU C++实现分别验证；C++实现还比较了FP32原始位模式。
- CPU模型没有调用NPU矩阵乘。独立NPU对照只存在于验证脚本中；ATK低标杆使用本地C++ CPU参考库。
- torch._scaled_mm曾报告CPU fallback，相关结果没有用作NPU证据。

[独立验证数据](../../flash_attention_npu_precision_atk/perf_dn/opt2/cube_model_holdout_final.json)、
[验证脚本](../../flash_attention_npu_precision_atk/verify_cube_matmul.py)、
[CPU实现](../../flash_attention_npu_precision_atk/cube_golden.cpp)、
[低标杆](../../flash_attention_npu_precision_atk/golden.py)。

高标杆仍从实际FP8输入以FP64计算；输入、随机种子及原生双标杆阈值2/1.2/1.2保持不变。
修正后的完整固定/随机矩阵重新执行，之前部分通过的数据不拼接为最终验收。
