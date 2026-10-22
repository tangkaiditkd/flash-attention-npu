/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef FAI950_ONLINE_SOFTMAX_QUANT_HPP
#define FAI950_ONLINE_SOFTMAX_QUANT_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "kernel_common.hpp"
#include "fa_block.h"

namespace Catlass::Epilogue::Block {
namespace QuantSoftmaxVF {
using namespace AscendC;
using namespace AscendC::Reg;
__simd_vf__ inline void InitDnIndexVF(__ubuf__ uint8_t* indexDn)
{
    RegTensor<uint8_t> baseIndexReg;
    RegTensor<uint8_t> indexReg;
    RegTensor<uint8_t> indexHighReg;
    MaskReg mask32 = CreateMask<int8_t, MaskPattern::VL32>();

    Arange((RegTensor<int8_t>&)baseIndexReg, 0);
    ShiftLefts(baseIndexReg, baseIndexReg, static_cast<int16_t>(2), mask32);
    for (uint16_t i = 0; i < 4; ++i) {
        Adds(indexReg, baseIndexReg, i, mask32);
        StoreAlign(indexDn + i * 64, indexReg, mask32);
        Adds(indexHighReg, indexReg, static_cast<uint8_t>(128), mask32);
        StoreAlign(indexDn + i * 64 + 32, indexHighReg, mask32);
    }
}

}

template <bool IS_DN, bool HAS_SOFTCAP, bool HAS_WINDOW, class LayoutP, class LayoutS>
class BlockEpilogue<EpilogueFAQuantOnlineSoftmax<IS_DN, HAS_SOFTCAP, HAS_WINDOW>,
                    Gemm::GemmType<float8_e4m3_t, LayoutP>, Gemm::GemmType<float, LayoutS>> {
public:
    using DispatchPolicy = EpilogueFAQuantOnlineSoftmax<IS_DN, HAS_SOFTCAP, HAS_WINDOW>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementInput = float;
    using ElementOutput = float8_e4m3_t;
    using ElementMask = uint8_t;
    static constexpr uint32_t UB_S_P_BUF_STAGES = 2;
    static constexpr uint32_t UB_DM_BUF_MAX_STAGES = 3;
    static constexpr uint32_t SM_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t SM_COL_MAX_ELEM_NUM = 256;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag>& resource, const UBufTileHelper& ub,
                                    float scaleValue_, float softcapValue_)
        : scaleValue(scaleValue_), softcapValue(softcapValue_), scalePositive(scaleValue_ >= 0.0f)
    {
        for (uint32_t i = 0; i < UB_S_P_BUF_STAGES; ++i) {
            lsUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(ub.sStartOffset + i * ub.sBufferBytes);
            lpUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOutput>(ub.pStartOffset + i * ub.pBufferBytes);
        }
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(ub.gmStartOffset);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(ub.glStartOffset);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<float>(ub.lmStartOffset);
        llUbTensor = resource.ubBuf.template GetBufferByByte<float>(ub.llStartOffset);
        for (uint32_t i = 0; i < UB_DM_BUF_MAX_STAGES; ++i) {
            dmUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                ub.dmStartOffset + i * SM_ROW_MAX_ELEM_NUM * sizeof(float));
        }
        pGatherIndexUbTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(ub.indexStartOffset);
        QuantSoftmaxVF::InitDnIndexVF((__ubuf__ uint8_t*)pGatherIndexUbTensor.GetPhyAddr());
    }

    __aicore__ inline void SetScoreScale(float value)
    {
        scaleValue = value;
        scalePositive = value >= 0.0f;
    }

    template <class TensorP>
    __aicore__ inline void operator()(TensorP& l1PTensorTla, uint32_t rows, uint32_t qSize,
        uint32_t qStart, uint32_t kvStart, uint32_t kvSize, int32_t qLen, int32_t kvLen,
        int32_t windowLeft, int32_t windowRight, MaskCategory maskCategory,
        bool isFirstKvSTile, uint32_t ubSBufId, uint32_t l1PBufId, Arch::CrossCoreFlag mm1ToSmFlag,
        Arch::CrossCoreFlag smToMm2Flag)
    {
        const uint32_t rowAlign = IS_DN ? 32U : 8U;
        const uint32_t nSplit = RoundUp(rows, rowAlign) / 2;
        const uint32_t firstRows = min(rows, nSplit);
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        const uint32_t n = subBlockIdx == 0 ? firstRows : rows - firstRows;
        const uint32_t rowStart = subBlockIdx == 0 ? 0 : firstRows;
        __ubuf__ float* sAddr = (__ubuf__ float*)lsUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ ElementOutput* pAddr = (__ubuf__ ElementOutput*)lpUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ float* lastMaxAddr = (__ubuf__ float*)gmUbTensor.GetPhyAddr();
        __ubuf__ float* lastSumAddr = (__ubuf__ float*)glUbTensor.GetPhyAddr();
        __ubuf__ float* expMaxAddr = (__ubuf__ float*)dmUbTensor[l1PBufId].GetPhyAddr();
        __ubuf__ uint8_t* gatherIndexAddr = (__ubuf__ uint8_t*)pGatherIndexUbTensor.GetPhyAddr();

        Arch::CrossCoreWaitFlag<4, PIPE_V>(mm1ToSmFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        if constexpr (IS_DN) {
            if (isFirstKvSTile) {
                OnlineSoftmaxDN<false>(sAddr, pAddr, lastMaxAddr, lastSumAddr, expMaxAddr, gatherIndexAddr,
                    n, rowStart, qSize, qStart, kvStart, kvSize, qLen, kvLen, maskCategory, windowLeft, windowRight);
            } else {
                OnlineSoftmaxDN<true>(sAddr, pAddr, lastMaxAddr, lastSumAddr, expMaxAddr, gatherIndexAddr,
                    n, rowStart, qSize, qStart, kvStart, kvSize, qLen, kvLen, maskCategory, windowLeft, windowRight);
            }
        } else {
            __ubuf__ float* nowMaxAddr = (__ubuf__ float*)lmUbTensor.GetPhyAddr();
            __ubuf__ float* nowSumAddr = (__ubuf__ float*)llUbTensor.GetPhyAddr();
            AscendC::Duplicate(lpUbTensor[ubSBufId].template ReinterpretCast<uint16_t>(), uint16_t(0), 8320);
            ComputeND(sAddr, pAddr, lastMaxAddr, lastSumAddr, nowMaxAddr, nowSumAddr, expMaxAddr,
                gatherIndexAddr, n, rowStart, qSize, qStart, kvStart, kvSize, qLen, kvLen,
                windowLeft, windowRight, scaleValue, softcapValue, isFirstKvSTile);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        Arch::CrossCoreSetFlag<4, PIPE_V>(mm1ToSmFlag);
        Arch::CrossCoreWaitFlag<4, PIPE_MTE3>(smToMm2Flag);
        if constexpr (IS_DN) {
            // Eight padded stripes become two contiguous 32-Q-row portions per AIV.
            AscendC::DataCopyParams copy(4, 64, 66, 0);
            AscendC::DataCopy(l1PTensorTla.data()[subBlockIdx * 64 * 256], lpUbTensor[ubSBufId], copy);
            AscendC::DataCopy(l1PTensorTla.data()[subBlockIdx * 64 * 256 + 32 * 256],
                lpUbTensor[ubSBufId][65 * 32], copy);
        } else {
            AscendC::DataCopyParams copy(8, 64, 1, 64);
            AscendC::DataCopy(l1PTensorTla.data()[subBlockIdx * 64 * 32], lpUbTensor[ubSBufId], copy);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        Arch::CrossCoreSetFlag<4, PIPE_MTE3>(smToMm2Flag);
    }

private:
    enum class KvRange : uint32_t {
        LOWER_QUARTER, LOWER_HALF, LOWER_THREE_QUARTERS,
        UPPER_THREE_QUARTERS, UPPER_HALF, UPPER_QUARTER, FULL
    };

    template <KvRange Range>
    struct KvStripe {
        static constexpr uint32_t BEGIN = Range == KvRange::UPPER_QUARTER ? 192U :
            (Range == KvRange::UPPER_HALF ? 128U : (Range == KvRange::UPPER_THREE_QUARTERS ? 64U : 0U));
        static constexpr uint32_t END = Range == KvRange::LOWER_QUARTER ? 64U :
            (Range == KvRange::LOWER_HALF ? 128U : (Range == KvRange::LOWER_THREE_QUARTERS ? 192U : 256U));
    };

    float scaleValue;
    float softcapValue;
    bool scalePositive;
    AscendC::LocalTensor<float> lsUbTensor[UB_S_P_BUF_STAGES];
    AscendC::LocalTensor<ElementOutput> lpUbTensor[UB_S_P_BUF_STAGES];
    AscendC::LocalTensor<float> dmUbTensor[UB_DM_BUF_MAX_STAGES];
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> lmUbTensor;
    AscendC::LocalTensor<float> llUbTensor;
    AscendC::LocalTensor<uint8_t> pGatherIndexUbTensor;

    template <class Reg, class Mask>
    __simd_callee__ static inline void Softcap(Reg& value, float cap, Mask& mask)
    {
        using namespace AscendC::Reg;
        // Form the exponential difference before division to reduce cancellation.
        Reg numerator, denominator;
        Muls(value, value, -2.0f, mask);
        Mins(value, value, 17.6f, mask);
        Exp(value, value, mask);
        Duplicate(numerator, cap);
        Axpy(numerator, value, -cap, mask);
        Adds(denominator, value, 1.0f, mask);
        Div(value, numerator, denominator, mask);
    }

    __simd_vf__ static inline void ComputeND(__ubuf__ float* s, __ubuf__ ElementOutput* p,
        __ubuf__ float* gm, __ubuf__ float* gl, __ubuf__ float* lm, __ubuf__ float* ll,
        __ubuf__ float* dm, __ubuf__ uint8_t* index, uint32_t rows, uint32_t rowStart,
        uint32_t qSize, uint32_t qStart, uint32_t kvStart, uint32_t kvSize,
        int32_t qLen, int32_t kvLen, int32_t left, int32_t right, float scale, float cap, bool first)
    {
        using namespace AscendC::Reg;
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        MaskReg bytes = CreateMask<uint8_t, MaskPattern::ALL>();
        RegTensor<float> x0, x1, x2, x3, maximum, reduced, minValue;
        RegTensor<int32_t> kIndex;
        RegTensor<float> oldMax, oldSum, alpha, sum0, sum1, sum2, sum3, sum;
        RegTensor<ElementOutput> p0, p1, p2, p3, packed, pair0, pair1;
        RegTensor<uint8_t> gatherIndex;
        static constexpr CastTrait c0 = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr CastTrait c1 = {RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr CastTrait c2 = {RegLayout::TWO, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr CastTrait c3 = {RegLayout::THREE, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        UnalignRegForStore maxStore, sumStore;
        __ubuf__ float* maxDst = lm;
        __ubuf__ float* sumDst = ll;
        Duplicate(minValue, -__builtin_inff());
        LoadAlign(gatherIndex, index);
        for (uint32_t row = 0; row < 64; ++row) {
            const int32_t center = int32_t(qStart + (rowStart + row) % qSize) + kvLen - qLen;
            const int32_t lower = center - left;
            const int32_t upper = center + right;
            Duplicate(maximum, -3.38953139e38f);
            for (uint32_t part = 0; part < 4; ++part) {
                RegTensor<float> value;
                Arange(kIndex, int32_t(kvStart + part * 64));
                MaskReg lo, hi, valid;
                CompareScalar<int32_t, AscendC::CMPMODE::GE>(lo, kIndex, lower, full);
                CompareScalar<int32_t, AscendC::CMPMODE::LE>(hi, kIndex, upper, full);
                MaskAnd(valid, lo, hi, full);
                uint32_t tailCount = kvSize > part * 64 ? min(64U, kvSize - part * 64) : 0U;
                MaskReg tail = UpdateMask<float>(tailCount);
                MaskAnd(valid, valid, tail, full);
                LoadAlign(value, s + row * 256 + part * 64);
                Muls(value, value, scale, full);
                if constexpr (HAS_SOFTCAP) Softcap(value, cap, full);
                if (row >= rows) Duplicate(value, -__builtin_inff());
                else Select(value, value, minValue, valid);
                Max(maximum, maximum, value, full);
                StoreAlign(s + row * 256 + part * 64, value, full);
            }
            Reduce<ReduceType::MAX, float, float, MaskMergeMode::ZEROING>(reduced, maximum, full);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(maxDst, reduced, maxStore, 1);
        }
        StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(maxDst, maxStore, 0);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        LoadAlign(maximum, lm);
        if (!first) {
            LoadAlign(oldMax, gm);
            Max(maximum, maximum, oldMax, full);
            FusedExpSub(alpha, oldMax, maximum, full);
            StoreAlign(dm, alpha, full);
        }
        StoreAlign(gm, maximum, full);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        for (uint32_t row = 0; row < 64; ++row) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(maximum, gm + row);
            LoadAlign(x0, s + row * 256);
            LoadAlign(x1, s + row * 256 + 64);
            LoadAlign(x2, s + row * 256 + 128);
            LoadAlign(x3, s + row * 256 + 192);
            FusedExpSub(x0, x0, maximum, full);
            FusedExpSub(x1, x1, maximum, full);
            FusedExpSub(x2, x2, maximum, full);
            FusedExpSub(x3, x3, maximum, full);
            Muls(x0, x0, 256.0f, full); Muls(x1, x1, 256.0f, full);
            Muls(x2, x2, 256.0f, full); Muls(x3, x3, 256.0f, full);
            Add(sum0, x0, x2, full); Add(sum1, x1, x3, full);
            Add(sum, sum0, sum1, full);
            Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(sum, sum, full);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(sumDst, sum, sumStore, 1);
            Cast<ElementOutput, float, c0>(p0, x0, full); Cast<ElementOutput, float, c1>(p1, x1, full);
            Cast<ElementOutput, float, c2>(p2, x2, full); Cast<ElementOutput, float, c3>(p3, x3, full);
            Or((RegTensor<uint8_t>&)pair0, (RegTensor<uint8_t>&)p0, (RegTensor<uint8_t>&)p1, bytes);
            Or((RegTensor<uint8_t>&)pair1, (RegTensor<uint8_t>&)p2, (RegTensor<uint8_t>&)p3, bytes);
            Or((RegTensor<uint8_t>&)packed, (RegTensor<uint8_t>&)pair0, (RegTensor<uint8_t>&)pair1, bytes);
            Gather(packed, packed, gatherIndex);
            __ubuf__ ElementOutput* dst = p + row * 32;
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(dst, packed, 65, bytes);
        }
        StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(sumDst, sumStore, 0);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        LoadAlign(sum, ll);
        if (!first) {
            LoadAlign(oldSum, gl);
            LoadAlign(alpha, dm);
            Mul(oldSum, oldSum, alpha, full);
            Add(sum, sum, oldSum, full);
        }
        StoreAlign(gl, sum, full);
    }
    template <bool IsUpdate>
    __aicore__ inline void OnlineSoftmaxDN(__ubuf__ float* sAddr, __ubuf__ ElementOutput* pAddr,
        __ubuf__ float* lastMaxAddr, __ubuf__ float* lastSumAddr, __ubuf__ float* expMaxAddr,
        __ubuf__ uint8_t* gatherIndexAddr, uint32_t rows, uint32_t rowStart, uint32_t qSize,
        uint32_t qStart, uint32_t kvStart, uint32_t kvSize, int32_t qLen, int32_t kvLen,
        MaskCategory maskCategory, int32_t left, int32_t right)
    {
        const int32_t causalStart = static_cast<int32_t>(qStart) + kvLen - qLen;
        if constexpr (HAS_WINDOW) {
            const bool fullyVisible = static_cast<int32_t>(kvStart) >=
                causalStart + static_cast<int32_t>(qSize) - 1 - left &&
                static_cast<int32_t>(kvStart + kvSize) - 1 <= causalStart + right;
            if (kvSize == SM_COL_MAX_ELEM_NUM) {
                if (fullyVisible) {
                    ComputeOnlineSoftmaxDN<IsUpdate, false, true>(sAddr, pAddr, lastMaxAddr, lastSumAddr,
                        expMaxAddr, gatherIndexAddr, rows, rowStart, qSize, causalStart, kvStart, kvSize,
                        scaleValue, softcapValue, scalePositive, left, right);
                } else {
                    MaskedSoftmaxDN<IsUpdate, true>(sAddr, pAddr, lastMaxAddr, lastSumAddr,
                        expMaxAddr, gatherIndexAddr, rows, rowStart, qSize, causalStart, kvStart, kvSize,
                        left, right);
                }
            } else {
                MaskedSoftmaxDN<IsUpdate, false>(sAddr, pAddr, lastMaxAddr, lastSumAddr,
                    expMaxAddr, gatherIndexAddr, rows, rowStart, qSize, causalStart, kvStart, kvSize,
                    left, right);
            }
        } else {
            const bool needMask = maskCategory == MaskCategory::MASK_CAUSAL &&
                                  static_cast<int32_t>(kvStart + kvSize) > causalStart;
            if (needMask) {
                MaskedSoftmaxDN<IsUpdate, false>(sAddr, pAddr, lastMaxAddr, lastSumAddr,
                    expMaxAddr, gatherIndexAddr, rows, rowStart, qSize, causalStart, kvStart, kvSize, 0, 0);
            } else if (kvSize == SM_COL_MAX_ELEM_NUM) {
                ComputeOnlineSoftmaxDN<IsUpdate, false, true>(sAddr, pAddr, lastMaxAddr, lastSumAddr,
                    expMaxAddr, gatherIndexAddr, rows, rowStart, qSize, causalStart, kvStart, kvSize,
                    scaleValue, softcapValue, scalePositive);
            } else if (kvSize <= 64U) {
                ComputeOnlineSoftmaxDN<IsUpdate, false, false, KvRange::LOWER_QUARTER>(sAddr, pAddr,
                    lastMaxAddr, lastSumAddr, expMaxAddr, gatherIndexAddr, rows, rowStart, qSize,
                    causalStart, kvStart, kvSize, scaleValue, softcapValue, scalePositive);
            } else if (kvSize <= 128U) {
                ComputeOnlineSoftmaxDN<IsUpdate, false, false, KvRange::LOWER_HALF>(sAddr, pAddr,
                    lastMaxAddr, lastSumAddr, expMaxAddr, gatherIndexAddr, rows, rowStart, qSize,
                    causalStart, kvStart, kvSize, scaleValue, softcapValue, scalePositive);
            } else if (kvSize <= 192U) {
                ComputeOnlineSoftmaxDN<IsUpdate, false, false, KvRange::LOWER_THREE_QUARTERS>(sAddr, pAddr,
                    lastMaxAddr, lastSumAddr, expMaxAddr, gatherIndexAddr, rows, rowStart, qSize,
                    causalStart, kvStart, kvSize, scaleValue, softcapValue, scalePositive);
            } else {
                ComputeOnlineSoftmaxDN<IsUpdate, false, false>(sAddr, pAddr, lastMaxAddr, lastSumAddr,
                    expMaxAddr, gatherIndexAddr, rows, rowStart, qSize, causalStart, kvStart, kvSize,
                    scaleValue, softcapValue, scalePositive);
            }
        }
    }

    template <bool IsUpdate, bool FullKv>
    __aicore__ inline void MaskedSoftmaxDN(__ubuf__ float* sAddr, __ubuf__ ElementOutput* pAddr,
        __ubuf__ float* lastMaxAddr, __ubuf__ float* lastSumAddr, __ubuf__ float* expMaxAddr,
        __ubuf__ uint8_t* gatherIndexAddr, uint32_t rows, uint32_t rowStart, uint32_t qSize,
        int32_t causalStart, uint32_t kvStart, uint32_t kvSize, int32_t left, int32_t right)
    {
        // Select a fixed KV stripe before VF; inactive chains produce zero P.
        uint32_t qOffset = rowStart % qSize;
        bool wraps = rows > qSize - qOffset;
        int32_t firstQ = causalStart + static_cast<int32_t>(wraps ? 0U : qOffset);
        int32_t lastQ = causalStart + static_cast<int32_t>(wraps ? qSize - 1U : qOffset + (rows ? rows - 1U : 0U));
        int32_t halfStart = static_cast<int32_t>(kvStart) + 128;
        int32_t lastKey = min(lastQ + right, static_cast<int32_t>(kvStart + kvSize) - 1);
        if (lastKey < static_cast<int32_t>(kvStart) + 64) {
            ComputeOnlineSoftmaxDN<IsUpdate, true, FullKv, KvRange::LOWER_QUARTER>(sAddr, pAddr,
                lastMaxAddr, lastSumAddr, expMaxAddr, gatherIndexAddr, rows, rowStart, qSize,
                causalStart, kvStart, kvSize, scaleValue, softcapValue, scalePositive, left, right);
        } else if (HAS_WINDOW && firstQ - left >= static_cast<int32_t>(kvStart) + 192) {
            ComputeOnlineSoftmaxDN<IsUpdate, true, FullKv, KvRange::UPPER_QUARTER>(sAddr, pAddr,
                lastMaxAddr, lastSumAddr, expMaxAddr, gatherIndexAddr, rows, rowStart, qSize,
                causalStart, kvStart, kvSize, scaleValue, softcapValue, scalePositive, left, right);
        } else if (lastKey < halfStart) {
            ComputeOnlineSoftmaxDN<IsUpdate, true, FullKv, KvRange::LOWER_HALF>(sAddr, pAddr, lastMaxAddr, lastSumAddr,
                expMaxAddr, gatherIndexAddr, rows, rowStart, qSize, causalStart, kvStart, kvSize,
                scaleValue, softcapValue, scalePositive, left, right);
        } else if (HAS_WINDOW && firstQ - left >= halfStart) {
            ComputeOnlineSoftmaxDN<IsUpdate, true, FullKv, KvRange::UPPER_HALF>(sAddr, pAddr, lastMaxAddr, lastSumAddr,
                expMaxAddr, gatherIndexAddr, rows, rowStart, qSize, causalStart, kvStart, kvSize,
                scaleValue, softcapValue, scalePositive, left, right);
        } else if (lastKey < static_cast<int32_t>(kvStart) + 192) {
            ComputeOnlineSoftmaxDN<IsUpdate, true, FullKv, KvRange::LOWER_THREE_QUARTERS>(sAddr, pAddr,
                lastMaxAddr, lastSumAddr, expMaxAddr, gatherIndexAddr, rows, rowStart, qSize,
                causalStart, kvStart, kvSize, scaleValue, softcapValue, scalePositive, left, right);
        } else if (HAS_WINDOW && firstQ - left >= static_cast<int32_t>(kvStart) + 64) {
            ComputeOnlineSoftmaxDN<IsUpdate, true, FullKv, KvRange::UPPER_THREE_QUARTERS>(sAddr, pAddr,
                lastMaxAddr, lastSumAddr, expMaxAddr, gatherIndexAddr, rows, rowStart, qSize,
                causalStart, kvStart, kvSize, scaleValue, softcapValue, scalePositive, left, right);
        } else {
            ComputeOnlineSoftmaxDN<IsUpdate, true, FullKv>(sAddr, pAddr, lastMaxAddr, lastSumAddr,
                expMaxAddr, gatherIndexAddr, rows, rowStart, qSize, causalStart, kvStart, kvSize,
                scaleValue, softcapValue, scalePositive, left, right);
        }
    }

    template <bool HasMask, bool FullKv, bool ScaleScore = true>
    __simd_callee__ static inline void LoadScoreDN(AscendC::Reg::RegTensor<float>& srcVreg,
        __ubuf__ float* srcUb, uint32_t kvRow, uint32_t kvSize, uint32_t kvStart,
        AscendC::Reg::RegTensor<int32_t>& qIndexVreg, float dScale, float cap,
        AscendC::Reg::MaskReg& pregTailN, int32_t left = 0, int32_t right = 0)
    {
        using namespace AscendC::Reg;
        LoadAlign(srcVreg, srcUb + kvRow * SM_ROW_MAX_ELEM_NUM);
        if constexpr (ScaleScore) {
            Muls(srcVreg, srcVreg, dScale, pregTailN);
            if constexpr (HAS_SOFTCAP) {
                Softcap(srcVreg, cap, pregTailN);
            }
        }
        if constexpr (HasMask || !FullKv) {
            RegTensor<float> minVreg;
            MaskReg pregVisible;
            Duplicate(minVreg, -__builtin_inff());
            if constexpr (HAS_WINDOW) {
                RegTensor<uint32_t> distance;
                const int32_t key = static_cast<int32_t>(kvStart + kvRow);
                Adds(distance, (RegTensor<uint32_t>&)qIndexVreg, static_cast<uint32_t>(right - key), pregTailN);
                CompareScalar<uint32_t, AscendC::CMPMODE::LE>(
                    pregVisible, distance, static_cast<uint32_t>(left + right), pregTailN);
                if constexpr (!FullKv) {
                    uint32_t count = kvRow < kvSize ? SM_ROW_MAX_ELEM_NUM : 0U;
                    MaskReg tail = UpdateMask<float>(count);
                    MaskAnd(pregVisible, pregVisible, tail, pregTailN);
                }
            } else {
                CompareScalar<int32_t, AscendC::CMPMODE::GE>(
                    pregVisible, qIndexVreg, static_cast<int32_t>(kvStart + kvRow), pregTailN);
            }
            Select(srcVreg, srcVreg, minVreg, pregVisible);
        }
    }

    template <bool HasMask, bool FullKv, bool ScaleScore, KvRange ValidRange = KvRange::FULL>
    __simd_callee__ static inline void ComputeMaxDN(AscendC::Reg::RegTensor<float>& maxVreg,
        __ubuf__ float* srcUb, uint32_t kvSize, uint32_t kvStart,
        AscendC::Reg::RegTensor<int32_t>& qIndexVreg, float dScale, float cap,
        AscendC::Reg::MaskReg& pregTailN, int32_t left = 0, int32_t right = 0)
    {
        using namespace AscendC::Reg;
        RegTensor<float> src0Vreg, src1Vreg, src2Vreg, src3Vreg;
        RegTensor<float> max0Vreg, max1Vreg, max2Vreg, max3Vreg;
        Duplicate(max0Vreg, -3.38953139e38f);
        Duplicate(max1Vreg, -3.38953139e38f);
        Duplicate(max2Vreg, -3.38953139e38f);
        Duplicate(max3Vreg, -3.38953139e38f);
        constexpr uint32_t kvBegin = KvStripe<ValidRange>::BEGIN;
        constexpr uint32_t kvCount = KvStripe<ValidRange>::END - kvBegin;
        for (uint32_t i = 0; i < kvCount / 4; ++i) {
            LoadScoreDN<HasMask, FullKv, ScaleScore>(src0Vreg, srcUb, kvBegin + i * 4, kvSize, kvStart,
                qIndexVreg, dScale, cap, pregTailN, left, right);
            LoadScoreDN<HasMask, FullKv, ScaleScore>(src1Vreg, srcUb, kvBegin + i * 4 + 1, kvSize, kvStart,
                qIndexVreg, dScale, cap, pregTailN, left, right);
            LoadScoreDN<HasMask, FullKv, ScaleScore>(src2Vreg, srcUb, kvBegin + i * 4 + 2, kvSize, kvStart,
                qIndexVreg, dScale, cap, pregTailN, left, right);
            LoadScoreDN<HasMask, FullKv, ScaleScore>(src3Vreg, srcUb, kvBegin + i * 4 + 3, kvSize, kvStart,
                qIndexVreg, dScale, cap, pregTailN, left, right);
            Max(max0Vreg, max0Vreg, src0Vreg, pregTailN);
            Max(max1Vreg, max1Vreg, src1Vreg, pregTailN);
            Max(max2Vreg, max2Vreg, src2Vreg, pregTailN);
            Max(max3Vreg, max3Vreg, src3Vreg, pregTailN);
        }
        Max(max0Vreg, max0Vreg, max2Vreg, pregTailN);
        Max(max1Vreg, max1Vreg, max3Vreg, pregTailN);
        Max(maxVreg, max0Vreg, max1Vreg, pregTailN);
    }

    template <bool IsUpdate, bool HasMask, bool FullKv, KvRange ValidRange = KvRange::FULL>
    __simd_vf__ static inline void ComputeOnlineSoftmaxDN(__ubuf__ float* srcUb,
        __ubuf__ ElementOutput* expUb, __ubuf__ float* lastMaxUb, __ubuf__ float* lastSumUb,
        __ubuf__ float* expMaxUb, __ubuf__ uint8_t* gatherIndexUb, uint32_t rows,
        uint32_t rowStart, uint32_t qSize, int32_t causalStart, uint32_t kvStart, uint32_t kvSize,
        float dScale, float cap, bool positiveScale, int32_t left = 0, int32_t right = 0)
    {
        using namespace AscendC::Reg;
        constexpr uint32_t mAlignedTile = SM_COL_MAX_ELEM_NUM / 4;
        constexpr uint32_t blockStride = mAlignedTile + 1;
        constexpr uint32_t kvBegin = KvStripe<ValidRange>::BEGIN;
        constexpr uint32_t kvEnd = KvStripe<ValidRange>::END;
        static constexpr CastTrait castTraitZeroRound = {
            RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr CastTrait castTraitOneRound = {
            RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr CastTrait castTraitTwoRound = {
            RegLayout::TWO, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr CastTrait castTraitThreeRound = {
            RegLayout::THREE, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<float>(rows);
        MaskReg pregFullB8 = CreateMask<uint8_t, MaskPattern::ALL>();
        RegTensor<float> src0Vreg, src1Vreg, src2Vreg, src3Vreg;
        RegTensor<float> max0Vreg;
        RegTensor<float> exp0Vreg, exp1Vreg, exp2Vreg, exp3Vreg;
        RegTensor<float> sum0Vreg, sum1Vreg, sum2Vreg, sum3Vreg;
        RegTensor<float> lastMaxVreg, lastSumVreg, expMaxVreg, updateSumVreg;
        RegTensor<int32_t> qIndexVreg;
        RegTensor<ElementOutput> cast0Vreg, packedPVreg;
        RegTensor<uint8_t> gatherIndexVreg;

        if constexpr (HasMask) {
            Arange(qIndexVreg, static_cast<int32_t>(rowStart));
            if (qSize == 1) {
                Duplicate(qIndexVreg, causalStart);
            } else {
                RegTensor<int32_t> divisorVreg, headVreg, productVreg;
                Duplicate(divisorVreg, static_cast<int32_t>(qSize));
                Div(headVreg, qIndexVreg, divisorVreg, pregFull);
                Mul(productVreg, headVreg, divisorVreg, pregFull);
                Sub(qIndexVreg, qIndexVreg, productVreg, pregFull);
                Adds(qIndexVreg, qIndexVreg, causalStart, pregFull);
            }
        }
        if constexpr ((HasMask || !FullKv) && !HAS_WINDOW) {
            RegTensor<int32_t> kvEndVreg;
            Duplicate(kvEndVreg, static_cast<int32_t>(kvStart + kvSize) - 1);
            if constexpr (HasMask) {
                Min(qIndexVreg, qIndexVreg, kvEndVreg, pregFull);
            } else {
                Duplicate(qIndexVreg, static_cast<int32_t>(kvStart + kvSize) - 1);
            }
        }
        // A positive scale and softcap are monotone: scale the reduced max once.
        if (positiveScale) {
            ComputeMaxDN<HasMask, FullKv, false, ValidRange>(max0Vreg, srcUb, kvSize, kvStart,
                qIndexVreg, dScale, cap, pregTailN, left, right);
            Muls(max0Vreg, max0Vreg, dScale, pregTailN);
            Maxs(max0Vreg, max0Vreg, -3.38953139e38f, pregTailN);
            if constexpr (HAS_SOFTCAP) {
                Softcap(max0Vreg, cap, pregTailN);
            }
        } else {
            ComputeMaxDN<HasMask, FullKv, true, ValidRange>(max0Vreg, srcUb, kvSize, kvStart,
                qIndexVreg, dScale, cap, pregTailN, left, right);
        }
        if constexpr (IsUpdate) {
            LoadAlign(lastMaxVreg, lastMaxUb);
            LoadAlign(lastSumVreg, lastSumUb);
            Max(max0Vreg, max0Vreg, lastMaxVreg, pregTailN);
            FusedExpSub(expMaxVreg, lastMaxVreg, max0Vreg, pregTailN);
            Mul(updateSumVreg, expMaxVreg, lastSumVreg, pregTailN);
            StoreAlign(expMaxUb, expMaxVreg, pregFull);
        }
        StoreAlign(lastMaxUb, max0Vreg, pregFull);
        // Keep the online max unchanged; P and its FP32 sum share the factor 256.
        Adds(max0Vreg, max0Vreg, -5.5451774444795623f, pregTailN);
        Duplicate(sum0Vreg, 0.0f);
        Duplicate(sum1Vreg, 0.0f);
        Duplicate(sum2Vreg, 0.0f);
        Duplicate(sum3Vreg, 0.0f);
        LoadAlign(gatherIndexVreg, gatherIndexUb);
        __ubuf__ ElementOutput* exp0Ub = expUb;

        // Cast four FP32 exp vectors into byte lanes, then Gather into the PV nZ stripes.
        for (uint32_t i = 0; i < mAlignedTile; ++i) {
            if constexpr (kvBegin <= 0U && kvEnd > 0U) {
                LoadScoreDN<HasMask, FullKv>(src0Vreg, srcUb, i, kvSize, kvStart,
                    qIndexVreg, dScale, cap, pregTailN, left, right);
            }
            if constexpr (kvBegin <= 64U && kvEnd > 64U) {
                LoadScoreDN<HasMask, FullKv>(src1Vreg, srcUb, i + mAlignedTile * 1, kvSize, kvStart,
                    qIndexVreg, dScale, cap, pregTailN, left, right);
            }
            if constexpr (kvBegin <= 128U && kvEnd > 128U) {
                LoadScoreDN<HasMask, FullKv>(src2Vreg, srcUb, i + mAlignedTile * 2, kvSize, kvStart,
                    qIndexVreg, dScale, cap, pregTailN, left, right);
            }
            if constexpr (kvBegin <= 192U && kvEnd > 192U) {
                LoadScoreDN<HasMask, FullKv>(src3Vreg, srcUb, i + mAlignedTile * 3, kvSize, kvStart,
                    qIndexVreg, dScale, cap, pregTailN, left, right);
            }
            if constexpr (kvBegin <= 0U && kvEnd > 0U) {
                FusedExpSub(exp0Vreg, src0Vreg, max0Vreg, pregTailN);
            } else {
                Duplicate(exp0Vreg, 0.0f);
            }
            if constexpr (kvBegin <= 64U && kvEnd > 64U) {
                FusedExpSub(exp1Vreg, src1Vreg, max0Vreg, pregTailN);
            } else {
                Duplicate(exp1Vreg, 0.0f);
            }
            if constexpr (kvBegin <= 128U && kvEnd > 128U) {
                FusedExpSub(exp2Vreg, src2Vreg, max0Vreg, pregTailN);
            } else {
                Duplicate(exp2Vreg, 0.0f);
            }
            if constexpr (kvBegin <= 192U && kvEnd > 192U) {
                FusedExpSub(exp3Vreg, src3Vreg, max0Vreg, pregTailN);
            } else {
                Duplicate(exp3Vreg, 0.0f);
            }
            if constexpr (kvBegin <= 0U && kvEnd > 0U) {
                Add(sum0Vreg, sum0Vreg, exp0Vreg, pregTailN);
            }
            if constexpr (kvBegin <= 64U && kvEnd > 64U) {
                Add(sum1Vreg, sum1Vreg, exp1Vreg, pregTailN);
            }
            if constexpr (kvBegin <= 128U && kvEnd > 128U) {
                Add(sum2Vreg, sum2Vreg, exp2Vreg, pregTailN);
            }
            if constexpr (kvBegin <= 192U && kvEnd > 192U) {
                Add(sum3Vreg, sum3Vreg, exp3Vreg, pregTailN);
            }
            if constexpr (kvBegin == 0U) {
                Cast<ElementOutput, float, castTraitZeroRound>(cast0Vreg, exp0Vreg, pregTailN);
            }
            if constexpr (kvBegin <= 64U && kvEnd > 64U) {
                if constexpr (kvBegin == 64U) {
                    Cast<ElementOutput, float, castTraitOneRound>(cast0Vreg, exp1Vreg, pregTailN);
                } else {
                    Cast<ElementOutput, float, castTraitOneRound>((RegTensor<ElementOutput>&)exp0Vreg,
                        exp1Vreg, pregTailN);
                }
            }
            if constexpr (kvBegin <= 128U && kvEnd > 128U) {
                if constexpr (kvBegin == 128U) {
                    Cast<ElementOutput, float, castTraitTwoRound>(cast0Vreg, exp2Vreg, pregTailN);
                } else {
                    Cast<ElementOutput, float, castTraitTwoRound>((RegTensor<ElementOutput>&)exp1Vreg,
                        exp2Vreg, pregTailN);
                }
            }
            if constexpr (kvEnd > 192U) {
                if constexpr (kvBegin == 192U) {
                    Cast<ElementOutput, float, castTraitThreeRound>(cast0Vreg, exp3Vreg, pregTailN);
                } else {
                    Cast<ElementOutput, float, castTraitThreeRound>((RegTensor<ElementOutput>&)exp2Vreg,
                        exp3Vreg, pregTailN);
                }
            }
            if constexpr (kvBegin == 0U && kvEnd > 64U) {
                Or((RegTensor<uint8_t>&)cast0Vreg, (RegTensor<uint8_t>&)cast0Vreg,
                    (RegTensor<uint8_t>&)exp0Vreg, pregFullB8);
            }
            if constexpr (kvBegin <= 64U && kvEnd > 192U) {
                Or((RegTensor<uint8_t>&)exp1Vreg, (RegTensor<uint8_t>&)exp1Vreg,
                    (RegTensor<uint8_t>&)exp2Vreg, pregFullB8);
                Or((RegTensor<uint8_t>&)cast0Vreg, (RegTensor<uint8_t>&)cast0Vreg,
                    (RegTensor<uint8_t>&)exp1Vreg, pregFullB8);
            } else if constexpr (kvBegin <= 64U && kvEnd > 128U) {
                Or((RegTensor<uint8_t>&)cast0Vreg, (RegTensor<uint8_t>&)cast0Vreg,
                    (RegTensor<uint8_t>&)exp1Vreg, pregFullB8);
            } else if constexpr (kvBegin == 128U && kvEnd > 192U) {
                Or((RegTensor<uint8_t>&)cast0Vreg, (RegTensor<uint8_t>&)cast0Vreg,
                    (RegTensor<uint8_t>&)exp2Vreg, pregFullB8);
            }
            Gather(packedPVreg, cast0Vreg, gatherIndexVreg);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                exp0Ub, packedPVreg, blockStride, 1, pregFullB8);
        }
        Add(sum0Vreg, sum0Vreg, sum2Vreg, pregFull);
        Add(sum1Vreg, sum1Vreg, sum3Vreg, pregFull);
        Add(sum0Vreg, sum0Vreg, sum1Vreg, pregFull);
        if constexpr (IsUpdate) {
            Add(sum0Vreg, updateSumVreg, sum0Vreg, pregTailN);
        }
        StoreAlign(lastSumUb, sum0Vreg, pregFull);
    }
};
} // namespace Catlass::Epilogue::Block
#endif
