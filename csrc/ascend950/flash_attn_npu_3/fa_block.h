/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef FAI_BLOCK_HPP
#define FAI_BLOCK_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

namespace Catlass::Epilogue {

template <bool HAS_SOFTCAP_>
struct EpilogueFAOnlineSoftmaxT {
    using ArchTag = Arch::Ascend950;
    static constexpr bool HAS_SOFTCAP = HAS_SOFTCAP_;
};

template <bool IS_DN_, bool HAS_SOFTCAP_, bool HAS_WINDOW_ = false>
struct EpilogueFAQuantOnlineSoftmax {
    using ArchTag = Arch::Ascend950;
    static constexpr bool IS_DN = IS_DN_;
    static constexpr bool HAS_WINDOW = HAS_WINDOW_;
    static constexpr bool HAS_SOFTCAP = HAS_SOFTCAP_;
};

// Byte offsets shared by the kernel, OnlineSoftmax and RescaleO.
struct UBufTileHelper {
    uint32_t sStartOffset, pStartOffset, loStartOffset, goStartOffset;
    uint32_t lmStartOffset, gmStartOffset, dmStartOffset, llStartOffset, glStartOffset;
    uint32_t lseStartOffset, indexStartOffset, totalBytes;
    uint32_t sBufferBytes, pBufferBytes, loBufferBytes;
    uint32_t otmpStages;
};

} // namespace Catlass::Epilogue

#endif // FAI_BLOCK_HPP
