// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(OIDN_HIP)
  #include <hip/hip_runtime.h>
#endif
#include "../device.h"

struct miopenHandle;
typedef struct miopenHandle* miopenHandle_t;

namespace oidn {

#if defined(OIDN_HIP)
  namespace
  {
    // Helper functions for kernel execution
    template<typename Ty, typename Tx, typename F>
    __global__ void runHIPKernel(Ty Dy, Tx Dx, const F f)
    {
      const Tx x = Tx(blockDim.x * blockIdx.x + threadIdx.x);
      const Ty y = Ty(blockDim.y * blockIdx.y + threadIdx.y);
      if (y < Dy && x < Dx)
        f(y, x);
    }

    template<typename Tz, typename Ty, typename Tx, typename F>
    __global__ void runHIPKernel(Tz Dz, Ty Dy, Tx Dx, const F f)
    {
      const Tx x = Tx(blockDim.x * blockIdx.x + threadIdx.x);
      const Ty y = Ty(blockDim.y * blockIdx.y + threadIdx.y);
      const Tz z = Tz(blockDim.z * blockIdx.z + threadIdx.z);
      if (z < Dz && y < Dy && x < Dx)
        f(z, y, x);
    }
  }

  void checkError(hipError_t error);
#endif

  class HIPDevice final : public Device
  { 
  public:
    ~HIPDevice();

    OIDN_INLINE miopenHandle_t getMIOpenHandle() const { return miopenHandle; }

    void wait() override;

    // Ops
    std::shared_ptr<Conv> newConv(const ConvDesc& desc) override;
    std::shared_ptr<Pool> newPool(const PoolDesc& desc) override;
    std::shared_ptr<Upsample> newUpsample(const UpsampleDesc& desc) override;
    std::shared_ptr<InputProcess> newInputProcess(const InputProcessDesc& desc) override;
    std::shared_ptr<OutputProcess> newOutputProcess(const OutputProcessDesc& desc) override;

    void imageCopy(const Image& src, const Image& dst) override;

    // Memory
    void* malloc(size_t byteSize, Storage storage) override;
    void free(void* ptr, Storage storage) override;
    void memcpy(void* dstPtr, const void* srcPtr, size_t byteSize) override;

  #if defined(OIDN_HIP)
    // Runs a kernel on the device
    template<typename Ty, typename Tx, typename F>
    OIDN_INLINE void runKernel(const Ty& Dy, const Tx& Dx, const F& f)
    {
      const dim3 blockDim(32, 32);
      const dim3 gridDim(ceil_div(Dx, blockDim.x), ceil_div(Dy, blockDim.y));

      runHIPKernel<<<gridDim, blockDim>>>(Dy, Dx, f);
      checkError(hipGetLastError());
    }

    template<typename Tz, typename Ty, typename Tx, typename F>
    OIDN_INLINE void runKernel(const Tz& Dz, const Ty& Dy, const Tx& Dx, const F& f)
    {
      const dim3 blockDim(32, 32, 1);
      const dim3 gridDim(ceil_div(Dx, blockDim.x), ceil_div(Dy, blockDim.y), ceil_div(Dz, blockDim.z));

      runHIPKernel<<<gridDim, blockDim>>>(Dz, Dy, Dx, f);
      checkError(hipGetLastError());
    }
  #endif

  protected:
    void init() override;
    void printInfo() override;

  private:
    miopenHandle_t miopenHandle;
  };

} // namespace oidn
