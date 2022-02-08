// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../output_process.h"
#include "cpu_op.h"

namespace oidn {

  class CPUOutputProcess : public CPUOp, public OutputProcess
  {
  public:
    CPUOutputProcess(const Ref<CPUDevice>& device, const OutputProcessDesc& desc);

    void run() override;
  };

} // namespace oidn
