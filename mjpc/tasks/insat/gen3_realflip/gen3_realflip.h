// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MJPC_TASKS_Gen3RealFlip_Gen3RealFlip_H_
#define MJPC_TASKS_Gen3RealFlip_Gen3RealFlip_H_

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/common/EigenTypes.h"
#include "mjpc/task.h"

namespace mjpc {
  class Gen3RealFlip : public Task {
  public:
    Gen3RealFlip() : residual_(this) {}
    std::string Name() const override;
    std::string XmlPath() const override;
    class ResidualFn : public mjpc::BaseResidualFn {
    public:
      explicit ResidualFn(const Gen3RealFlip* task) : mjpc::BaseResidualFn(task) {}
      void Residual(const mjModel* model, const mjData* data,
                    double* residual) const override;
      VecDf NetEffort(const mjModel* model, const mjData* data) const;
    };

    Vec3f getEEPosition(const mjModel* model, const mjData* data) const;
    Vec4f getEERotation(const mjModel* model, const mjData* data) const;
  protected:
    std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
      return std::make_unique<ResidualFn>(this);
    }
    ResidualFn* InternalResidual() override { return &residual_; }

  private:
    ResidualFn residual_;
  };
}  // namespace mjpc

#endif  // MJPC_TASKS_Gen3RealFlip_Gen3RealFlip_H_
