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

#ifndef MJPC_MJPC_TASKS_PANDA_PANDA_H_
#define MJPC_MJPC_TASKS_PANDA_PANDA_H_

#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/common/EigenTypes.h"
#include <absl/random/random.h>

namespace mjpc {
class Panda : public Task {
 public:
  Panda() : gen_(0), residual_(this) {}
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Panda* task) : mjpc::BaseResidualFn(task) {}
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
    VecDf NetEffort(const mjModel* model, const mjData* data) const;
  };
  void TransitionLocked(mjModel* model, mjData* data) override;

//  absl::BitGen gen_;
  std::mt19937 gen_;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};
}  // namespace mjpc


#endif  // MJPC_MJPC_TASKS_PANDA_PANDA_H_
