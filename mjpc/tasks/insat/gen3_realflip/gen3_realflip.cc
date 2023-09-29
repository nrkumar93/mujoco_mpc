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

#include "gen3_realflip.h"

#include <cmath>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
  std::string Gen3RealFlip::XmlPath() const {
    return GetModelPath("insat/gen3_flip/task.xml");
  }
  std::string Gen3RealFlip::Name() const { return "Gen3RealFlip"; }

// ------- Residuals for planar_pusher task ------
//     load_pos: load should at goal position
// ------------------------------------------
  void Gen3RealFlip::ResidualFn::Residual(const mjModel* model,
                                          const mjData* data,
                                          double* residual) const {
    int idx = 0;

    // ---------- jpos ----------
    for (int i=0; i<model->nq; ++i)
    {
      residual[idx++] = task_->parameters[i] - data->qpos[i];
//      residual[idx++] = data->qpos[i] - parameters[i];
    }

//  auto ee_pos = getEEPosition(model, data);
//  auto ee_rot = getEERotation(model, data);
//  residual[idx++] = 1.3 - ee_pos[0];
//  residual[idx++] = 0.65 - ee_pos[1];
//  residual[idx++] = 0.421 - ee_pos[2];
//  residual[idx++] = -0.8556 - ee_rot[0];
//  residual[idx++] = -0.161 - ee_rot[1];
//  residual[idx++] = 0.482 - ee_rot[2];
//  residual[idx++] = -0.0912 - ee_rot[3];

    // ---------- jvel ----------
    for (int i=0; i<model->nv; ++i)
    {
      residual[idx++] = data->qvel[i];
    }

  // ---------- frc_con ----------
    auto effort = NetEffort(model, data);
    for (int i=0; i<model->nv; ++i)
    {
      residual[idx++] = effort(i);
    }

//    double* target = mjpc::SensorByName(model, data, "target");
//    double* object = mjpc::SensorByName(model, data, "object");
//    mju_sub(residual + model->nv, object, target, 3);

  }

  VecDf Gen3RealFlip::ResidualFn::NetEffort(const mjModel *model, const mjData *data) const {

    VecDf effort(model->nq);
    mju_add(effort.data(), data->qfrc_smooth, data->qfrc_constraint, model->nq);
    for (int i=0; i<model->nq; ++i)
    {
      if ((effort(i) > 0 && data->qfrc_smooth[i] < 0) ||
          (effort(i) < 0 && data->qfrc_smooth[i] > 0) ||
          std::fabs(effort(i)) < 1e-6) {
        effort(i) = 0;
      }
//      else {
//        effort(i) = data->qfrc_smooth[i];
//      }
    }
    return effort;
  }

  Vec3f Gen3RealFlip::getEEPosition(const mjModel *model, const mjData *data) const
  {
    VecDf ee_pos(3);
    mju_copy(ee_pos.data(), data->xpos + 3*(model->nbody-1), 3);

    return ee_pos;
  }

  Vec4f Gen3RealFlip::getEERotation(const mjModel *model, const mjData *data) const
  {
    VecDf ee_rot(4);
    mju_copy(ee_rot.data(), data->xquat + 4 * (model->nbody - 1), 4);

    return ee_rot;
  }


}  // namespace mjpc
