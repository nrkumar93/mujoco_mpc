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

#include "gen3_hebi.h"

#include <cmath>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
  std::string Gen3Hebi::XmlPath() const {
    return GetModelPath("insat/gen3_hebi/task.xml");
  }
  std::string Gen3Hebi::Name() const { return "Gen3Hebi"; }

// ------- Residuals for planar_pusher task ------
//     load_pos: load should at goal position
// ------------------------------------------
  void Gen3Hebi::ResidualFn::Residual(const mjModel* model, const mjData* data,
                          double* residual) const {
    int idx = 0;

    // ---------- jpos ----------
    for (int i=0; i<model->nq; ++i)
    {
      residual[idx++] = data->qpos[i] - task_->parameters[i];
    }

    // ---------- jvel ----------
    for (int i=0; i<model->nv; ++i)
    {
      residual[idx++] = data->qvel[i];
    }

    // ---------- jacc ----------
//    for (int i=0; i<model->na; ++i)
//    {
//      residual[idx++] = data->qacc[i];
//    }

  // ---------- frc_con ----------
//    for (int i=0; i<model->nv; ++i)
//    {
//      residual[idx++] = 1.0/data->qfrc_constraint[i];
//    }

    // ---------- acc ----------
//    for (int i=0; i<model->nv; ++i)
//    {
////      residual[6+i] = data->qacc[i];
//      residual[6+i] = data->qfrc_smooth[i] + data->qfrc_constraint[i];
//    }

//    static int skip=0;
//    ++skip;
//    if (skip%10000 == 0)
//    {
//      std::cout << skip << " " << "frc residue: ";
//      for (int i=0; i<model->nv; ++i)
//      {
//        std::cout << (int)residual[6+i] << "\t";
//      }
//      std::cout << std::endl;

//      std::cout << skip << " " << "qfrc_smooth: ";
//      for (int i=0; i<model->nv; ++i)
//      {
//        std::cout << (int)data->qfrc_smooth[i] << "\t";
//      }
//      std::cout << "qfrc_constraint: ";
//      for (int i=0; i<model->nv; ++i)
//      {
//        std::cout << (int)data->qfrc_constraint[i] << "\t";
//      }
//      std::cout << std::endl;
//    }

  }

  Vec3f Gen3Hebi::getEEPosition(const mjModel *model, const mjData *data) const
  {
    VecDf ee_pos(3);
    mju_copy(ee_pos.data(), data->xpos + 3*(model->nbody-1), 3);

    return ee_pos;
  }

}  // namespace mjpc
