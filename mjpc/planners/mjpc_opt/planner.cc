/*
  * Copyright (c) 2023, Ramkumar Natarajan
  * All rights reserved.
  *
  * Redistribution and use in source and binary forms, with or without
  * modification, are permitted provided that the following conditions are met:
  *
  *     * Redistributions of source code must retain the above copyright
  *       notice, this list of conditions and the following disclaimer.
  *     * Redistributions in binary form must reproduce the above copyright
  *       notice, this list of conditions and the following disclaimer in the
  *       documentation and/or other materials provided with the distribution.
  *     * Neither the name of the Carnegie Mellon University nor the names of its
  *       contributors may be used to endorse or promote products derived from
  *       this software without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
  * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
  * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
  * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
  * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  * POSSIBILITY OF SUCH DAMAGE.

 */
/*!
 * \file planner.cc 
 * \author Ram Natarajan (rnataraj@cs.cmu.edu)
 * \date 4/7/23
*/

#include <mjpc/planners/mjpc_opt/planner.h>

namespace mjpc {

MjpcOpt::MjpcOpt(mjModel* model, const mjData* data) : pool_(1) {
  task_.Reset(model, data);

  // ----- iLQG planner ----- //
  planner_.Initialize(model, task_);
  planner_.Allocate();
  planner_.Reset(kMaxTrajectoryHorizon);

}

iLQGPolicy MjpcOpt::optimize(VecDf &low1, VecDf &low2, double dt, int horizon) {

  VecDf state1(planner_.model->nq + planner_.model->nv + planner_.model->na);
  VecDf state2(planner_.model->nq + planner_.model->nv + planner_.model->na);

  state1.head(planner_.model->nq) = low1;
  state1.tail(planner_.model->nv + planner_.model->na).setZero();
  task_.setStart(state1); // Set the start state

  state2.head(planner_.model->nq) = low2;
  state2.tail(planner_.model->nv + planner_.model->na).setZero();
  task_.setGoal(state2); // Set the goal state

  int steps =
      mju_max(mju_min(horizon / dt + 1, kMaxTrajectoryHorizon), 1);
  planner_.model->opt.timestep = dt;

}


}
