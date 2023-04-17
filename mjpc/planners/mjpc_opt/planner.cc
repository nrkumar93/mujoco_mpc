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
  state_len_ = model->nq + model->nv + model->na;

  task_.Reset(model, data);

  // ----- iLQG planner ----- //
  planner_.Initialize(model, task_);
  planner_.Allocate();
  planner_.Reset(kMaxTrajectoryHorizon);

}

iLQGPolicy MjpcOpt::optimize(int horizon) {
  planner_.SetState(task_.start_);

  // ---- optimize ----- //
  MatDf prev_soln(horizon, state_len_);
  for (int i = 0; i < params_.max_iter_; i++) {
    planner_.OptimizePolicy(horizon, pool_);

    MatDf soln =
        Eigen::Map<MatDf>(planner_.candidate_policy[0].trajectory.states.data(), state_len_, horizon);

    /// Check for convergence
    if (i > 0) {
      if ((prev_soln-soln).norm() < params_.conv_thresh_) {
        prev_soln = soln;
        break;
      }
      prev_soln = soln;
    }
  }

  return planner_.candidate_policy[0];
}

iLQGPolicy MjpcOpt::optimize(const VecDf &low1, const VecDf &low2, double dt, double T) {
  VecDf aux1 = VecDf(planner_.model->nv + planner_.model->na).setZero();
  VecDf aux2 = VecDf(planner_.model->nv + planner_.model->na).setZero();

  return optimize(low1, low2, aux1, aux2, dt, T);
}

iLQGPolicy MjpcOpt::optimize(const VecDf &low1,
                             const VecDf &low2,
                             const VecDf &aux1,
                             const VecDf &aux2,
                             double dt, double T) {
  planner_.Reset(kMaxTrajectoryHorizon);

  VecDf state1(planner_.model->nq + planner_.model->nv + planner_.model->na);
  VecDf state2(planner_.model->nq + planner_.model->nv + planner_.model->na);

  state1.head(planner_.model->nq) = low1;
  state1.tail(planner_.model->nv + planner_.model->na) = aux1;

  state2.head(planner_.model->nq) = low2;
  state2.tail(planner_.model->nv + planner_.model->na) = aux2;

  task_.setStart(state1); // Set the start state
  task_.setGoal(state2); // Set the goal state

  planner_.model->opt.timestep = dt;

  int horizon =
      mju_max(mju_min(T / dt + 1, kMaxTrajectoryHorizon), 1);

  return optimize(horizon);
}

iLQGPolicy MjpcOpt::warmOptimize(const VecDf &low1,
                                 const VecDf &low2,
                                 const mjpc::iLQGPolicy &pol1,
                                 const mjpc::iLQGPolicy &pol2,
                                 double dt) {
  int pol2_len = pol2.trajectory.horizon;
  Eigen::Map<const VecDf> traj_st(pol1.trajectory.states.data(), state_len_);
  Eigen::Map<const VecDf> traj_go(pol2.trajectory.states.data() + ((pol2_len-1)*state_len_), state_len_);

  VecDf aux1 = traj_st.tail(planner_.model->nv + planner_.model->na);
  VecDf aux2 = traj_go.tail(planner_.model->nv + planner_.model->na);

  return warmOptimize(low1, low2, aux1, aux2, pol1, pol2, dt);
}

iLQGPolicy MjpcOpt::warmOptimize(const VecDf &low1,
                                 const VecDf &low2,
                                 const VecDf &aux1,
                                 const VecDf &aux2,
                                 const mjpc::iLQGPolicy &pol1,
                                 const mjpc::iLQGPolicy &pol2,
                                 double dt) {
  planner_.Reset(kMaxTrajectoryHorizon);

  VecDf state1(planner_.model->nq + planner_.model->nv + planner_.model->na);
  VecDf state2(planner_.model->nq + planner_.model->nv + planner_.model->na);

  state1.head(planner_.model->nq) = low1;
  state1.tail(planner_.model->nv + planner_.model->na) = aux1;

  state2.head(planner_.model->nq) = low2;
  state2.tail(planner_.model->nv + planner_.model->na) = aux2;

  task_.setStart(state1); /// Set the start state
  task_.setGoal(state2); /// Set the goal state

  auto pol = pol1;
  pol.Concatenate(pol2); /// Merge the policies
  planner_.model->opt.timestep = dt; /// Set the timestep
  int horizon = pol.trajectory.horizon; /// Get the horizon of concatenated policy

  planner_.policy = pol; /// Initial guess
  return optimize(horizon);
}


}
