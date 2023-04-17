//  * Copyright (c) 2023, Ramkumar Natarajan
//  * All rights reserved.
//  *
//  * Redistribution and use in source and binary forms, with or without
//  * modification, are permitted provided that the following conditions are met:
//  *
//  *     * Redistributions of source code must retain the above copyright
//  *       notice, this list of conditions and the following disclaimer.
//  *     * Redistributions in binary form must reproduce the above copyright
//  *       notice, this list of conditions and the following disclaimer in the
//  *       documentation and/or other materials provided with the distribution.
//  *     * Neither the name of the Carnegie Mellon University nor the names of its
//  *       contributors may be used to endorse or promote products derived from
//  *       this software without specific prior written permission.
//  *
//  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
//  * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  * POSSIBILITY OF SUCH DAMAGE.
//

/*!
 * \file MjpcOpt.cpp 
 * \author Ram Natarajan (rnataraj@cs.cmu.edu)
 * \date 4/16/23
*/

#include <planners/insat/opt/MjpcOpt.hpp>

namespace ps
{

  MjpcTraj MjpcOpt::optimize(const ps::InsatAction *act, const VecDf &s1, const VecDf &s2, int thread_id)
  {
    double max_time = distToMaxExecDuration((s1-s2).norm());
    auto policy = opt_.optimize(s1, s2, opt_params_.dt, max_time);

    MjpcTraj traj;
    if (!policy.trajectory.failure)
    {
      traj.traj_ = policy;
      traj.disc_traj_ = Eigen::Map<MatDf>(policy.trajectory.states.data(), opt_.state_len_, policy.trajectory.horizon);
      traj.story_ = "direct optimize between two states";
      return traj;
    }

    traj.story_ = "direct optimize failed";
    return traj;
  }

  MjpcTraj MjpcOpt::warmOptimize(const ps::InsatAction *act, const ps::TrajType &traj1, const ps::TrajType &traj2,
                                 int thread_id)
  {
    auto low1 = traj1.MjpcTraj::disc_traj_.leftCols(1);
    auto low2 = traj1.MjpcTraj::disc_traj_.rightCols(1);
    auto policy = opt_.warmOptimize(low1, low2, traj1.MjpcTraj::traj_, traj2.MjpcTraj::traj_, opt_params_.dt);

    MjpcTraj traj;
    if (!policy.trajectory.failure)
    {
      traj.traj_ = policy;
      traj.disc_traj_ = Eigen::Map<MatDf>(policy.trajectory.states.data(), opt_.state_len_, policy.trajectory.horizon);
      traj.story_ = "warm optimize with two trajectories";
      return traj;
    }

    traj.story_ = "warm optimize failed";
    return traj;
  }

}