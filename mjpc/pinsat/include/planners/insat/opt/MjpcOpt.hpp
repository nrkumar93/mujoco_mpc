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
 * \file MjpcOpt.hpp 
 * \author Ram Natarajan (rnataraj@cs.cmu.edu)
 * \date 4/16/23
*/

#ifndef EPASE_MJPCOPT_HPP
#define EPASE_MJPCOPT_HPP

// PS
#include <common/Types.hpp>
#include <iostream>
#include <common/robots/Abb.hpp>

namespace ps
{

  class MjpcOpt
  {
  public:
    typedef mjpc::MjpcOpt OptType;

    // Robot
    typedef IRB1600 RobotParamsType;

    struct MjpcOptParams
    {
      enum class ConstraintMode
      {
        WAYPT = 0,
        CONTROLPT
      };

      MjpcOptParams();

      ConstraintMode constraint_mode_;

      double dt;

      double duration_cost_w_ = 1.0;
      double length_cost_w_ = 0.1;

      bool zero_vel_start_ = false;
      bool zero_vel_goal_ = false;

      /// Adaptive Mjpc optimization
      VecDf global_start_; /// For now assuming higher derivatives = 0
      VecDf global_goal_; /// For now assuming higher derivatives = 0
      double start_goal_dist_;
    };


    MjpcOpt(const InsatParams& insat_params,
            const RobotParamsType& robot_params,
            const MjpcOptParams& opt_params,
            ParamsType& search_params);

    void SetGoalChecker(std::function<bool(const StateVarsType&)> callback);

    void updateStartAndGoal(StateVarsType& start, StateVarsType& goal);

    bool isGoal(const VecDf& state) const;

    /// trajectory samplers with fixed time
    MatDf sampleTrajectory(const MjpcTraj::TrajInstanceType& traj, double dt) const;

    MatDf sampleTrajectory(const MjpcTraj& traj, double dt) const;

    /// trajectory samplers with adaptive time
    MatDf sampleTrajectory(const MjpcTraj::TrajInstanceType& traj) const;

    MatDf sampleTrajectory(const MjpcTraj& traj) const;

    void addDurationAndPathCost(OptType& opt) const;

    void addStateSpaceBounds(OptType& opt) const;

    double distToMaxExecDuration(double dist) const;

    /// non adaptive standard version
    MjpcTraj optimize(const InsatAction* act, const VecDf& s1, const VecDf& s2, int thread_id);

    MjpcTraj warmOptimize(const InsatAction* act, const TrajType& traj1, const TrajType & traj2, int thread_id);

    MjpcTraj warmOptimize(const InsatAction* act, const TrajType& traj, int thread_id);



    virtual double calculateCost(const TrajType& traj) const;

    virtual double calculateCost(const MatDf& traj) const;

    /// Params
    InsatParams insat_params_;
    IRB1600 robot_params_;
    MjpcOptParams opt_params_;
    ParamsType search_params_;

    /// Adaptive MjpcOpt
    std::function<double(const StateVarsType&)> goal_checker_;

    /// Optimizer handle
    mjpc::MjpcOpt opt_;

  };

}



#endif //EPASE_MJPCOPT_HPP
