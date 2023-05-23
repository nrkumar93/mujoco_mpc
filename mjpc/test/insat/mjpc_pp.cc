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
 * \file mjpc_pp.cc 
 * \author Ram Natarajan (rnataraj@cs.cmu.edu)
 * \date 4/24/23
*/

#include "mjpc/cmake-build-release/_deps/mujoco-src/include/mujoco/mujoco.h"
#include <cassert>
#include "mjpc/planners/ilqg/planner.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/test/testdata/particle_residual.h"
#include "mjpc/tasks/insat/gen3_hebi/gen3_hebi.h"
#include "mjpc/tasks/insat/gen3_flip/gen3_flip.h"
#include "mjpc/threadpool.h"
#include "mjpc/array_safety.h"
#include "mjpc/common/EigenTypes.h"

namespace mjpc {
namespace {

// model
mjModel* model;

// state
State state;

// task
//Gen3Hebi task;
Gen3Flip task;

// sensor
extern "C" {
void sensor(const mjModel* m, mjData* d, int stage);
}

// sensor callback
void sensor(const mjModel* model, mjData* data, int stage) {
  if (stage == mjSTAGE_ACC) {
    task.Residual(model, data, data->sensordata);
  }
}



}  // namespace
}  // namespace mjpc

mjModel* LoadTestModel(std::string_view path) {
  // filename
  char filename[1024];
//  const std::string path_str = absl::StrCat("../../../mjpc/test/testdata/", path);;
  const std::string path_str = absl::StrCat("", path);
  mujoco::util_mjpc::strcpy_arr(filename, path_str.c_str());

  // load model
  char loadError[1024] = "";
  mjModel* model = mj_loadXML(filename, nullptr, loadError, 1000);
  if (loadError[0]) std::cerr << "load error: " << loadError << '\n';

  return model;
}

MatDf linInterp(const VecDf& p1, const VecDf& p2, int N)
{
  MatDf traj(N, p1.size());
  for (int i=0; i<N; ++i)
  {
    double j = i/static_cast<double>(N);
    traj.row(i) = p1*(1-j) + p2*j;
  }
  traj.bottomRows(1) = p2.transpose();

  return traj;
}

MatDf upsample(const MatDf& traj, double dx)
{
  MatDf traj_up(0, traj.cols());
  for (int i=0; i<traj.rows()-1; ++i)
  {
    VecDf p1 = traj.row(i);
    VecDf p2 = traj.row(i+1);

    int N = static_cast<int>((p2-p1).norm()/dx);
    traj_up.conservativeResize(traj_up.rows()+N, traj.cols());
    traj_up.bottomRows(N) = linInterp(p1, p2, N);
  }

  return traj_up;
}



// test iLQG planner on particle task
int main() {
  using namespace mjpc;

  // load model
//  model = LoadTestModel("particle_task.xml");
//  model = LoadTestModel("/home/gaussian/cmu_ri_phd/phd_research/mujoco_mpc/mjpc/tasks/insat/gen3_hebi/task.xml");
  model = LoadTestModel("/home/gaussian/cmu_ri_phd/phd_research/mujoco_mpc/mjpc/tasks/insat/gen3_flip/task.xml");
  task.Reset(model);

  // create data
  mjData* data = mj_makeData(model);

  // set data
  mj_forward(model, data);

  // sensor callback
  mjcb_sensor = sensor;

  // ----- state ----- //
  // State state;
  state.Initialize(model);
  state.Allocate(model);
  state.Reset();
  state.Set(model, data);

  // ----- iLQG planner ----- //
  iLQGPlanner planner;
  planner.Initialize(model, task);
  planner.Allocate();
  planner.Reset(kMaxTrajectoryHorizon);

  // ----- settings ----- //
  int iterations = 50;
  double horizon = 4.1;
//  double horizon = 0.05;
  double timestep = 0.01;
  bool upsampling = false;
  int steps =
      mju_max(mju_min(horizon / timestep + 1, kMaxTrajectoryHorizon), 1);
  model->opt.timestep = timestep;

  MatDf path = loadEigenFromFile<MatDf>("/home/gaussian/cmu_ri_phd/phd_research/mujoco_mpc/mjpc/pinsat/logs/pp/traj.txt", ' ');
  VecDf init_vel(model->nv);
  init_vel.setZero();
//  init_vel << -0.226081232795585, -0.493956309040456, -0.499425388612798, 0.0313000826490541, -0.336311513861378, 0.244498225608607, 0.761775556154479;

  if (upsampling)
    path = upsample(path, 0.2);

  // threadpool
  ThreadPool pool(1);

  double *mocap;
  double *userdata;
  iLQGPolicy warm_policy;

//  MatDf op_traj(steps*(path.rows()-1), model->nq);
  MatDf op_traj(steps*(path.rows()-1), 1 + model->nq + model->nv + model->nu);
  op_traj.setZero();
  VecDf s1(model->nq + model->nv);
  VecDf s2(model->nq + model->nv);
  s1.setZero();
  s2.setZero();
  for (int i=0; i<path.rows()-1; ++i)
  {
    s1.head(model->nq) = path.row(i);
    s2.head(model->nq) = path.row(i+1);

    if (i==0)
    {
      s1.tail(model->nv) = init_vel;
    }

    state.Set(model, data, s1.data(), mocap, userdata, timestep);

    for (int j=0; j<model->nq; ++j)
    {
      task.parameters[j] = s2(j);
      std::cout << task.parameters[j] << "\t";
    }
    std::cout << std::endl;

    // set state
    planner.SetState(state);
    // ---- optimize ----- //
    VecDf prev_soln(model->nq + model->nv);
    VecDf soln_end(model->nq + model->nv);
    VecDf soln_st(model->nq + model->nv);
    prev_soln.setZero();
    for (int k = 0; k < iterations; k++) {
      planner.OptimizePolicy(steps, pool);

      for (int l=0; l<(model->nq + model->nv); ++l)
      {
        soln_st(l) = planner.candidate_policy[0]
            .trajectory.states[l];
        soln_end(l) = planner.candidate_policy[0]
            .trajectory.states[(steps - 1) * (model->nq + model->nv)+l];
      }

//      if ((prev_soln-soln_end).norm() < 1e-3)
//      {
//        break;
//      }
//      prev_soln = soln_end;
    }
    s1.tail(model->nv) = soln_end.tail(model->nv);
    std::cout << "vel at start of " << i << "th traj: " << soln_st.tail(model->nv).transpose() << std::endl;
    std::cout << "vel at end of " << i << "th traj: " << soln_end.tail(model->nv).transpose() << std::endl;

    if (i==0)
      warm_policy = planner.candidate_policy[0];
    else
      warm_policy.Concatenate(planner.candidate_policy[0]);

    for (int j=0; j<steps; ++j)
    {
      op_traj(i*steps+j, 0) = planner.candidate_policy[0].trajectory.times[j];
//      for (int l=0; l<(model->nq); ++l)
      for (int l=0; l<(model->nq + model->nv); ++l)
      {
        op_traj(i*steps+j, 1+l) = planner.candidate_policy[0]
            .trajectory.states[j * (model->nq + model->nv)+l];
//        op_traj(i*steps+j, l) = planner.candidate_policy[0]
//            .trajectory.states[j * (model->nq + model->nv + model->na)+l];
      }
      for (int l=0; l<(model->nu); ++l)
      {
        op_traj(i*steps+j, 1+(model->nq + model->nv)+l) = planner.candidate_policy[0]
            .trajectory.actions[j * (model->nu)+l];
      }
    }

    std::string op_traj_file = "/home/gaussian/cmu_ri_phd/phd_research/mujoco_mpc/mjpc/pinsat/logs/pp/op_traj.txt";
    writeEigenToFile(op_traj_file, op_traj);
  }


  // ----- iLQG planner WARM ----- //
  // ---- warm optimize ----- //
  std::cout << std::endl;
  std::cout << "NOW WARM STARTING" << std::endl;
  std::cout << std::endl;

  iLQGPlanner warm_planner;
  warm_planner.Initialize(model, task);
  warm_planner.Allocate();
  warm_planner.Reset(warm_policy.trajectory.horizon);

  VecDf full_state(model->nq + model->nv);
  full_state.head(model->nq) = path.row(0);
  full_state.tail(model->nv) = init_vel;

  state.Reset();
  state.Set(model, data);
  state.Set(model, data, full_state.data(), mocap, userdata, timestep);
  std::cout << "start state: " << path.row(0) << std::endl;

  auto goal = path.row(path.rows()-1);
  for (int j=0; j<model->nq; ++j)
  {
    task.parameters[j] = goal(j);
    std::cout << task.parameters[j] << "\t";
  }
  std::cout << std::endl;

  warm_planner.SetState(state);

  warm_planner.policy = warm_policy;
  for (int i = 0; i < iterations*2; i++) {
//  for (int i = 0; i < 10; i++) {
    warm_planner.OptimizePolicy(warm_policy.trajectory.horizon, pool);
  }

  op_traj.setZero();
  for (int j=0; j<warm_planner.candidate_policy[0].trajectory.horizon; ++j)
  {
    op_traj(j, 0) = planner.candidate_policy[0].trajectory.times[j];
//    for (int l=0; l<(model->nq); ++l)
    for (int l=0; l<(model->nq + model->nv); ++l)
    {
      op_traj(j, 1+l) = warm_planner.candidate_policy[0]
          .trajectory.states[j * (model->nq + model->nv)+l];
//      op_traj(j, l) = planner.candidate_policy[0]
//          .trajectory.states[j * (model->nq + model->nv + model->na)+l];
    }
    for (int l=0; l<(model->nu); ++l)
    {
      op_traj(j, 1+(model->nq + model->nv)+l) = warm_planner.candidate_policy[0]
          .trajectory.actions[j * (model->nu)+l];
    }
  }

  std::string op_traj_file = "/home/gaussian/cmu_ri_phd/phd_research/mujoco_mpc/mjpc/pinsat/logs/pp/op_traj.txt";
  writeEigenToFile(op_traj_file, op_traj);


// delete data
  mj_deleteData(data);

// delete model
  mj_deleteModel(model);
}
