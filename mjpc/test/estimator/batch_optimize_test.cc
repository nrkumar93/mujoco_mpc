// Copyright 2023 DeepMind Technologies Limited
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

#include <vector>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include "gtest/gtest.h"
#include "mjpc/estimators/batch.h"
#include "mjpc/test/load.h"
#include "mjpc/test/simulation.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

TEST(BatchOptimize, Particle2D) {
  // load model
  mjModel* model = LoadTestModel("estimator/particle/task.xml");
  mjData* data = mj_makeData(model);

  // discrete inverse dynamics
  model->opt.enableflags |= mjENBL_INVDISCRETE;

  // dimensions
  int nq = model->nq, nv = model->nv, ns = model->nsensordata;

  // ----- simulate ----- //
  int T = 10;
  Simulation sim(model, T);
  auto controller = [](double* ctrl, double time) {
    ctrl[0] = mju_sin(10 * time);
    ctrl[1] = 10 * mju_cos(10 * time);
  };
  sim.Rollout(controller);

  // ----- estimator ----- //

  // initialize
  Batch estimator(model, T);
  mju_copy(estimator.configuration.Data(), sim.qpos.Data(), nq * T);
  mju_copy(estimator.configuration_previous.Data(), sim.qpos.Data(), nq * T);
  mju_copy(estimator.force_measurement.Data(), sim.qfrc_actuator.Data(),
           nv * T);
  mju_copy(estimator.sensor_measurement.Data(), sim.sensor.Data(), ns * T);

  // ----- random perturbation ----- //

  // randomly perturb
  for (int t = 0; t < T; t++) {
    // unpack
    double* q = estimator.configuration.Data() + t * nq;

    // add noise
    for (int i = 0; i < nq; i++) {
      absl::BitGen gen_;
      q[i] += 0.001 * absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
  }

  // set weights
  estimator.scale_prior = 1.0e-3;

  // set process noise
  std::fill(estimator.noise_process.begin(), estimator.noise_process.end(),
            1.0);

  // set sensor noise
  std::fill(estimator.noise_sensor.begin(), estimator.noise_sensor.end(), 1.0);

  // optimize
  estimator.Optimize();

  // error
  std::vector<double> configuration_error(nq * T);
  mju_sub(configuration_error.data(), estimator.configuration.Data(),
          sim.qpos.Data(), nq * T);

  // test cost decrease
  EXPECT_LE(estimator.GetCost(), estimator.GetCostInitial());

  // test gradient tolerance
  EXPECT_NEAR(mju_norm(estimator.GetCostGradient(), nv * T) / (nv * T), 0.0,
              1.0e-3);

  // test recovered configuration trajectory
  EXPECT_NEAR(mju_norm(configuration_error.data(), nq * T) / (nq * T), 0.0,
              1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(BatchOptimize, Box3D) {
  // load model
  mjModel* model = LoadTestModel("estimator/box/task0.xml");
  mjData* data = mj_makeData(model);

  // discrete inverse dynamics
  model->opt.enableflags |= mjENBL_INVDISCRETE;

  // dimension
  int nq = model->nq, nv = model->nv;
  int ns = model->nsensordata;

  // ----- simulate ----- //

  int T = 32;
  Simulation sim(model, T);
  auto controller = [](double* ctrl, double time) {};
  double qvel[6] = {0.01, -0.02, -0.03, 0.001, -0.002, 0.003};
  sim.SetState(data->qpos, qvel);
  sim.Rollout(controller);

  // ----- estimator ----- //

  // initialize
  Batch estimator(model, T);
  estimator.settings.gradient_tolerance = 1.0e-6;
  mju_copy(estimator.configuration.Data(), sim.qpos.Data(), nq * T);
  mju_copy(estimator.configuration_previous.Data(), sim.qpos.Data(), nq * T);
  mju_copy(estimator.force_measurement.Data(), sim.qfrc_actuator.Data(),
           nv * T);
  mju_copy(estimator.sensor_measurement.Data(), sim.sensor.Data(), ns * T);

  // ----- random perturbation ----- //

  // loop over configurations
  for (int t = 0; t < T; t++) {
    // unpack
    double* q = estimator.configuration.Get(t);
    double dq[6];
    // add noise
    for (int i = 0; i < nv; i++) {
      absl::BitGen gen_;
      dq[i] = 0.01 * absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
    // integrate configuration
    mj_integratePos(model, q, dq, 1.0);
  }

  // set weights
  estimator.scale_prior = 0.25;

  // set process noise
  std::fill(estimator.noise_process.begin(), estimator.noise_process.end(),
            1.0);

  // set sensor noise
  std::fill(estimator.noise_sensor.begin(), estimator.noise_sensor.end(), 1.0);

  // optimize
  estimator.Optimize();

  // error
  std::vector<double> configuration_error(nq * T);
  mju_sub(configuration_error.data(), estimator.configuration.Data(),
          sim.qpos.Data(), nq * T);

  // test cost decrease
  EXPECT_LE(estimator.GetCost(), estimator.GetCostInitial());

  // test gradient tolerance
  EXPECT_NEAR(mju_norm(estimator.GetCostGradient(), nv * T) / (nv * T), 0.0,
              1.0e-3);

  // test configuration trajectory error
  EXPECT_NEAR(mju_norm(configuration_error.data(), nq * T) / (nq * T), 0.0,
              1.0e-3);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(BatchOptimize, Quadruped) {
  // verbose
  bool verbose = false;

  // load model
  mjModel* model = LoadTestModel("estimator/quadruped/task.xml");
  mjData* data = mj_makeData(model);

  // discrete inverse dynamics
  model->opt.enableflags |= mjENBL_INVDISCRETE;

  // dimension
  int nq = model->nq, nv = model->nv, nu = model->nu;
  int ns = model->nsensordata;

  if (verbose) {
    printf("Quadruped dimensions:\n");
    printf("nq: %i\n", nq);
    printf("nv: %i\n", nv);
    printf("nu: %i\n", nu);
    printf("ns: %i\n", ns);
  }

  // ----- simulate ----- //
  int T = 64;
  if (verbose) printf("T: %i\n", T);
  Simulation sim(model, T);
  auto controller = [](double* ctrl, double time) {};
  sim.Rollout(controller);

  // ----- estimator ----- //

  // initialize
  Batch estimator(model, T);
  estimator.settings.verbose_optimize = verbose;
  estimator.settings.verbose_prior = verbose;
  mju_copy(estimator.configuration.Data(), sim.qpos.Data(), nq * T);
  mju_copy(estimator.configuration_previous.Data(), sim.qpos.Data(), nq * T);
  mju_copy(estimator.force_measurement.Data(), sim.qfrc_actuator.Data(),
           nv * T);
  mju_copy(estimator.sensor_measurement.Data(), sim.sensor.Data(), ns * T);

  // ----- random perturbation ----- //

  // loop over configurations
  for (int t = 0; t < T; t++) {
    // unpack
    double* q = estimator.configuration.Get(t);
    std::vector<double> noise(nv);
    // add noise
    for (int i = 0; i < nv; i++) {
      // absl::BitGen gen_;
      noise[i] = 0.05;  // * absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
    // integrate configuration
    mj_integratePos(model, q, noise.data(), 1.0);
  }

  // settings
  estimator.settings.max_smoother_iterations = 1;
  estimator.settings.max_search_iterations = 10;

  // set weights
  estimator.scale_prior = 0.333;
  mju_fill(estimator.noise_process.data(), 1.0, estimator.DimensionProcess());
  mju_fill(estimator.noise_sensor.data(), 1.0, estimator.DimensionSensor());

  // optimize
  estimator.Optimize();

  // error
  std::vector<double> configuration_error(nq * T);
  mju_sub(configuration_error.data(), estimator.configuration.Data(),
          sim.qpos.Data(), nq * T);

  // test cost decrease
  EXPECT_LE(estimator.GetCost(), estimator.GetCostInitial());

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
