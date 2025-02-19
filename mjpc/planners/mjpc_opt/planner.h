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
 * \file   InsatPlanner.cpp
 * \author Ramkumar Natarajan (rnataraj@cs.cmu.edu)
 * \date   4/7/23
 */

#ifndef MJPC_PLANNERS_INSAT_OPTIMIZER_H_
#define MJPC_PLANNERS_INSAT_OPTIMIZER_H_

#include <mjpc/planners/ilqg/planner.h>
#include <mjpc/insat_task.h>
#include <mjpc/threadpool.h>
#include <mjpc/common/EigenTypes.h>

namespace mjpc {

class MjpcOpt {
 public:

  struct Params {
    int max_iter_;
    double conv_thresh_;
  };

  // constructor
  MjpcOpt(mjModel* model, const mjData* data);

  iLQGPolicy optimize(int horizon);

  iLQGPolicy optimize(const VecDf& low1, const VecDf& low2, double dt, double T);

  iLQGPolicy optimize(const VecDf& low1, const VecDf& low2,
                      const VecDf& aux1, const VecDf& aux2,
                      double dt, double T);

  iLQGPolicy warmOptimize(const VecDf& low1, const VecDf& low2,
                          const iLQGPolicy& pol1, const iLQGPolicy& pol2, double dt);

  iLQGPolicy warmOptimize(const VecDf& low1, const VecDf& low2,
                          const VecDf& aux1, const VecDf& aux2,
                          const iLQGPolicy& pol1, const iLQGPolicy& pol2, double dt);


  int state_len_;
  Params params_;
  InsatTask task_;
  iLQGPlanner planner_;
  ThreadPool pool_;

};

}

#endif //MJPC_PLANNERS_INSAT_OPTIMIZER_H_
