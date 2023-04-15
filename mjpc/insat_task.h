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
 * \file insat_task.h 
 * \author Ram Natarajan (rnataraj@cs.cmu.edu)
 * \date 4/7/23
*/

#ifndef MJPC_TASKS_INSAT_TASK_H_
#define MJPC_TASKS_INSAT_TASK_H_

#include <shared_mutex>

#include <mjpc/task.h>
#include <mjpc/states/state.h>
#include <mjpc/common/EigenTypes.h>

namespace mjpc {


class InsatTask : public Task
{
 public:
  InsatTask();

  void Reset(const mjModel* model, const mjData* data);

  std::string Name() const override;
  std::string XmlPath() const override;
  // ------- Residuals for cartpole task ------
  //   Number of residuals: 4
  //     Residual (0):
  //     Residual (1):
  //     Residual (2):
  //     Residual (3):
  // ------------------------------------------
  void Residual(const mjModel* model, const mjData* data,
                double* residual) const override;


  void setStart(const VecDf& start);

  void setGoal(const VecDf& goal);

  State start_;
  State goal_;

  bool free_start_ = false;
  bool free_goal_ = false;

};

}

#endif //MJPC_TASKS_INSAT_TASK_H_
