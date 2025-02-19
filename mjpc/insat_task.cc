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
 * \file insat_task.cc 
 * \author Ram Natarajan (rnataraj@cs.cmu.edu)
 * \date 4/8/23
*/

#include "insat_task.h"

namespace mjpc
{

InsatTask::InsatTask() {


}

std::string InsatTask::Name() const {
  return "Insat Task";
}

std::string InsatTask::XmlPath() const {
  return "/path/to/xml";
}

void InsatTask::Reset(const mjModel* model, const mjData* data) {
  Task::Reset(model);

  // Initialize the start state
  start_.Initialize(model);
  start_.Allocate(model);
  start_.Reset();
  start_.Set(model, data);


  // Initialize the goal state
  goal_.Initialize(model);
  goal_.Allocate(model);
  goal_.Reset();
  goal_.Set(model, data);
}

void InsatTask::Residual(const mjModel *model, const mjData *data, double *residual) const {

}

void InsatTask::setStart(const VecDf &start) {

}

void InsatTask::setGoal(const VecDf &goal) {

}

}