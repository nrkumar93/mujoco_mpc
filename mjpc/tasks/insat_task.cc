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

#include <mjpc/tasks/insat_task.h>

namespace mjpc
{

InsatTask::InsatTask(std::vector<std::shared_ptr<mjpc::Task>> tasks, int task_id) {

  sim = std::make_unique<mj::Simulate>(
      std::make_unique<mujoco::GlfwAdapter>(),
      std::make_shared<Agent>());

  sim->agent->SetTaskList(std::move(tasks));
  std::string task_name = tasks[task_id]->Name();

  if (task_name.empty()) {
    sim->agent->gui_task_id = task_id;
  } else {
    sim->agent->gui_task_id = sim->agent->GetTaskIdByName(task_name);
    if (sim->agent->gui_task_id == -1) {
      std::cerr << "Invalid --task flag: '" << task_name
                << "'. Valid values:\n";
      std::cerr << sim->agent->GetTaskNames();
      mju_error("Invalid --task flag.");
    }
  }

  sim->filename = sim->agent->GetTaskXmlPath(sim->agent->gui_task_id);
  m = LoadModel(sim->filename, *sim);
  if (m) d = mj_makeData(m);

  sim->mnew = m;
  sim->dnew = d;

  // control noise
  free(ctrlnoise);
  ctrlnoise = (mjtNum*)malloc(sizeof(mjtNum) * m->nu);
  mju_zero(ctrlnoise, m->nu);

  sim->agent->Initialize(m);
  sim->agent->Allocate();
  sim->agent->Reset();
  sim->agent->PlotInitialize();

  sim->delete_old_m_d = true;
  sim->loadrequest = 2;

}

mjModel *InsatTask::LoadModel(std::string filename, mj::Simulate &sim) {
  // make sure filename is not empty
  if (filename.empty()) {
    return nullptr;
  }

  // load and compile
  char loadError[1024] = "";
  mjModel* mnew = 0;
  if (absl::StrContains(filename, ".mjb")) {
    mnew = mj_loadModel(filename.c_str(), nullptr);
    if (!mnew) {
      mju::strcpy_arr(loadError, "could not load binary model");
    }
  } else {
    mnew = mj_loadXML(filename.c_str(), nullptr, loadError,
                      mj::Simulate::kMaxFilenameLength);
    // remove trailing newline character from loadError
    if (loadError[0]) {
      int error_length = mju::strlen_arr(loadError);
      if (loadError[error_length - 1] == '\n') {
        loadError[error_length - 1] = '\0';
      }
    }
  }

  mju::strcpy_arr(sim.loadError, loadError);

  if (!mnew) {
    std::printf("%s\n", loadError);
    return nullptr;
  }

  // compiler warning: print and pause
  if (loadError[0]) {
    // mj_forward() below will print the warning message
    std::printf("Model compiled, but simulation warning (paused):\n  %s\n",
                loadError);
    sim.run = 0;
  }

  return mnew;
}


}