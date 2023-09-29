# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.16)

# INTERPROCEDURAL_OPTIMIZATION is enforced when enabled.
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
# Default to GLVND if available.
set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)

# This line has to appear before 'PROJECT' in order to be able to disable incremental linking
set(MSVC_INCREMENTAL_DEFAULT ON)

project(
  mujoco_mpc
  VERSION 0.0.1
  DESCRIPTION "MuJoCo predictive control"
  HOMEPAGE_URL "https://github.com/deepmind/mujoco_mpc"
)

enable_language(C)
enable_language(CXX)
if(APPLE)
  enable_language(OBJC)
  enable_language(OBJCXX)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH /opt/drake)

include(MpcOptions)
include(MujocoMacOS)

option(MJPC_BUILD_TESTS "Build tests for MJPC" ON)
add_compile_options(-w) # Suppress ALL warnings (https://stackoverflow.com/questions/65011913/cmake-suppress-all-warnings)

include(FindOrFetch)

find_package(Python)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(SBPL REQUIRED)
find_package(drake REQUIRED)

include_directories(include
        ${SBPL_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${drake_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/pinsat/include)

set(MUJOCO_BUILD_EXAMPLES OFF)
set(MUJOCO_BUILD_SIMULATE ON)
set(MUJOCO_BUILD_TESTS OFF)
set(MUJOCO_TEST_PYTHON_UTIL OFF)

findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  mujoco
  LIBRARY_NAME
  mujoco
  GIT_REPO
  https://github.com/deepmind/mujoco.git
  GIT_TAG
  main
  TARGETS
  mujoco::mujoco
  mujoco::platform_ui_adapter
  EXCLUDE_FROM_ALL
)

set(MUJOCO_DEP_VERSION_abseil
    c8a2f92586fe9b4e1aff049108f5db8064924d8e # LTS 20230125.1
    CACHE STRING "Version of `abseil` to be fetched."
)

set(MUJOCO_DEP_VERSION_glfw3
    7482de6071d21db77a7236155da44c172a7f6c9e # 3.3.8
    CACHE STRING "Version of `glfw` to be fetched."
)

set(MJPC_DEP_VERSION_lodepng
    b4ed2cd7ecf61d29076169b49199371456d4f90b
    CACHE STRING "Version of `lodepng` to be fetched."
)

set(BUILD_SHARED_LIBS_OLD ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS
    OFF
    CACHE INTERNAL "Build SHARED libraries"
)

findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  GTest
  LIBRARY_NAME
  googletest
  GIT_REPO
  https://github.com/google/googletest.git
  GIT_TAG
  ${MUJOCO_DEP_VERSION_gtest}
  TARGETS
  gtest
  gmock
  gtest_main
  EXCLUDE_FROM_ALL
)

set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_BUILD_TESTING OFF)
findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  absl
  LIBRARY_NAME
  abseil-cpp
  GIT_REPO
  https://github.com/abseil/abseil-cpp.git
  GIT_TAG
  ${MUJOCO_DEP_VERSION_abseil}
  TARGETS
  absl::core_headers
  EXCLUDE_FROM_ALL
)

set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)
set(GLFW_INSTALL OFF)

findorfetch(
  USE_SYSTEM_PACKAGE
  MJPC_USE_SYSTEM_GLFW
  PACKAGE_NAME
  glfw
  LIBRARY_NAME
  glfw
  GIT_REPO
  https://github.com/glfw/glfw.git
  GIT_TAG
  ${MUJOCO_DEP_VERSION_glfw3}
  TARGETS
  glfw
  EXCLUDE_FROM_ALL
)

set(BUILD_SHARED_LIBS
    ${BUILD_SHARED_LIBS_OLD}
    CACHE BOOL "Build SHARED libraries" FORCE
)
unset(BUILD_SHARED_LIBS_OLD)

FetchContent_Declare(
  menagerie
  GIT_REPOSITORY https://github.com/deepmind/mujoco_menagerie.git
  GIT_TAG main
)

FetchContent_GetProperties(menagerie)
if(NOT menagerie_POPULATED)
  FetchContent_Populate(menagerie)
endif()

add_custom_target(
  copy_menagerie_resources ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/shadow_hand/assets
          ${CMAKE_CURRENT_BINARY_DIR}/tasks/hand/assets
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/franka_emika_panda/assets
          ${CMAKE_CURRENT_BINARY_DIR}/tasks/panda/assets
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/unitree_a1/assets
          ${CMAKE_CURRENT_BINARY_DIR}/tasks/quadruped/assets
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/franka_emika_panda
          ${CMAKE_CURRENT_BINARY_DIR}/tasks/manipulation
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/robotiq_2f85
          ${CMAKE_CURRENT_BINARY_DIR}/tasks/manipulation
  COMMAND ${Python_EXECUTABLE}
          ${CMAKE_CURRENT_BINARY_DIR}/tasks/manipulation/merge_panda_robotiq.py
          ${CMAKE_CURRENT_BINARY_DIR}/tasks/manipulation/panda_robotiq.xml
  COMMENT "Copying menagerie assets into binary directory")

if(NOT TARGET lodepng)
  FetchContent_Declare(
    lodepng
    GIT_REPOSITORY https://github.com/lvandeve/lodepng.git
    GIT_TAG ${MJPC_DEP_VERSION_lodepng}
  )

  FetchContent_GetProperties(lodepng)
  if(NOT lodepng_POPULATED)
    FetchContent_Populate(lodepng)
    # This is not a CMake project.
    set(LODEPNG_SRCS ${lodepng_SOURCE_DIR}/lodepng.cpp)
    set(LODEPNG_HEADERS ${lodepng_SOURCE_DIR}/lodepng.h)
    add_library(lodepng STATIC ${LODEPNG_HEADERS} ${LODEPNG_SRCS})
    target_compile_options(lodepng PRIVATE ${MUJOCO_MACOS_COMPILE_OPTIONS})
    target_link_options(lodepng PRIVATE ${MUJOCO_MACOS_LINK_OPTIONS})
    target_include_directories(lodepng PUBLIC ${lodepng_SOURCE_DIR})
  endif()
endif()

set(MJPC_COMPILE_OPTIONS "${AVX_COMPILE_OPTIONS}" "${EXTRA_COMPILE_OPTIONS}")
set(MJPC_LINK_OPTIONS "${EXTRA_LINK_OPTIONS}")

add_library(threadpool STATIC)
target_sources(
  threadpool
  PUBLIC threadpool.h
  PRIVATE threadpool.cc
)
target_link_libraries(
  threadpool
  absl::base
)
target_include_directories(threadpool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(
        libmjpc STATIC
        states/state.cc
        states/state.h
        agent.cc
        agent.h
        trajectory.cc
        trajectory.h
        utilities.cc
        utilities.h
        tasks/tasks.cc
        tasks/tasks.h
        tasks/acrobot/acrobot.cc
        tasks/acrobot/acrobot.h
        tasks/cartpole/cartpole.cc
        tasks/cartpole/cartpole.h
        tasks/fingers/fingers.cc
        tasks/fingers/fingers.h
        tasks/hand/hand.cc
        tasks/hand/hand.h
        tasks/humanoid/stand/stand.cc
        tasks/humanoid/stand/stand.h
        tasks/humanoid/tracking/tracking.cc
        tasks/humanoid/tracking/tracking.h
        tasks/humanoid/walk/walk.cc
        tasks/humanoid/walk/walk.h
        tasks/manipulation/common.cc
        tasks/manipulation/common.h
        tasks/manipulation/manipulation.cc
        tasks/manipulation/manipulation.h
        tasks/panda/panda.cc
        tasks/panda/panda.h
        tasks/particle/particle.cc
        tasks/particle/particle.h
        tasks/planar_pusher/planar_pusher.h
        tasks/planar_pusher/planar_pusher.cc
        tasks/quadrotor/quadrotor.cc
        tasks/quadrotor/quadrotor.h
        tasks/quadruped/quadruped.cc
        tasks/quadruped/quadruped.h
        tasks/swimmer/swimmer.cc
        tasks/swimmer/swimmer.h
        tasks/walker/walker.cc
        tasks/walker/walker.h
        tasks/insat/gen3_hebi/gen3_hebi.cc
        tasks/insat/gen3_flip/gen3_flip.cc
        tasks/insat/gen3_realflip/gen3_realflip.cc
        planners/planner.cc
        planners/planner.h
        planners/policy.h
        planners/include.cc
        planners/include.h
        planners/cost_derivatives.cc
        planners/cost_derivatives.h
        planners/model_derivatives.cc
        planners/model_derivatives.h
        planners/robust/robust_planner.cc
        planners/robust/robust_planner.h
        planners/sampling/planner.cc
        planners/sampling/planner.h
        planners/sampling/policy.cc
        planners/sampling/policy.h
        planners/gradient/gradient.cc
        planners/gradient/gradient.h
        planners/gradient/planner.cc
        planners/gradient/planner.h
        planners/gradient/policy.cc
        planners/gradient/policy.h
        planners/gradient/settings.h
        planners/gradient/spline_mapping.cc
        planners/gradient/spline_mapping.h
        planners/ilqg/backward_pass.cc
        planners/ilqg/backward_pass.h
        planners/ilqg/boxqp.h
        planners/ilqg/planner.cc
        planners/ilqg/planner.h
        planners/ilqg/policy.cc
        planners/ilqg/policy.h
        planners/ilqg/settings.h
        planners/ilqs/planner.cc
        planners/ilqs/planner.h
        estimators/batch.cc
        estimators/batch.h
        estimators/estimator.h
        estimators/include.cc
        estimators/include.h
        estimators/kalman.cc
        estimators/kalman.h
        estimators/model_parameters.cc
        estimators/model_parameters.h
        estimators/trajectory.h
        estimators/unscented.cc
        estimators/unscented.h
        app.cc
        app.h
        norm.cc
        norm.h
        simulate.cc
        simulate.h
        task.cc
        task.h
#        insat_task.h
#        insat_task.cc
        $<TARGET_OBJECTS:mujoco::platform_ui_adapter>
)
set_target_properties(libmjpc PROPERTIES OUTPUT_NAME mjpc PUBLIC_HEADER "${LIBMJPC_INCLUDES}")
target_compile_options(libmjpc PUBLIC ${MJPC_COMPILE_OPTIONS})
target_compile_definitions(libmjpc PRIVATE MJSIMULATE_STATIC)
target_link_libraries(
  libmjpc
  absl::any_invocable
  absl::flat_hash_map
  absl::random_random
  glfw
  lodepng
  mujoco::mujoco
  mujoco::platform_ui_adapter
  threadpool
  Threads::Threads
  mujoco::platform_ui_adapter
)
target_include_directories(libmjpc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_executable(
  mjpc
  main.cc
)
target_link_libraries(
  mjpc
  absl::flags
  absl::flags_parse
  absl::random_random
  absl::strings
  libmjpc
  mujoco::mujoco
  threadpool
  Threads::Threads
)
target_include_directories(mjpc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(mjpc PUBLIC ${MJPC_COMPILE_OPTIONS})
target_link_options(mjpc PRIVATE ${MJPC_LINK_OPTIONS})
target_compile_definitions(mjpc PRIVATE MJSIMULATE_STATIC)
if(APPLE)
  target_sources(mjpc PRIVATE macos_gui.mm)
  target_link_libraries(mjpc "-framework Cocoa")
endif()

######### (P)INSAT additions #########
set(PINSAT_SOURCES
  pinsat/src/common/State.cpp
  pinsat/src/common/Edge.cpp
  pinsat/src/common/insat/InsatEdge.cpp
  pinsat/src/planners/Planner.cpp
  pinsat/src/planners/BFSPlanner.cpp
  pinsat/src/planners/WastarPlanner.cpp
  pinsat/src/planners/PwastarPlanner.cpp
  pinsat/src/planners/PasePlanner.cpp
  pinsat/src/planners/EpasePlanner.cpp
  pinsat/src/planners/GepasePlanner.cpp
  pinsat/src/planners/MplpPlanner.cpp
  pinsat/src/planners/RrtPlanner.cpp
  pinsat/src/planners/RrtConnectPlanner.cpp
  pinsat/src/planners/insat/PinsatPlanner.cpp
  pinsat/src/planners/insat/InsatPlanner.cpp
  pinsat/src/planners/insat/opt/DummyOpt.cpp
  pinsat/src/planners/insat/opt/BSplineOpt.cpp
  pinsat/src/planners/insat/opt/MjpcOpt.cpp
)

# run_robot_nav_2d
add_executable(run_robot_nav_2d
        pinsat/examples/robot_nav_2d/run_robot_nav_2d.cpp
        pinsat/examples/robot_nav_2d/RobotNav2dActions.cpp
        ${PINSAT_SOURCES})
target_link_libraries(run_robot_nav_2d
        ${OpenCV_LIBRARIES}
        ${SBPL_LIBRARIES}
        ${drake_LIBRARIES}
        pthread)
target_include_directories(run_robot_nav_2d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# run_insat_nav_2d
add_executable(run_insat_nav_2d
        pinsat/examples/insat/robot_nav_2d/run_insat_nav_2d.cpp
        pinsat/examples/insat/robot_nav_2d/InsatNav2dActions.cpp
        ${PINSAT_SOURCES})
target_link_libraries(run_insat_nav_2d
        ${OpenCV_LIBRARIES}
        ${SBPL_LIBRARIES}
        ${drake_LIBRARIES}
        pthread)
target_include_directories(run_insat_nav_2d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# run_manipulation
#add_executable(run_manipulation
#        pinsat/examples/manipulation/run_manipulation.cpp
#        pinsat/examples/manipulation/ManipulationActions.cpp
#        pinsat/examples/manipulation/bfs3d.cpp
#        ${PINSAT_SOURCES})
#target_link_libraries(run_manipulation
#        mujoco::mujoco
#        libmjpc
#        ${drake_LIBRARIES}
#        pthread)
#target_include_directories(run_manipulation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# run_ptc
add_executable(run_ptc
        pinsat/examples/manipulation/run_ptc.cpp
        pinsat/examples/manipulation/ManipulationActions.cpp
        pinsat/examples/manipulation/bfs3d.cpp
        ${PINSAT_SOURCES})
target_link_libraries(run_ptc
        mujoco::mujoco
        libmjpc
        ${drake_LIBRARIES}
        pthread)
target_include_directories(run_ptc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

### opt_test
add_executable(opt_test
        test/insat/opt_test.cc
        )
target_link_libraries(opt_test
        libmjpc
        )
target_include_directories(opt_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(opt_test PUBLIC ${MJPC_COMPILE_OPTIONS})

### opt_test
add_executable(mjpc_pp
        test/insat/mjpc_pp.cc
        )
target_link_libraries(mjpc_pp
        libmjpc
        )
target_include_directories(mjpc_pp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(mjpc_pp PUBLIC ${MJPC_COMPILE_OPTIONS})

add_subdirectory(tasks)

if(BUILD_TESTING AND MJPC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
