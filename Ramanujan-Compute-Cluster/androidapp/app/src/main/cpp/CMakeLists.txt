cmake_minimum_required(VERSION 3.27)
project(cluster_android LANGUAGES C CXX)
set(GPU_ENABLED ON CACHE BOOL "" FORCE)
# This file is six levels below the repository root.
get_filename_component(RAMANUJAN_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../../.." ABSOLUTE)
add_subdirectory("${RAMANUJAN_ROOT}/ramanujan-native/native" "${CMAKE_CURRENT_BINARY_DIR}/ramanujan-native")
# The Android dlopen loader declares cl_queue_properties (a 2.0 header type).
# Header visibility does not require the device to support OpenCL 2.0.
target_compile_definitions(ramanujan_llm PRIVATE CL_TARGET_OPENCL_VERSION=200)
add_library(cluster_preflight SHARED preflight.cpp)
target_link_libraries(cluster_preflight PRIVATE dl)
