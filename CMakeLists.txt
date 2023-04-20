cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)

project(libcneus)

find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

include_directories(cpp/include)
cuda_add_library(
    libcneus SHARED
    cpp/src/bindings.cpp
    cpp/src/marching_rays_kernel.cu
    cpp/src/grid_sample_kernel.cu
    cpp/src/misc_kernel.cu
    cpp/src/scatter_kernel.cu
    cpp/src/sliding_window.cu
    cpp/src/up_sample_kernel.cu
    cpp/src/spherical_harmonics.cu
)
set_target_properties(libcneus PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(libcneus PROPERTIES PREFIX "")
target_link_libraries(libcneus ${TORCH_PYTHON_LIBRARY})
target_compile_definitions(libcneus PUBLIC TORCH_EXTENSION_NAME=libcneus)
target_compile_definitions(libcneus PUBLIC TORCH_API_INCLUDE_EXTENSION_H)
target_compile_definitions(libcneus PUBLIC ${TORCH_CXX_FLAGS})
target_include_directories(libcneus PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(libcneus PUBLIC ${PYTHON_INCLUDE_DIRS})
