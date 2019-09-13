cmake_minimum_required(VERSION 3.9.1)
project (p2)

enable_language("CUDA")
find_package(CUDA 9.0 REQUIRED)


#set (CMAKE_C_COMPILER /usr/bin/gcc-4.9)
#set (CMAKE_CXX_COMPILER /usr/bin/g++-4.9)
set(CMAKE_CXX_STANDARD 11)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;)

include_directories(src)

set (SOURCE_FILES
        src/helper.h
        src/grid2D.cuh
        src/configParser.h
        src/configParser.cc
        src/grid3D.cuh)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

cuda_add_executable( 
    heat2D3D 
    ${SOURCE_FILES} 
    main.cu)