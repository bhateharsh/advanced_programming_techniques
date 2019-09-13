cmake_minimum_required(VERSION 3.9.1)
project (se2)
find_package(MPI REQUIRED)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")


set (SOURCE_FILES src/heat_transfer_1D.cc)

include_directories(src)
include_directories( ${MPI_CXX_INCLUDE_PATH})

add_executable( heat1D ${SOURCE_FILES})
target_link_libraries(heat1D ${MPI_CXX_LIBRARIES})
