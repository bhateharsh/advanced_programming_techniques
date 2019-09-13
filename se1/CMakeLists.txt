cmake_minimum_required(VERSION 3.9)
project(se1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
include_directories(
src
)
set(SOURCE_FILES
src/vector3.h
src/vector3.cc
)
add_executable(${PROJECT_NAME} ${SOURCE_FILES} main.cc)