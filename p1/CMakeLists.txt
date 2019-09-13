cmake_minimum_required(VERSION 3.8)
project(p1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

include_directories(
        src
)

set(SOURCE_FILES
        src/array.h
        src/array.cc
        src/simple_string.h
        src/simple_string.cc
        )

add_executable(${PROJECT_NAME} ${SOURCE_FILES} test.cc)