cmake_minimum_required(VERSION 3.18)

project(LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(COMMON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

find_package(SDL2 CONFIG REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${COMMON_INCLUDE_DIR})

add_subdirectory(src)