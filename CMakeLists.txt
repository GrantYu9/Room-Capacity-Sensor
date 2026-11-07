cmake_minimum_required(VERSION 4.3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(room_capacity_sensor)

add_subdirectory(include)
add_subdirectory(interface)
