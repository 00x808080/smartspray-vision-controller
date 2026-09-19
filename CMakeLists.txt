cmake_minimum_required(VERSION 3.16)
project(smartspray_vision_controller LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(controller src/controller.cpp)
target_include_directories(controller PUBLIC src)

add_executable(smartspray_demo src/main.cpp)
target_link_libraries(smartspray_demo PRIVATE controller)

enable_testing()
add_executable(controller_tests tests/controller_tests.cpp)
target_link_libraries(controller_tests PRIVATE controller)
add_test(NAME controller_tests COMMAND controller_tests)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    foreach(target controller smartspray_demo controller_tests)
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endforeach()
endif()
