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

set(project_targets controller smartspray_demo controller_tests)
option(SMARTSPRAY_BUILD_VISION "Build optional CPU detector and M3 consistency harness" OFF)
if(SMARTSPRAY_BUILD_VISION)
    set(ONNXRUNTIME_ROOT "" CACHE PATH "Official ONNX Runtime CPU release directory")
    if(NOT ONNXRUNTIME_ROOT)
        message(FATAL_ERROR "Vision requires explicit -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.22.0")
    endif()
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
              PATHS "${ONNXRUNTIME_ROOT}/include" NO_DEFAULT_PATH)
    find_library(ONNXRUNTIME_LIBRARY onnxruntime
                 PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH)
    if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR "ONNXRUNTIME_ROOT must contain include/ and lib/ from an official CPU release")
    endif()
    add_library(onnxruntime_cpu SHARED IMPORTED)
    set_target_properties(onnxruntime_cpu PROPERTIES IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
                          INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")

    # Ubuntu's minimal component development packages do not provide OpenCVConfig.cmake.
    find_path(OPENCV_INCLUDE_DIR opencv2/core.hpp PATH_SUFFIXES opencv4)
    foreach(component core imgproc imgcodecs)
        find_library(OPENCV_${component}_LIBRARY NAMES opencv_${component})
        if(NOT OPENCV_${component}_LIBRARY)
            message(FATAL_ERROR "Missing required OpenCV component: ${component}")
        endif()
    endforeach()
    if(NOT OPENCV_INCLUDE_DIR)
        message(FATAL_ERROR "Missing OpenCV headers")
    endif()
    find_package(nlohmann_json 3.11 REQUIRED)

    add_library(vision_core src/vision.cpp)
    target_include_directories(vision_core PUBLIC src)
    target_include_directories(vision_core SYSTEM PUBLIC "${OPENCV_INCLUDE_DIR}")
    target_link_libraries(vision_core PUBLIC nlohmann_json::nlohmann_json
                          "${OPENCV_core_LIBRARY}" "${OPENCV_imgproc_LIBRARY}"
                          "${OPENCV_imgcodecs_LIBRARY}")
    add_library(vision_runtime src/vision_runtime.cpp)
    target_link_libraries(vision_runtime PUBLIC vision_core onnxruntime_cpu)
    add_executable(smartspray_infer src/vision_main.cpp)
    target_link_libraries(smartspray_infer PRIVATE vision_runtime)
    add_executable(vision_tests tests/vision_tests.cpp)
    target_link_libraries(vision_tests PRIVATE vision_runtime)
    add_test(NAME vision_unit_tests COMMAND vision_tests)
    set_tests_properties(vision_unit_tests PROPERTIES LABELS "offline")

    set(SMARTSPRAY_REAL_MODEL "" CACHE FILEPATH "Optional local ONNX model for real-model smoke test")
    set(SMARTSPRAY_REAL_IMAGE "" CACHE FILEPATH "Optional local PNG/BMP image for real-model smoke test")
    add_test(NAME vision_real_model COMMAND vision_tests --real
             "${SMARTSPRAY_REAL_MODEL}" "${SMARTSPRAY_REAL_IMAGE}")
    set_tests_properties(vision_real_model PROPERTIES SKIP_RETURN_CODE 77 LABELS "real_model")

    add_library(simulation src/simulation.cpp)
    target_link_libraries(simulation PUBLIC controller nlohmann_json::nlohmann_json)
    add_library(demo_core src/demo.cpp src/demo_render.cpp src/sha256.cpp)
    target_link_libraries(demo_core PUBLIC simulation vision_core)
    add_executable(smartspray_vision_demo src/demo_main.cpp)
    target_link_libraries(smartspray_vision_demo PRIVATE demo_core vision_runtime)
    add_executable(demo_tests tests/demo_tests.cpp)
    target_link_libraries(demo_tests PRIVATE demo_core)
    add_test(NAME demo_integration_tests COMMAND demo_tests)
    set_tests_properties(demo_integration_tests PROPERTIES LABELS "offline")
    add_executable(controller_parity src/controller_parity.cpp)
    target_link_libraries(controller_parity PRIVATE simulation)
    add_test(NAME controller_parity_self_test COMMAND controller_parity --self-test)
    set_tests_properties(controller_parity_self_test PROPERTIES LABELS "offline")
    list(APPEND project_targets vision_core vision_runtime smartspray_infer vision_tests controller_parity
         simulation demo_core smartspray_vision_demo demo_tests)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    foreach(target IN LISTS project_targets)
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endforeach()
endif()
