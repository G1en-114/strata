cmake_minimum_required(VERSION 3.20)
project(lpp_runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ONNX Runtime: headers vendored in third_party (v1.32.0), library from the onnxruntime-gpu wheel
# of the ML environment (same version). Override ORT_LIB_DIR for another installation.
set(ORT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime-1.32.0/include)
if(NOT ORT_LIB_DIR)
  execute_process(COMMAND "$ENV{HOME}/AI infra/.venv/bin/python" -c
                  "import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'))"
                  OUTPUT_VARIABLE ORT_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
find_file(ORT_LIB NAMES libonnxruntime.so libonnxruntime.so.1.32.0 PATHS ${ORT_LIB_DIR} NO_DEFAULT_PATH REQUIRED)
message(STATUS "onnxruntime: ${ORT_LIB}")

add_executable(lpp_paint src/main.cpp src/protocol.cpp src/grammar.cpp src/raster.cpp src/session.cpp src/painter.cpp
                         src/tokenizer.cpp src/text_encoder.cpp)
target_include_directories(lpp_paint PRIVATE ${ORT_INCLUDE_DIR} src)
target_link_libraries(lpp_paint PRIVATE ${ORT_LIB})
# the wheel ships libonnxruntime.so.1.32.0 but the library's SONAME is libonnxruntime.so.1:
# give the loader that name next to the binary
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/ortlib)
file(CREATE_LINK ${ORT_LIB} ${CMAKE_BINARY_DIR}/ortlib/libonnxruntime.so.1 SYMBOLIC)
# execution-provider plugins are looked up next to the loaded libonnxruntime
foreach(ep providers_shared providers_cuda providers_tensorrt)
  if(EXISTS ${ORT_LIB_DIR}/libonnxruntime_${ep}.so)
    file(CREATE_LINK ${ORT_LIB_DIR}/libonnxruntime_${ep}.so ${CMAKE_BINARY_DIR}/ortlib/libonnxruntime_${ep}.so SYMBOLIC)
  endif()
endforeach()
set_target_properties(lpp_paint PROPERTIES BUILD_RPATH "${CMAKE_BINARY_DIR}/ortlib;${ORT_LIB_DIR}")
target_compile_options(lpp_paint PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)

# grammar cross-check tool (tests/test_runtime_grammar.py): no ONNX Runtime needed
add_executable(lpp_grammar_walk src/grammar_walk.cpp src/protocol.cpp src/grammar.cpp)
target_include_directories(lpp_grammar_walk PRIVATE src)
target_compile_options(lpp_grammar_walk PRIVATE -O2 -Wall -Wextra)

# CUDA runtime (static one-step graphs + CUDA Graph path): headers and libcudart from the nvidia cu13 wheel
if(NOT CUDART_ROOT)
  set(CUDART_ROOT "${ORT_LIB_DIR}/../../nvidia/cu13")
endif()
# the wheel's headers are incomplete (no crt/host_defines.h): use the toolkit's headers (same major
# version, 13.x); the library stays the wheel's libcudart.so.13, the one ORT's CUDA EP loads, so the
# process holds a single CUDA runtime
find_path(CUDART_INCLUDE cuda_runtime_api.h PATHS /opt/cuda/include /usr/local/cuda/include ${CUDART_ROOT}/include NO_DEFAULT_PATH)
find_file(CUDART_LIB NAMES libcudart.so.13 libcudart.so PATHS ${CUDART_ROOT}/lib NO_DEFAULT_PATH)
if(CUDART_INCLUDE AND CUDART_LIB)
  message(STATUS "cudart: ${CUDART_LIB}")
  target_include_directories(lpp_paint PRIVATE ${CUDART_INCLUDE})
  target_link_libraries(lpp_paint PRIVATE ${CUDART_LIB})
  target_compile_definitions(lpp_paint PRIVATE LPP_HAVE_CUDART)
else()
  message(STATUS "cudart not found: static step graphs disabled")
endif()
