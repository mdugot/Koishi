cmake_minimum_required(VERSION 3.19)
project(Koishi)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Boost 1.75 REQUIRED COMPONENTS python numpy)

include_directories(
${PROJECT_SOURCE_DIR}/includes
)

add_library(klib SHARED
sources/initializer/feed.cpp
sources/initializer/fill.cpp
sources/initializer/initializer.cpp
sources/initializer/uniform.cpp
sources/operation/addition.cpp
sources/operation/constant.cpp
sources/operation/count.cpp
sources/operation/inverse.cpp
sources/operation/log.cpp
sources/operation/multiplication.cpp
sources/operation/number.cpp
sources/operation/numberException.cpp
sources/operation/operation.cpp
sources/operation/percentile.cpp
sources/operation/pow.cpp
sources/operation/reduction.cpp
sources/operation/sigmoid.cpp
sources/operation/sum.cpp
sources/operation/transformation.cpp
sources/operation/variable.cpp
sources/tensor/tensor.cpp
sources/tensor/tensorException.cpp
sources/tensor/tensors.cpp
sources/wrapper/wrapper.cpp
sources/wrapper/wrapperTools.cpp
)

add_executable(Test sources/main.cpp)
target_link_libraries(Test klib)
