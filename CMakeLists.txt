cmake_minimum_required(VERSION 3.19)
project(Koishi)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Boost 1.75 REQUIRED COMPONENTS python numpy)
find_package(Python3 3.8 REQUIRED COMPONENTS Development)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -D PYTHON_WRAPPER")

include_directories(${Boost_INCLUDE_DIRS}) 
message(STATUS ${Python3_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS}) 
include_directories(
${PROJECT_SOURCE_DIR}/includes
)

add_library(koishi SHARED
sources/initializer/feed.cpp
sources/initializer/fill.cpp
sources/initializer/initializer.cpp
sources/initializer/uniform.cpp
sources/operation/get.cpp
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
sources/operation/exp.cpp
sources/operation/sum.cpp
sources/operation/transformation.cpp
sources/operation/variable.cpp
sources/tensor/tensor.cpp
sources/tensor/tensorException.cpp
sources/wrapper/wrapper.cpp
sources/wrapper/wrapperTools.cpp
)

set_target_properties(koishi PROPERTIES PREFIX "")
target_link_libraries(koishi ${Python3_LIBRARY_RELEASE})
target_link_libraries(koishi ${Boost_LIBRARIES})
