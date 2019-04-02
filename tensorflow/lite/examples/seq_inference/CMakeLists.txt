cmake_minimum_required(VERSION 3.5)
project(seq_inference LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_BUILD_TYPE "Release")

SET(TENSORFLOW_BASE_DIR /home/xiaobizh/Projects/tensorflow)

# specify the cross compiler
SET(CMAKE_C_COMPILER   /opt/gcc-armv7l-meego-linux-gnueabi-2016.01_linux/cross/bin/armv7l-meego-linux-gnueabi-gcc)
SET(CMAKE_CXX_COMPILER /opt/gcc-armv7l-meego-linux-gnueabi-2016.01_linux/cross/bin/armv7l-meego-linux-gnueabi-g++)

## where is the target environment
SET(CMAKE_FIND_ROOT_PATH  /opt/gcc-armv7l-meego-linux-gnueabi-2016.01_linux/cross/armv7l-meego-linux-gnueabi/sys-root/)
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

## specify the compiler flag
SET(CMAKE_C_FLAGS  "-std=gnu99 -O3 -mfpu=neon")
SET(CMAKE_CXX_FLAGS  "-std=c++11 -O3 -mfpu=neon")


#set(CMAKE_C_COMPILER arm-linux-gnueabi-gcc)
#set(CMAKE_CXX_COMPILER arm-linux-gnueabi-g++)

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfloat-abi=softfp -mfpu=neon-fp-armv8 -mcpu=cortex-a53 -funsafe-math-optimizations" CACHE STRING "" FORCE)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG --std=c++11 -mfloat-abi=softfp -mfpu=neon-fp-armv8 -mcpu=cortex-a53 -funsafe-math-optimizations" CACHE STRING "" FORCE)
#SET(CMAKE_LD_FLAGS  "-static")

#set(CMAKE_SYSROOT /opt/roborock_toolchain/ruby2/arm-linux-gnueabi/)
#set(CMAKE_FIND_ROOT_PATH /opt/roborock_toolchain/ruby2/arm-linux-gnueabi/)

#set(CMAKE_CROSSCOMPILING true)
#set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
#set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
#set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)



set(SEQ_INF_SRC
        seq_inference.cc
        featurevec_helpers.cc)

include_directories(${TENSORFLOW_BASE_DIR} "${TENSORFLOW_BASE_DIR}/tensorflow/lite/tools/make/downloads/flatbuffers/include" )
#link_directories("${TENSORFLOW_BASE_DIR}/tensorflow/lite/tools/make/gen/ruby2_armv8/lib/")
link_directories("${TENSORFLOW_BASE_DIR}/tensorflow/lite/tools/make/gen/bev_armv7l/lib/")
#link_directories("${TENSORFLOW_BASE_DIR}/tensorflow/lite/build/")
add_executable(seq_inference ${SEQ_INF_SRC})
target_link_libraries(seq_inference libtensorflow-lite.a -lstdc++ -lpthread -lm -ldl)
