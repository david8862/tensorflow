# Settings for Ruby2.
ifeq ($(TARGET),ruby2)
  TARGET_ARCH := armv8
  TARGET_TOOLCHAIN_PREFIX := arm-linux-gnueabi-

  ifeq ($(TARGET_ARCH), armv8)
    CXXFLAGS += \
      --sysroot=/opt/roborock_toolchain/ruby2/arm-linux-gnueabi/ \
      -mfloat-abi=softfp \
      -mfpu=neon-fp-armv8 \
      -mcpu=cortex-a53 \
      -funsafe-math-optimizations \
      -ftree-vectorize \
      -fPIC

    CCFLAGS += \
      --sysroot=/opt/roborock_toolchain/ruby2/arm-linux-gnueabi/ \
      -mfloat-abi=softfp \
      -mfpu=neon-fp-armv8 \
      -mcpu=cortex-a53 \
      -funsafe-math-optimizations \
      -ftree-vectorize \
      -fPIC

    LDFLAGS :=
  endif


  LIBS := \
    -lstdc++ \
    -lpthread \
    -lm \
    -ldl

endif
