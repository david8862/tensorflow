# Settings for Bigeasy Video.
ifeq ($(TARGET),bev)
  TARGET_ARCH := armv7l
  TARGET_TOOLCHAIN_PREFIX := /opt/gcc-armv7l-meego-linux-gnueabi-2016.01_linux/cross/bin/armv7l-meego-linux-gnueabi-

  ifeq ($(TARGET_ARCH), armv7l)
    CXXFLAGS += \
      -march=armv7-a \
      -mfpu=vfpv3 \
      -mtune=cortex-a8 \
      -mlittle-endian \
      -mfloat-abi=softfp \
      -fPIC

    CCFLAGS += \
      -march=armv7-a \
      -mfpu=vfpv3 \
      -mtune=cortex-a8 \
      -mlittle-endian \
      -mfloat-abi=softfp \
      -fPIC

    LDFLAGS :=
  endif

  LIBS := \
    -lstdc++ \
    -lpthread \
    -lm \
    -ldl
endif
