# 編譯器：gcc（GCC OpenMP offload）或 nvc（NVIDIA HPC SDK）
#   make                              # host OpenMP
#   make offload                      # GCC GPU offload
#   make COMPILER=nvc                 # nvc host OpenMP
#   make COMPILER=nvc offload         # nvc GPU offload（建議 Linux + module load nvhpc）
#   make nvc-offload                  # 同上（捷徑）
#   make offload GPU_ARCH=80          # 指定 GPU 架構（nvc: cc80,sm_80；僅影響 offload）
COMPILER ?= gcc

TARGET  = fmm
SRCS    = main.c quadtree.c \
          caculate/binomial.c \
          caculate/P2M.c caculate/M2M.c caculate/M2L.c \
          caculate/L2L.c caculate/L2P.c caculate/P2P.c
OBJS    = $(SRCS:.c=.o)
HEADERS = fmm.h quadtree.h operators.h

# Detect nvc compiler automatically if available in PATH or in a common NVIDIA HPC SDK location.
NVC_BIN ?= $(shell command -v nvc 2>/dev/null)
ifeq ($(strip $(NVC_BIN)),)
  NVC_BIN := $(firstword $(wildcard /opt/nvidia/hpc_sdk/*/compilers/bin/nvc))
endif

ifeq ($(COMPILER),nvc)
  CC      := $(NVC_BIN)
  CFLAGS  := -std=c11 -O2 -mp -I.
  LDFLAGS := -mp -lm
  # NVIDIA HPC SDK：OpenMP target offload 至 GPU
  OFFLOAD_CFLAGS  := -mp=gpu
  OFFLOAD_LDFLAGS := -mp=gpu
  ifdef GPU_ARCH
    OFFLOAD_LDFLAGS += -gpu=cc$(GPU_ARCH),sm_$(GPU_ARCH)
  else
    OFFLOAD_LDFLAGS += -gpu=managed
  endif
else ifeq ($(COMPILER),gcc)
  # 偵測是否在 Mac (Darwin) 環境
  UNAME_S := $(shell uname -s)
  ifeq ($(UNAME_S),Darwin)
    CC    := /opt/homebrew/bin/gcc-15
  else
    CC    := gcc
  endif
  CFLAGS  := -std=c11 -Wall -Wextra -O2 -fopenmp -I.
  LDFLAGS := -fopenmp -lm
  OFFLOAD_TARGET  ?= nvptx64-nvidia-cuda
  OFFLOAD_CFLAGS  := -fopenmp-targets=$(OFFLOAD_TARGET)
  OFFLOAD_LDFLAGS := -fopenmp-targets=$(OFFLOAD_TARGET) -foffload=-lm
  ifneq (,$(findstring nvptx,$(OFFLOAD_TARGET)))
    OFFLOAD_LDFLAGS += -lnvptx -lnvidia-fatbinaryloader
  endif
else
  $(error Unsupported COMPILER=$(COMPILER). Use gcc or nvc.)
endif

.PHONY: all clean debug offload debug-offload nvc nvc-offload

all: $(TARGET)

offload: CFLAGS += $(OFFLOAD_CFLAGS)
offload: LDFLAGS += $(OFFLOAD_LDFLAGS)
offload: $(TARGET)

nvc: COMPILER := nvc
nvc: all

nvc-offload: COMPILER := nvc
nvc-offload: offload

debug: CFLAGS += -g -O0 -DDEBUG
debug: $(TARGET)

debug-offload: CFLAGS += -g -O0 -DDEBUG $(OFFLOAD_CFLAGS)
debug-offload: LDFLAGS += $(OFFLOAD_LDFLAGS)
debug-offload: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

ifeq ($(OS),Windows_NT)
  RM = del /Q
else
  RM = rm -f
endif

clean:
	$(RM) $(OBJS) $(TARGET)
