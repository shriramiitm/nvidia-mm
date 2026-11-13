NVCC     ?= nvcc
CXX      ?= g++
# GPU architecture. "native" needs CUDA >= 11.5; otherwise e.g. make ARCH=sm_75
ARCH     ?= native
NVFLAGS  = -O3 -std=c++14 -arch=$(ARCH)
CXXFLAGS = -O2 -std=c++14 -Wall -Wextra

all: matmul compare

matmul: src/matmul.cu
	$(NVCC) $(NVFLAGS) -o $@ $< -lcublas   # cuBLAS only for the reference (--impl cublas)

compare: src/compare.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

test: all
	bash scripts/run_tests.sh

bench: all
	bash scripts/bench.sh

clean:
	rm -f matmul compare

.PHONY: all test bench profile reproduce clean

profile: all
	bash scripts/profile.sh

reproduce:
	bash scripts/reproduce.sh
