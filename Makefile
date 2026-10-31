NVCC     ?= nvcc
CXX      ?= g++
# GPU architecture. "native" needs CUDA >= 11.5; otherwise e.g. make ARCH=sm_75
ARCH     ?= native
NVFLAGS  = -O3 -std=c++14 -arch=$(ARCH)
CXXFLAGS = -O2 -std=c++14 -Wall -Wextra

all: matmul compare

matmul: matmul.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

compare: compare.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

test: all
	bash run_tests.sh

bench: all
	bash bench.sh

clean:
	rm -f matmul compare

.PHONY: all test bench clean
