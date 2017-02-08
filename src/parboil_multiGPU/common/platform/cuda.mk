# (c) 2007 The Board of Trustees of the University of Illinois.

# Cuda-related definitions common to all benchmarks

########################################
# Variables
########################################

# Paths
CUDAHOME=/usr/local/cuda

# Programs
CUDACC=nvcc

CUDALINK=nvcc

# Flags
CUDACFLAGS=$(INCLUDEFLAGS) -O3 -Xcompiler "-m32" -c $(EXTRA_CUDACFLAGS)

CUDALDFLAGS=$(LDFLAGS)                                    \
	-lcuda -L$(CUDA)/lib -L$(CUDAHOME)/lib64 -L$(PARBOIL_ROOT)/common/lib $(EXTRA_CUDALDFLAGS)

CUDALIBS=-lcuda $(LIBS)
