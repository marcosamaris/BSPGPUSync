# (c) 2007 The Board of Trustees of the University of Illinois.

# Cuda-related definitions common to all benchmarks

########################################
# Variables
########################################

# Paths
CUDAHOME=/usr/local/MCUDA

CUDACC=mcc

CUDALINK=g++

CUDACFLAGS=-preproc-options "$(INCLUDEFLAGS)" -Xcompiler "-g -c"

CUDALDFLAGS=$(LDFLAGS) -L$(CUDAHOME)/lib64 -L$(PARBOIL_ROOT)/common/src $(EXTRA_CUDALDFLAGS)

CUDALIBS=-lpthread -lm -lmcuda -L$(CUDAHOME)/lib64 -L$(PARBOIL_ROOT)/common/lib $(LIBS)
