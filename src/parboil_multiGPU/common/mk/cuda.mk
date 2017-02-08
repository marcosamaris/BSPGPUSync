# (c) 2007 The Board of Trustees of the University of Illinois.

# Cuda-related definitions common to all benchmarks

########################################
# Variables
########################################

# Paths
CUDAHOME=/usr/local/cuda

# Programs
CUDACC=nvcc

# Flags
CUDACFLAGS=$(INCLUDEFLAGS)  -Xcompiler -Xptxas -dlcm=cg  $(EXTRA_CUDACFLAGS) -use_fast_math -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20
#CUDACFLAGS=$(INCLUDEFLAGS) -pg  -Xcompiler  $(EXTRA_CUDACFLAGS) -use_fast_math -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20
CUDALDFLAGS=$(LDFLAGS) -Xcompiler                                     \
	-L$(CUDAHOME)/lib64 -L$(PARBOIL_ROOT)/common/src $(EXTRA_CUDALDFLAGS) -lpthread -lcublas
CUDALIBS=-lcuda -lcublas $(LIBS)
