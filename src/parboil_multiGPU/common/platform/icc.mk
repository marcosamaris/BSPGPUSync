# (c) 2007 The Board of Trustees of the University of Illinois.

# Rules common to all makefiles

# Commands to build objects from source file using C compiler
# with gcc

# gcc (default)
CC = icc
EXTRA_CFLAGS = -c
  
CXX = icc++
EXTRA_CXXFLAGS =
  
LINKER = icc
EXTRA_LDFLAGS = -lm -lpthread

AR = ar
RANLIB = ranlib
  
