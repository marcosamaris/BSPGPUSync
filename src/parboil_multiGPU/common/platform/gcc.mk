# (c) 2007 The Board of Trustees of the University of Illinois.

# Rules common to all makefiles

# Commands to build objects from source file using C compiler
# with gcc

# gcc (default)
CC = gcc
EXTRA_CFLAGS = -c
  
CXX = g++
EXTRA_CXXFLAGS =
  
LINKER = gcc
EXTRA_LDFLAGS = -lm -lpthread

AR = ar
RANLIB = ranlib
  
