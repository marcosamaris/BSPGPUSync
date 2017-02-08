# (c) 2007 The Board of Trustees of the University of Illinois.

SRCDIR=$(PARBOIL_ROOT)/common/src
BUILD_DIR=$(PARBOIL_ROOT)/common/lib

include $(PARBOIL_ROOT)/common/mk/common.mk
include $(PARBOIL_ROOT)/common/mk/cuda.mk

OBJS=$(BUILD_DIR)/parboil.o
OBJS_TWO=$(BUILD_DIR)/parboil_cuda.o
TARGET=$(BUILD_DIR)/libparboil.a 
TARGET_TWO=$(BUILD_DIR)/libparboil_cuda.a

.PHONY: all clean

all : $(TARGET) $(TARGET_TWO)

clean:
	rm -f $(OBJS) $(TARGET) $(OBJS_TWO) $(TARGET_TWO)

${BUILD_DIR}/%.o : $(SRCDIR)/%.c
	$(CC) -c -g $(CFLAGS) $^ -o $@

$(TARGET) : $(OBJS)
	$(AR) rc $@ $?
	$(RANLIB) $@

$(TARGET_TWO) : $(OBJS_TWO)
	$(AR) rc $@ $?
	$(RANLIB) $@
