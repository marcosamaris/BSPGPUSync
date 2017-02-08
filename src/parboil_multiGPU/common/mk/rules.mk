# (c) 2007 The Board of Trustees of the University of Illinois.

# Rules common to all makefiles

########################################
# Environment variable check
########################################

# The second-last directory in the $(BUILDDIR) path
# must have the name "build".  This reduces the risk of terrible
# accidents if paths are not set up correctly.
ifeq ("$(notdir $(BUILDDIR))", "")
$(error $$BUILDDIR is not set correctly)
endif

ifneq ("$(notdir $(patsubst %/,%,$(dir $(BUILDDIR))))", "build")
$(error $$BUILDDIR is not set correctly)
endif

########################################
# Derived variables
########################################

OBJS = $(call INBUILDDIR,$(SRCDIR_OBJS))

########################################
# Rules
########################################

clean :
	rm -f $(BUILDDIR)/*
	if [ -d $(BUILDDIR) ]; then rmdir $(BUILDDIR); fi

.PHONY: $(BIN)

ifeq ("$(LINK_MODE)","C")
$(BIN) : $(OBJS)
	echo Platform = $(PLATFORM)
	PARBOIL_ROOT=$(PARBOIL_ROOT) make -C $(PARBOIL_ROOT)/common/src
	$(CC) $(LDFLAGS) $^ -o $@ -lparboil $(LIBS)
endif

ifeq ("$(LINK_MODE)","CPP")
$(BIN) : $(OBJS)
	echo Platform = $(PLATFORM)
	PARBOIL_ROOT=$(PARBOIL_ROOT) make -C $(PARBOIL_ROOT)/common/src
	$(CXX) $(LDFLAGS) $^ -o $@ -lparboil $(LIBS)
endif

include $(PARBOIL_ROOT)/common/platform/platform.mk

$(BUILDDIR) :
	mkdir $(BUILDDIR)

$(BUILDDIR)/%.o : $(SRCDIR)/%.c
	mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cc
	mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@
