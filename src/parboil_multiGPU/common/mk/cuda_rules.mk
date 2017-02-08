# (c) 2007 The Board of Trustees of the University of Illinois.

# Cuda-related rules common to all benchmarks

########################################
# Derived variables
########################################

CUDAOBJS = $(call INBUILDDIR,$(SRCDIR_CUDAOBJS))

########################################
# Rules-Xptxas -dlcm=cg
########################################

ifeq ("$(LINK_MODE)", "CUDA")
$(BIN) : $(OBJS) $(CUDAOBJS)
	PARBOIL_ROOT=$(PARBOIL_ROOT) make -C $(PARBOIL_ROOT)/common/src
	$(CUDALINK) -L/$(CUDAHOME)/lib64 -lcublas  $(CUDALDFLAGS) $^ -o $@ -lparboil_cuda $(CUDALIBS) 
endif

$(BUILDDIR)/%.o : $(SRCDIR)/%.cu
	mkdir -p $(BUILDDIR)
	#$(CUDACC) -Xptxas -dlcm=cg -use_fast_math -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35  $< $(CUDACFLAGS) # -o $@
	$(CUDACC) -ccbin gcc-4.8 -pg  -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35  $< $(CUDACFLAGS) # -o $@
	-mv $(basename $(notdir $<)).o $@

$(BUILDDIR)/%.ptx : $(SRCDIR)/%.cu
	mkdir -p $(BUILDDIR)
	#$(CUDACC) -Xptxas -dlcm=cg -use_fast_math -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35  $(CUDACFLAGS) -ptx $< -o $@
	$(CUDACC)  -pg -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35  $(CUDACFLAGS) -ptx $< -o $@

$(BUILDDIR)/%.cubin : $(SRCDIR)/%.cu
	mkdir -p $(BUILDDIR)
	#$(CUDACC) -Xptxas -dlcm=cg  -use_fast_math -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35  $(CUDACFLAGS) -cubin $< -o $@
	$(CUDACC)  -pg  -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35  $(CUDACFLAGS) -cubin $< -o $@
