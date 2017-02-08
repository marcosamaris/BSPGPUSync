PP_PATH=$(PARBOIL_ROOT)/common/platform

ifeq ("$(PLATFORM)","gcc")
    include $(PP_PATH)/gcc.mk
endif

ifeq ("$(PLATFORM)","icc")
    include $(PP_PATH)/icc.mk
endif

ifeq ("$(PLATFORM)", "cuda")
    include $(PP_PATH)/cuda.mk
endif

ifeq ("$(PLATFORM)", "mcuda")
    include $(PP_PATH)/mcuda.mk
endif

# default
ifeq ("$(PLATFORM)", "default")
    include $(PP_PATH)/gcc.mk
    include $(PP_PATH)/cuda.mk
endif

