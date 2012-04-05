#####    cuda.Makefile    #####
# Makefile template for CUDA projects.
#
#	by Pedro Costa <dev@pfac.info>
#	January 2011

#	Mode flag
MODE	=	RLS

#	CUDA compiler
NVCC		=	nvcc

#	CUDA compiler flags
NVCCLAGS	=	
ifeq ($(MODE),DBG)
NVCCLAGS	+=	-g -G
else ifeq ($(MODE),RLS)
NVCCLAGS	+=	-O3
endif

#	Linker flags
LDFLAGS	=	-lcuda

#	Actions
COMPILE.cu	=	$(NVCC) $(NVCCLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c

.PHONY: clean

%:	%.cu
	$(LINK.cu) $^ $(LDLIBS) $(OUTPUT_OPTION)

%.o:	%.cu
	$(COMPILE.cu) $(OUTPUT_OPTION) $<

#clean:
#	$(RM) *.o
