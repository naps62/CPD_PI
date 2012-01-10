#####    cuda.Makefile    #####
# Makefile template for CUDA projects.
#
#	by Pedro Costa <dev@pfac.info>
#	January 2011

#	Mode flag
MODE	=	RLS

#	CUDA compiler
CUC		=	nvcc

#	CUDA compiler flags
CUFLAGS	=	
ifeq ($(MODE),DBG)
CUFLAGS	+=	-g
else ifeq ($(MODE),RLS)
CUFLAGS	+=	-O3
endif

#	Linker flags
LDFLAGS	=	-lcuda

#	Actions
COMPILE.cu	=	$(CUC) $(CUFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c

.PHONY: clean

%:	%.cu
	$(LINK.cu) $^ $(LDLIBS) $(OUTPUT_OPTION)

%.o:	%.cu
	$(COMPILE.cu) $(OUTPUT_OPTION) $<

clean:
	$(RM) *.o
