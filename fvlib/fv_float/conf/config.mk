#	Directories
SRCD	=	src
INCD	=	include
OBJD	=	obj
LIBD	=	lib
BIND	=	bin
CUDAD	=	/usr/local/cuda

#	Libraries
LIBS	=	fv cuda

#	Compile mode
MODE	=	RLS

#	C++ Compil[ator]
CXX	=	nvcc
#CXX		=	g++

#	Include directories
INC		=	-I $(ROOTD)/$(INCD) -I $(CUDAD)/include -I $(CUDASDK_DIR)/C/common/inc

#	Compiler flags
CXXFLAGS	=	 -Xcompiler="-fpermissive -Wall -Wextra -Wfatal-errors"
CXXFLAGS	+=	 $(INC)

ifeq ($(MODE),DBG)
CXXFLAGS	+=	-g -G
NVCCFLAGS	+=	-g -G
SUFFIX=_$(MODE)
else ifeq ($($MODE),CUBIN)
CXXFLAGS	+= -cubin
else ifeq ($(MODE),GPROF)
CXXFLAGS	+=	-g -pg -O3
SUFFIX=_$(MODE)
else ifeq ($(MODE),CALLGRIND)
CXXFLAGS	+=	-g -O2
SUFFIX=_$(MODE)
else
CXXFLAGS	+=	-O3
SUFFIX=
endif

#	Linker flags
LDFLAGS	=	-L $(ROOTD)/lib

default: all

vim:
	@cd $(ROOTD); $_
