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
INC		=	-I $(ROOTD)/$(INCD) -I $(CUDAD)/include

#	Compiler flags
CXXFLAGS	=	-arch sm_20 -Xcompiler="-fpermissive -Wall"
CXXFLAGS	+=	 $(INC)

ifeq ($(MODE),DBG)
CXXFLAGS	+=	-g -G
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
