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
XCOMPFLAGS	=	-Wall -Wextra
CXXFLAGS	=	-arch sm_20
CXXFLAGS	+=	 $(INC)

HOST	= $(shell hostname)
ifeq ($(MODE),naps62crunch)
CXXFLAGS	+=	-DNO_CUDA=1
endif

ifeq ($(MODE),DBG)
XCOMPFLAGS	+=	-g
CXXFLAGS	+=	-g -G
else
XCOMPFLAGS	+=	-O3
CXXFLAGS	+=	-O3
endif

CXXFLAGS	+= -Xcompiler="$(XCOMPFLAGS)"

#	Linker flags
LDFLAGS	=	-L $(ROOTD)/lib

default: all

vim:
	@cd $(ROOTD); $_
