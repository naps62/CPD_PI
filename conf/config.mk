#	Directories
SRCD	=	src
INCD	=	include
OBJD	=	obj
LIBD	=	lib
BIND	=	bin

#	Libraries
LIBS	=	fv

#	Compile mode
MODE	=	RLS

#	C Compil[ator]
CC		=	gcc

#	C++ Compil[ator]
CXX		=	g++

#	Include directories
INC		=	-I $(ROOTD)/$(INCD)

#	Compiler flags
CXXFLAGS	=	-Wall \
				-Wextra \
				-Wfatal-errors \
				-ansi \
				-pedantic \
				$(INC)
ifeq ($(MODE),DBG)
CXXFLAGS	+=	-g
SUFFIX=_$(MODE)
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
CFLAGS		=	$(CXXFLAGS)

#	Linker flags
LDFLAGS	=	-L $(ROOTD)/lib