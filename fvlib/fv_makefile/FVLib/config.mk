MODE	=	RELEASE

#	C++ Compil[ator]
CXX		=	g++

#	Include directories
INC		=	-I $(ROOT)/include

#	Compiler flags
CXXFLAGS	=	-Wall	\
				-Wextra	\
				-Wfatal-errors	\
				-ansi	\
				-pedantic	\
				$(INC)
ifeq ($(MODE),DBG)
CXXFLAGS	+=	-g
SUFFIX=_$(MODE)
else ifeq ($(MODE),GPROF)
CXXFLAGS	+=	-pg 
SUFFIX=_$(MODE)
else
CXXFLAGS	+=	-O3
SUFFIX=
endif

#	Linker flags
LDFLAGS	=	-L$(ROOT)/lib -lFVLib
