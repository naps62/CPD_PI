#	Directories
#		root
ifndef ROOTD
ROOTD	=	.
endif
SRCD	=	$(ROOTD)/src
INCD	=	$(ROOTD)/include
OBJD	=	$(ROOTD)/obj
LIBD	=	$(ROOTD)/lib
BIND	=	$(ROOTD)/bin
TEMPLATED	=	$(ROOTD)/templates

DIR		=	$(shell pwd | egrep -o "[[:alnum:]_.:\-]+$$")

#	Compile mode
MODE	?=	RLS

#	C Compil[ator]
CC		=	gcc

#	C++ Compil[ator]
CXX		=	g++

#	Compiler flags

ifndef CXXFLAGS
CXXFLAGS	=	-Wall \
				-Wextra \
				-Wfatal-errors \
				-ansi \
				-pedantic	\
				-I$(INCD)
ifeq ($(MODE),DBG)
CXXFLAGS	+=	-g
else ifeq ($(MODE),GPROF)
CXXFLAGS	+=	-g -pg -O3
else ifeq ($(MODE),CALLGRIND)
CXXFLAGS	+=	-g -O2
else
CXXFLAGS	+=	-O3
SUFFIX=
endif
else
CXXFLAGS	:=	$(CXXFLAGS:-I%=-I$(INCD))
endif

ifdef no_CXXFLAGS
CXXFLAGS	:=	$(filter-out $(no_CXXFLAGS),$(CXXFLAGS))
endif

CFLAGS		=	$(CXXFLAGS)

#	Linker flags
LDFLAGS	=	-L$(LIBD)

export
