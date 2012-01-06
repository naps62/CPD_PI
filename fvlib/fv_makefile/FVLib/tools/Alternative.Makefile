#
# Makefile for tools
#
#include ../FV-config.conf #get the machine parameter and directorie

ROOT	=	../

-include $(ROOT)config.mk

EXES	=	fvcd fvcm

OBJS	=	fvcm.o fvcd.o

FVCD_OBJ	=	fvcd.o
FVCM_OBJ	=	fvcm.o
FVCD_EXE	=	$(ROOT)bin/fvcd$(SUFFIX)
FVCM_EXE	=	$(ROOT)bin/fvcm$(SUFFIX)

default:	$(EXES)
	for exe in $^;	\
	do	\
		mv "$${exe}" "../bin/$${exe}$(SUFFIX)";	\
	done

clean:
	$(RM) *.o
