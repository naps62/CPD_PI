#
# Makefile for FVlib
#
#include ../FV-config.conf #get the machine parameter and directorie

ROOT	=	..

-include $(ROOT)/config.mk

OBJS	=	XML.o	\
			Parameter.o	\
			Table.o	\
			FVio.o	\
			FVMesh1D.o	\
			FVMesh2D.o	\
			FVMesh3D.o	\
			Gmsh.o	\
			FVGaussPoint.o	\
			FVStencil.o	\
			FVRecons1D.o	\
			FVRecons2D.o	\
			FVRecons3D.o

LIB_A	=	../lib/libFVLib.a

default: 	$(OBJS)
	ar cru $(LIB_A) $(OBJS)
	ranlib $(LIB_A)

clean:
	$(RM) *.o
