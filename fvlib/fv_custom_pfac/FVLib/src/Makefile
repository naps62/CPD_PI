#
# Makefile for FVlib
#
include ../FV-config.conf #get the machine parameter and directorie

OBJ=XML.o Parameter.o Table.o FVio.o FVMesh1D.o FVMesh2D.o FVMesh3D.o Gmsh.o \
    FVGaussPoint.o FVStencil.o FVRecons1D.o FVRecons2D.o FVRecons3D.o  


makelib: $(OBJ)
	ar cru libFVLib.a $(OBJ)
	ranlib libFVLib.a
	mv libFVLib.a ../lib/.
clean:
	rm *.o


.SUFFIXES: .c .cc .cpp .cxx .o

.cpp.o:
	$(CCC)  $(CFLAGS) -c -Wall $<

.cc.o:
	$(CCC)  $(CFLAGS) -c  -Wall $<
