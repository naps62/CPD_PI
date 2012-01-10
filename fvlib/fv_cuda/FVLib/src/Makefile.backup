#
# Makefile for FVlib
#
#include ../FV-config.conf #get the machine parameter and directorie



#makelib: $(OBJ)
#	ar cru libFVLib.a $(OBJ)
#	ranlib libFVLib.a
#	mv libFVLib.a ../lib/.
#clean:
#	rm *.o


#.SUFFIXES: .c .cc .cpp .cxx .o

ifdef ALLOW

OBJ=XML.o Parameter.o Table.o FVio.o FVMesh1D.o FVMesh2D.o FVMesh3D.o Gmsh.o \
    FVGaussPoint.o FVStencil.o FVRecons1D.o FVRecons2D.o FVRecons3D.o  

LIB_A=../lib/libFVLib.a

default: $(OBJ)
	ar cru $(LIB_A) $(OBJ)
	ranlib $(LIB_A)

.cpp.o:
	$(CC) $(CFLAGS) $(INC) -c $<

else

default:
	@echo "Please go to ../ and call the top-level Makefile"

endif

clean:
	$(RM) *.o

#.cc.o:
#	$(CCC)  $(CFLAGS) -c  -Wall $<
