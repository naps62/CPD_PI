#
# Makefile for test
#
FVLib_CONFIG=FVLib-config
#FVLib_CXX    =`$(FVLib_CONFIG) --cxx`
#FVLib_CFLAGS  =`$(FVLib_CONFIG) --cflags` -fopenmp -D _DEBUGS -g -pg
#FVLib_LIBS     =`$(FVLib_CONFIG) --libs` 
FVLib_CXX = g++
FVLib_CFLAGS=-O3 -I../FVLib/include
FVLib_LIBS= -L../FVLib/lib -lFVLib
OBJ_POLU = polu.o 
OBJ_VELOCITY = velocity.o 

polu:    $(OBJ_POLU) $(OBJ_VELOCITY)
	$(FVLib_CXX) $(FVLib_CFLAGS)  $(OBJ_POLU) -o polu  $(FVLib_LIBS)
	$(FVLib_CXX) $(FVLib_CFLAGS)  $(OBJ_VELOCITY) -o velocity  $(FVLib_LIBS)

clean:
	/bin/rm -f $(OBJ_POLU) $(OBJ_VELOCITY)  polu velocity

.SUFFIXES: .c .cc .cpp .cxx .o

.cpp.o:
	$(FVLib_CXX)$  $(FVLib_CFLAGS) -c $<

.cc.o:
	$(FVLib_CXX)   $(FVLib_CFLAGS) -c  $<
