#
# Makefile for tools
#
#include ../FV-config.conf #get the machine parameter and directorie

ifdef ALLOW

OBJ=fvcm.o fvcd.o

FVCD_OBJ=fvcd.o
FVCM_OBJ=fvcm.o
FVCD_EXE=../bin/fvcd$(SUFFIX)
FVCM_EXE=../bin/fvcm$(SUFFIX)

#tools: $(OBJ)
#	$(CCC) $(CFLAGS)  fvcm.o -o fvcm  $(LIBS)
#	$(CCC) $(CFLAGS)  fvcd.o -o fvcd  $(LIBS)
#	mv fvcm ../bin/.
#	mv fvcd ../bin/.
#clean:  
#	rm ../bin/fvcm
#	rm ../bin/fvcd
#	rm $(OBJ)

#.SUFFIXES: .c .cc .cpp .cxx .o

default: $(OBJ)
	$(CC) $(CFLAGS) $(FVCD_OBJ) -o $(FVCD_EXE) $(LIB)
	$(CC) $(CFLAGS) $(FVCM_OBJ) -o $(FVCM_EXE) $(LIB)

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
