#
# Makefile for tools
#
include ../FV-config.conf #get the machine parameter and directorie
OBJ=fvcm.o fvcd.o


tools: $(OBJ)
	$(CCC) $(CFLAGS)  fvcm.o -o fvcm  $(LIBS)
	$(CCC) $(CFLAGS)  fvcd.o -o fvcd  $(LIBS)
	mv fvcm ../bin/.
	mv fvcd ../bin/.
clean:  
	rm ../bin/fvcm
	rm ../bin/fvcd
	rm $(OBJ)

.SUFFIXES: .c .cc .cpp .cxx .o

.cpp.o:
	$(CCC)  $(CFLAGS) -c -Wall $<

.cc.o:
	$(CCC)  $(CFLAGS) -c  -Wall $<
