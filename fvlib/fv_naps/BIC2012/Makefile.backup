

ROOT=../FVLib

include $(ROOT)/config.mk

EXE=polu velocity
OBJ=polu.o velocity.o

default: polu
vim: default

EXES=polu velocity
OBJS=polu.o velocity.o
EXE_POLU=polu
EXE_VELOCITY=velocity

default: $(EXES)

$(EXES): $(OBJS)
	$(CXX) $(CXXFLAGS) $@.o -o $@ $(LDFLAGS)

clean:
	@echo 
	@echo --- Cleaning BIC2012 folder
	$(RM) *.o polu velocity
