#
# Makefile for tools
#
#include ../FV-config.conf #get the machine parameter and directorie

ROOT	=	..

include $(ROOT)/config.mk

EXE_DIR	=	../bin
EXES	=	$(EXE_DIR)/fvcd $(EXE_DIR)/fvcm
OBJS	=	fvcm.o fvcd.o

default:	$(EXES)

$(EXES): $(OBJS)
	@echo
	@echo --- Building $@
	$(CXX) $(CXXFLAGS) $@.o -o $@ $(LDFLAGS)
	mv $@ ../bin/
#	for exe in $^;	\
#	do	\
#		mv "$${exe}" "../bin/$${exe}$(SUFFIX)";	\
#	done

clean:
	@echo
	@echo --- Cleaning tools folder
	$(RM) *.o
