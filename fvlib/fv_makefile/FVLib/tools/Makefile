#
# Makefile for tools
#
#include ../FV-config.conf #get the machine parameter and directorie

ROOT	=	..

include $(ROOT)/config.mk

EXE_DIR	=	../bin
EXES	=	fvcd fvcm
OBJS	=	fvcm.o fvcd.o

default:	$(EXES)
	mv $(EXES) $(EXE_DIR)

$(EXES): $(OBJS)
	$(CXX) $(CXXFLAGS) $@.o -o $@ $(LDFLAGS)

#%: %.o
#	@echo
#	@echo --- Building $@
#	$(CXX) -o $@ $^ $(LDFLAGS)
#	mv $@ $(EXE_DIR)/$@

#	for exe in $^;	\
#	do	\
#		mv "$${exe}" "../bin/$${exe}$(SUFFIX)";	\
#	done

clean:
	@echo
	@echo --- Cleaning tools folder
	$(RM) *.o
	for exe in $(EXES); do $(RM) $${exe}; done
