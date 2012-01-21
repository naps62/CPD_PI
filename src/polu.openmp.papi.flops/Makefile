ROOTD	=	../..
include $(ROOTD)/conf/src.bin.mk
noflags	=	-ansi -pedantic
CXXFLAGS	:=	$(filter-out $(noflags),$(CXXFLAGS)) -fopenmp

#	Extra dependencies
papi_hpp	=	papi main

$(papi_hpp:%=$(ROOTD)/$(OBJD)/$(BIN)/%.o):	papi.hpp
