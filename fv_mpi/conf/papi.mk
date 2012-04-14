no_CXXFLAGS	=	-ansi -pedantic
CXXFLAGS	:=	$(filter-out $(no_CXXFLAGS),$(CXXFLAGS))

#	Extra dependencies must be registered in the papi_hpp variable
$(papi_hpp:%=$(ROOTD)/$(OBJD)/$(BIN)/%.o):	$(INCD)/papi.hpp
