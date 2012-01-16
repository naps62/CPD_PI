include $(ROOTD)/conf/bin.mk
include $(ROOTD)/conf/src.mk

$(ROOTD)/$(OBJD)/$(BIN)/%.o:	%.cpp
	@if [ ! -d "$(@D)" ];	\
		then mkdir "$(@D)";	\
	fi
	$(COMPILE.cpp) $(OUTPUT_OPTION) $<

$(ROOTD)/$(OBJD)/$(BIN)/%.o:	%.cc
	@if [ ! -d "$(@D)" ];	\
		then mkdir "$(@D)";	\
	fi
	$(COMPILE.cc) $(OUTPUT_OPTION) $<

all:	$(OBJS:%=$(ROOTD)/$(OBJD)/$(BIN)/%)

clean:
	$(RM) $(OBJS:%=$(ROOTD)/$(OBJD)/$(BIN)/%)
