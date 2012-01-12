include $(ROOTD)/conf/src.mk

$(ROOTD)/$(OBJD)/lib$(LIB)/%.o:	%.cpp
	@if [ ! -d "$(@D)" ];	\
	then	\
		mkdir "$(@D)";	\
	fi
	$(COMPILE.cpp) $(OUTPUT_OPTION) $<

$(ROOTD)/$(OBJD)/lib$(LIB)/%.o:	%.cc
	@if [ ! -d "$(@D)" ];	\
	then	\
		mkdir "$(@D)";	\
	fi
	$(COMPILE.cc) $(OUTPUT_OPTION) $<

default: echo all

echo:
	@echo $(OBJS)

all:	$(OBJS:%=$(ROOTD)/$(OBJD)/lib$(LIB)/%)

clean:
	$(RM) $(OBJS:%=$(ROOTD)/$(OBJD)/lib$(LIB)/%)

