include $(ROOTD)/conf/bin.mk
include $(ROOTD)/conf/src.mk

$(ROOTD)/$(OBJD)/$(BIN)/%.o:	%.cpp
	@if [ ! -d "$(@D)" ];	\
	then	\
		mkdir "$(@D)";	\
		cd "$(@D)";	\
		cp "$(ROOTD)/templates/Makefile.obj.bin" Makefile;	\
	fi
	$(COMPILE.cpp) $(OUTPUT_OPTION) $<

$(ROOTD)/$(OBJD)/$(BIN)/%.o:	%.cc
	@if [ ! -d "$(@D)" ];	\
	then	\
		mkdir "$(@D)";	\
		cd "$(@D)";	\
		cp "$(ROOTD)/templates/Makefile.obj.bin" Makefile;	\
	fi
	$(COMPILE.cc) $(OUTPUT_OPTION) $<


$(ROOTD)/$(OBJD)/$(BIN)/%.o:	%.c
	@if [ ! -d "$(@D)" ];	\
	then	\
		mkdir "$(@D)";	\
		cd "$(@D)";	\
		cp "$(ROOTD)/templates/Makefile.obj.bin" Makefile;	\
	fi
	$(COMPILE.c) $(OUTPUT_OPTION) $<

all:	$(OBJS:%=$(ROOTD)/$(OBJD)/$(BIN)/%)

clean:
	$(RM) $(ROOTD)/$(OBJD)/$(BIN)/*.o
#	$(RM) $(OBJS:%=$(ROOTD)/$(OBJD)/$(BIN)/%)
