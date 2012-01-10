ROOTD	=	../..

include $(ROOTD)/conf/config.mk

BIN		=	fvcm

include $(ROOTD)/conf/simplebin.mk

$(ROOTD)/$(OBJD)/$(BIN)/%.o:	%.cpp
	$(COMPILE.cpp) $(OUTPUT_OPTION) $<

all:	$(OBJS:%=$(ROOTD)/$(OBJD)/$(BIN)/%)

clean:
	$(RM) $(OBJS:%=$(ROOTD)/$(OBJD)/$(BIN)/%)
