ROOTD	=	../..

include $(ROOTD)/conf/config.mk

LIB		=	fv

include $(ROOTD)/conf/$(LIB).mk

$(ROOTD)/$(OBJD)/lib$(LIB)/%.o:	%.cpp
	$(COMPILE.cpp) $(OUTPUT_OPTION) $<

all:	$(OBJS:%=$(ROOTD)/$(OBJD)/lib$(LIB)/%)

clean:
	$(RM) $(OBJS:%=$(ROOTD)/$(OBJD)/lib$(LIB)/%)
