ROOTD	=	../..

include $(ROOTD)/conf/config.mk

BIN		=	fvcd

include $(ROOTD)/conf/simplefv.mk

$(ROOTD)/$(BIND)/$(BIN):	$(OBJS)
	$(LINK.o) $^ $(LOADLIBES) $(LDLIBS) $(OUTPUT_OPTION)

all:	$(BIN:%=$(ROOTD)/$(BIND)/%)

clean:
	$(RM) $(BIN:%=$(ROOTD)/$(BIND)/%)
