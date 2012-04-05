include $(ROOTD)/conf/bin.mk
include $(ROOTD)/conf/obj.mk

$(ROOTD)/$(BIND)/$(BIN):	$(OBJS)
	@if [ ! -d "$(@D)" ];	\
	then	\
		mkdir "$(@D)";	\
	fi
	$(CXX) $(OUTPUT_OPTION) $^ $(LOADLIBES) $(LDLIBS) $(LDFLAGS) $(TARGET_ARCH)

all:	$(BIN:%=$(ROOTD)/$(BIND)/%)

clean:
	$(RM) $(BIN:%=$(ROOTD)/$(BIND)/%)
