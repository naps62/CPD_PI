include $(ROOTD)/conf/bin.mk
include $(ROOTD)/conf/obj.mk

$(BIND)/$(DIR):	$(OBJS)
	@if [ ! -d "$(@D)" ];	\
	then	\
		mkdir "$(@D)";	\
	fi
	$(CXX) $(OUTPUT_OPTION) $^ $(LOADLIBES) $(LDLIBS) $(LDFLAGS) $(TARGET_ARCH)

all:	$(BIND)/$(DIR)

clean:
	$(RM) $(BIND)/$(DIR)
