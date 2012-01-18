include $(ROOTD)/conf/obj.bin.mk

LDFLAGS	+=	-lfv -lcuda

$(ROOTD)/$(BIND)/$(BIN):	$(ROOTD)/$(LIBD)/libfv.a
