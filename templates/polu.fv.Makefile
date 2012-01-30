include $(ROOTD)/conf/obj.bin.mk

LDFLAGS	+=	-lfv

$(ROOTD)/$(BIND)/$(BIN):	$(ROOTD)/$(LIBD)/libfv.a
