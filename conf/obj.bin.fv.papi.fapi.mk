include $(ROOTD)/conf/obj.bin.mk

LDFLAGS	+=	-lfv -lpapi -lfapi

$(ROOTD)/$(BIND)/$(BIN):	$(ROOTD)/$(LIBD)/libfv.a
$(ROOTD)/$(BIND)/$(BIN):	$(ROOTD)/$(LIBD)/libfapi.a
