ROOTD	=	../..

include $(ROOTD)/conf/obj.bin.mk

LDFLAGS	+=	-lfv -lfvcpu -lpapi -lpapipcc

$(ROOTD)/$(BIND)/$(BIN):	$(ROOTD)/$(LIBD)/libfv.a
$(ROOTD)/$(BIND)/$(BIN):	$(ROOTD)/$(LIBD)/libfvcpu.a
$(ROOTD)/$(BIND)/$(BIN):	$(ROOTD)/$(LIBD)/libpapipcc.a
