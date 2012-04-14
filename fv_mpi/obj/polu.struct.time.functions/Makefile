ROOTD	=	../..

include $(ROOTD)/conf/obj.bin.mk

LDFLAGS	+=	-lfv -lfvcpu -ltk

$(ROOTD)/$(BIND)/$(BIN):	$(ROOTD)/$(LIBD)/libfv.a
$(ROOTD)/$(BIND)/$(BIN):	$(ROOTD)/$(LIBD)/libfvcpu.a
$(ROOTD)/$(BIND)/$(BIN):	$(ROOTD)/$(LIBD)/libtk.a
