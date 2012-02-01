ROOTD	=	../..

include $(ROOTD)/conf/obj.bin.mk

LDFLAGS	+=	-lfv -lfvcpu

$(ROOTD)/$(BIND)/$(BIN):	$(ROOTD)/$(LIBD)/libfv.a
