ROOTD	=	../..

include $(ROOTD)/conf/obj.bin.mk

LDFLAGS	+=	-lfv -lfvcpu -fopenmp

$(ROOTD)/$(BIND)/$(BIN):	$(ROOTD)/$(LIBD)/libfv.a
$(ROOTD)/$(BIND)/$(BIN):	$(ROOTD)/$(LIBD)/libfvcpu.a
