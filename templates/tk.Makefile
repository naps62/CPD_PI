ROOTD	=	../..

include $(ROOTD)/conf/obj.bin.mk

LDFLAGS	+=	-ltk

$(ROOTD)/$(BIND)/$(BIN):	$(ROOTD)/$(LIBD)/libtk.a
