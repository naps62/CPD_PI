LDFLAGS	+=	-lfv

ifdef BIN
$(ROOTD)/$(BIND)/$(BIN):	$(ROOTD)/$(LIBD)/libfv.a
endif
