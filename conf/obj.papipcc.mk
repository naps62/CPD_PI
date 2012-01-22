LDFLAGS	+=	-lpapi -lpapipcc

ifdef BIN
$(ROOTD)/$(BIND)/$(BIN):	$(ROOTD)/$(LIBD)/libpapipcc.a
endif
