ROOTD	=	.
include $(ROOTD)/conf/config.mk

.PHONY:	test-soa

%:
	@echo "<<==::    $@    ::==>>"
	@echo ">>>>> $(SRCD)"
	@cd $(SRCD); $_ $@
	@echo "<<<<< $(SRCD)"
	@echo ">>>>> $(OBJD)"
	@cd $(OBJD); $_ $@
	@echo "<<<<< $(OBJD)"
all:
	$_ $@

objs:
	@echo ">>>>> $(SRCD)"
	@cd $(SRCD); $_
	@echo "<<<<< $(SRCD)"

test-soa:	\
	libfv	\
	libpapipcc	\
	polu.soa.brins	\
	polu.soa.btm	\
	polu.soa.flops	\
	polu.soa.l1dcm	\
	polu.soa.l2dcm	\
	polu.soa.ldins	\
	polu.soa.srins	\
	polu.soa.totins	\
	polu.soa.vecins
