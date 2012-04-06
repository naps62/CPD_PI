ROOTD	=	.
include $(ROOTD)/conf/config.mk

.PHONY:	doc test-soa test-aos

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

doc:
	doxygen conf/Doxyfile

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
	polu.soa.fnctime	\
	polu.soa.fpins	\
	polu.soa.l1ca	\
	polu.soa.l2dca	\
	polu.soa.l2ica	\
	polu.soa.l2tca	\
	polu.soa.l1dcm	\
	polu.soa.l1icm	\
	polu.soa.l1tcm	\
	polu.soa.l2cm	\
	polu.soa.ldins	\
	polu.soa.srins	\
	polu.soa.totins	\
	polu.soa.tottime	\
	polu.soa.vecins	\
	polu.soa.omp.fnctime	\
	polu.soa.omp.tottime

test-aos:	\
	libfv	\
	libfvcpu	\
	libpapipcc	\
	polu.aos.brins	\
	polu.aos.btm	\
	polu.aos.flops	\
	polu.aos.fnctime	\
	polu.aos.fpins	\
	polu.aos.l1ca	\
	polu.aos.l2dca	\
	polu.aos.l2ica	\
	polu.aos.l2tca	\
	polu.aos.l1dcm	\
	polu.aos.l1icm	\
	polu.aos.l1tcm	\
	polu.aos.l2cm	\
	polu.aos.ldins	\
	polu.aos.srins	\
	polu.aos.totins	\
	polu.aos.tottime	\
	polu.aos.vecins	\
	polu.aos.omp.fnctime	\
	polu.aos.omp.tottime
