ROOTD	=	.
include $(ROOTD)/conf/config.mk

.PHONY:	doc test

%:
	@echo "<<==::    $@    ::==>>"
	@echo ">>>>> $(SRCD)"
	@cd $(SRCD); $_ $@
	@echo "<<<<< $(SRCD)"
	@echo ">>>>> $(OBJD)"
	@cd $(OBJD); $_ $@
	@echo "<<<<< $(OBJD)"

all: objs libs bins

doc:
	doxygen conf/Doxyfile

objs:
	@echo ">>>>> $(SRCD)"
	@cd $(SRCD); $_
	@echo "<<<<< $(SRCD)"

libs bins:
	@echo ">>>>> $(OBJD)";
	@cd "$(SRCD)"; $_ $@;
	@echo "<<<<< $(OBJD)";

test:\
	libfv\
	libfvcpu\
	polu.original.fnctime\
	polu.aos.fnctime\
	polu.aos.omp.fnctime\
	polu.soa.fnctime\
	polu.soa.omp.fnctime
