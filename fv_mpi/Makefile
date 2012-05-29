ROOTD	=	.
include $(ROOTD)/conf/config.mk

.PHONY:	doc default

default:	\
	libfv	\
	mpi.fvcm	\
	mpi.fvcd	\
	mpi.velocity	\
	mpi.sequential	\
	mpi.polu

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
