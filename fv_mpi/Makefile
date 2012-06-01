ROOTD	=	.
include $(ROOTD)/conf/config.mk

.PHONY:	doc default

default:	\
	libfv	\
	mpi.fvcm	\
	mpi.fvcd	\
	mpi.velocity	\
	mpi.sequential	\
	mpi.polu \
	mpi.polu.time0	\
	mpi.polu.time1	\
	mpi.polu.time2

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
