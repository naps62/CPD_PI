ROOTD	=	../..

TEMPLATE	=	polu.fvcpu.Makefile

include $(ROOTD)/conf/src.omp.mk

no_CXXFLAGS	=	-ansi -pedantic

CXXFLAGS	:=	$(filter-out $(no_CXXFLAGS),$(CXXFLAGS))

ifdef TIME
ifeq ($(TIME),ALL)
CXXFLAGS	+=	-DTIME_ALL
else
ifeq ($(TIME),MAIN)
CXXFLAGS	+=	-DTIME_MAIN
else
ifeq ($(TIME),ITERATION)
CXXFLAGS	+=	-DTIME_ITERATION
else
ifeq ($(TIME),FUNCTIONS)
CXXFLAGS	+=	-DTIME_FUNCTIONS
endif#	FUNCTIONS
endif#	ITERATION
endif#	MAIN
endif#	ALL
endif#	TIME
