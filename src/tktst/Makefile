ROOTD	=	../..

TEMPLATE	=	tk.Makefile

include $(ROOTD)/conf/src.mk

no_CXXFLAGS	=	-ansi -pedantic

CXXFLAGS	:=	$(filter-out $(no_CXXFLAGS),$(CXXFLAGS))
