include ../polu.aos/Makefile

no_CXXFLAGS	=	-ansi -pedantic
CXXFLAGS	:=	$(filter-out $(no_CXXFLAGS),$(CXXFLAGS))
CXXFLAGS	+=	-DPROFILE_LIMITED=1000
CXXFLAGS	+=	-DPROFILE_WARMUP=100
