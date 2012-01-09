ROOT	=	../FVLib

include $(ROOT)/config.mk

EXES	=	polu velocity
OBJS	=	polu.o velocity.o

default: $(EXES)

vim: default

.PHONY: $(EXES)
$(EXES): $(OBJS)
	$(CXX) $(CXXFLAGS) $@.o -o $@ $(LDFLAGS)


clean:
	@echo 
	@echo --- Cleaning BIC2012 folder
	$(RM) *.o $(EXES)
