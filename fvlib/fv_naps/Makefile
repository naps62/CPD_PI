ROOTD=.

include conf/config.mk

default: all symlinks

all:	\
	objs	\
	libs	\
	bins

objs:
	@echo ">>>>> $(SRCD)"
	@cd $(SRCD); $_
	@echo "<<<<< $(SRCD)"

libs bins:
	@echo ">>>>> $(OBJD)"
	@cd $(OBJD); $_ $@
	@echo "<<<<< $(OBJD)"

clean:
	@echo ">>>>> $(OBJD)"
	@cd $(OBJD); $_ clean
	@echo "<<<<< $(OBJD)"
	@echo ">>>>> $(SRCD)"
	@cd $(SRCD); $_ clean
	@echo "<<<<< $(SRCD)"

symlinks:
	@echo Creating symlinks
	@for file_name in `ls $(PWD)/bin` ; \
		do \
		ln -vfs $(PWD)/bin/$$file_name ../simulations/$$file_name; \
		done
	@echo $(PWD) > ../simulations/PATH
	@echo $(MODE) > ../simulations/MODE
