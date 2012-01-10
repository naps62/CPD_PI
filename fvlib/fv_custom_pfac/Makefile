include conf/config.mk

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
