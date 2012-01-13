ROOTD	=	.
include $(ROOTD)/conf/config.mk

DATAD	=	data

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

data:
	@cd "$(DATAD)"; $_;

profile:
	@cd "$(DATAD)"; $_ $@;

polu polu%:
	@cd bin;	\
	./$@ "../data/xml/param.xml";	\
	if [ ! -d "../results" ];	\
	then	\
		mkdir "../results";	\
	fi;	\
	timestamp=`date "+%m-%d_%H:%M:%S"`;	\
	cp "polution.xml" "../results/$@.out.$${timestamp}"

clean:
	@echo ">>>>> $(OBJD)"
	@cd $(OBJD); $_ clean
	@echo "<<<<< $(OBJD)"
	@echo ">>>>> $(SRCD)"
	@cd $(SRCD); $_ clean
	@echo "<<<<< $(SRCD)"
