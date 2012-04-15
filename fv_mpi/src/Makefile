ROOTD	=	..

include $(ROOTD)/conf/config.mk

DIRS	=	$(shell find . -mindepth 1 -maxdepth 1 -type d)
LIBS	=	$(filter lib%,$(DIRS))
BINS	=	$(filter-out $(LIBS),$(DIRS))

.PHONY:	all $(DIRS)

all:	libs bins

bins:	$(BINS)

libs:	$(LIBS)

$(DIRS) $(LINKS):
	@echo ">>>>> $@"
	@cd $@; $_;
	@echo "<<<<< $@"

clean:
	@for dir in $(DIRS);	\
	do	\
		echo ">>>>> $$dir";	\
		cd $$dir;	\
		$_ clean;	\
		cd ..;	\
		echo "<<<<< $$dir";	\
	done
