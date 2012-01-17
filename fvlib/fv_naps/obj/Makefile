ROOTD	=	..

include $(ROOTD)/conf/config.mk

DIRS	=	$(shell find . -maxdepth 1 -mindepth 1 -type d)

#	=	$(AUX:./%=%)
LIBS	=	$(filter lib%, $(DIRS:./%=%))
BINS	=	$(filter-out lib%, $(DIRS:./%=%))

.PHONY:	all $(DIRS)

all:	libs bins

libs:	$(LIBS)

bins:	$(BINS)

$(LIBS) $(BINS):
	#@echo ">>>>> $@"
	@cd $@; $_;
	#@echo "<<<<< $@"

clean:
	@for dir in $(BINS) $(LIBS);	\
	do	\
		echo ">>>>> $$dir";	\
		cd $$dir;	\
		$_ clean;	\
		cd ..;	\
		echo "<<<<< $$dir";	\
	done
