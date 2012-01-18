ROOTD	=	..

include $(ROOTD)/conf/config.mk

DIRS	=	$(shell find . -mindepth 1 -maxdepth 1 -type d)

.PHONY:	all $(DIRS)

all:	$(DIRS)

$(DIRS):
#	@echo ">>>>> $@"
	@cd $@; $_;
#	@echo "<<<<< $@"

clean:
	@for dir in $(DIRS);	\
	do	\
#		echo ">>>>> $$dir";	\
		cd $$dir;	\
		$_ clean;	\
		cd ..;	\
#		echo "<<<<< $$dir";	\
	done
