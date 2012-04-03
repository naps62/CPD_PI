ROOTD	=	..

include $(ROOTD)/conf/config.mk

DIRS	=	$(shell find . -mindepth 1 -maxdepth 1 -type d)
NO_DIRS	=	$(shell find . -mindepth 1 -maxdepth 1 -type d -name "old*")
DIRS	:=	$(filter-out $(NO_DIRS), $(DIRS))

.PHONY:	all $(DIRS)

all:	$(DIRS)
	echo $(DIRS)

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
