ROOTD	=	..

include $(ROOTD)/conf/config.mk

DIRS	=	$(shell find . -mindepth 1 -maxdepth 1 -type d)
OLD_DIRS	=	$(shell find . -mindepth 1 -maxdepth 1 -type d -name "old*")
IGN_DIRS	=	$(shell find . -mindepth 1 -maxdepth 1 -type d -name "ign*")
DIRS	:=	$(filter-out $(OLD_DIRS), $(DIRS))
DIRS	:=	$(filter-out $(IGN_DIRS), $(DIRS))

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
