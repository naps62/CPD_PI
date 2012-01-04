export PATH:=$(PWD)/common:$(PATH)

DIRS=$(shell find . -type d -name "fv_*")
SUB_MAKE=../common/Makefile

default: all

all: fv_*
	for f in $(DIRS); do (make sub_lib --directory=$$f --file=$(SUB_MAKE)); done

clean:
	@read -p "Are you sure? [yn] " r; \
	if [ $$r = "y" ]; then	\
		for f in $(DIRS); do (make clean --directory=$$f --file=$(SUB_MAKE)); done;	\
	else	\
		echo "Aborted";	\
	fi

