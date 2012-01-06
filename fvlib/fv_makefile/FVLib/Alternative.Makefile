#
# Makefile for FVlib
#
#include FV-config.conf #get the machine parameter and directorie

-include "config.mk"

.PHONY: src tools
default: src tools

src tools:
	cd $@; $_ --file="Alternative.Makefile"

clean:
	cd src; $_ --file="Alternative.Makefile" clean
	cd tools; $_ --file="Alternative.Makefile" clean
