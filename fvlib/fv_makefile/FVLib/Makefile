#
# Makefile for FVlib
#
#include FV-config.conf #get the machine parameter and directorie

-include "config.mk"

.PHONY: src tools
default: src tools

src tools:
	@echo
	@echo --------------------
	@echo --- Entering $@
	@echo --------------------
	@cd $@; $_
	@echo --------------------
	@echo --- Done with $@
	@echo --------------------

clean:
	@echo 
	@echo --------------------
	@echo --- Entering src
	@echo --------------------
	cd src; $_ clean
	@echo
	@echo --------------------
	@echo --- Entering tools
	@echo --------------------
	cd tools; $_ clean
