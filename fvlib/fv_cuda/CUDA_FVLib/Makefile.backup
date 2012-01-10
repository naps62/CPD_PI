#
# Makefile for FVlib
#
#include FV-config.conf #get the machine parameter and directorie

MODE=RELEASE

LIB=-L $(PWD)/lib -lFVLib
INC=-I $(PWD)/include

CC=g++

CFLAGS_COMMON=-Wall -Wextra
CFLAGS_RLS=-O3
CFLAGS_DBG=-g
CFLAGS_GPROF=-g -gprof

ifeq ($(MODE),DBG)
CFLAGS=$(CFLAGS_COMMON) $(CFLAGS_DBG)
SUFFIX=_$(MODE)
else ifeq ($(MODE),GPROF)
CFLAGS=$(CFLAGS_COMMON) $(CFLAGS_GPROF)
SUFFIX=_$(MODE)
else
CFLAGS=$(CFLAGS_COMMON) $(CFLAGS_RLS)
SUFFIX=
endif

.PHONY: src tools
default: src tools

src tools:
	cd $@; make ALLOW=1 CC=$(CC) CFLAGS="$(CFLAGS)" SUFFIX=$(SUFFIX) INC="$(INC)" LIB="$(LIB)"

clean:
	cd src; make clean
	cd tools; make clean

#makelib:
#	cd src && make 
#	cd tools && make
#clean:
#	cd src && make clean
#	cd tools && make clean
