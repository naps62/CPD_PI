#!/bin/bash

if [ $# -eq 0 ]; then

	echo <<EOF
Usage:
	run <lib> <exec> "<args>"

	Example: lib fv_default fvcm "
EOF
fi

SCRIPT_DIR=scripts
GPROF2DOT=$SCRIPT_DIR/gprof2dot.py

case $1 in
	"fvcd","fvcm")
		EXE_DIR=FVLibs/$1
		;;
	*)
		EXE_DIR=BIC2012/$1
		;;
esac
EXE_NAME=$2


