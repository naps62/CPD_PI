#!/bin/bash

if [ $# -eq 0 ]; then

	echo <<EOF
	Usage:
		
		run <lib> <exec> <args>
		OR
		gdb <lib> <exec> <args>
		OR
		gprof <lib> <exec> <args>
	
		Example:		   run fv_default fvcd foz.xml velocity.xml velocity.msh -c
		will trigger:	fv_default/FVLibs/fvcd foz.xml velocity.xml velocity.msh -c
EOF
fi

# READ ARGS
RUN_TYPE=$0
SCRIPT_DIR=scripts
GPROF2DOT=$SCRIPT_DIR/gprof2dot.py
LIB=$1
EXE=$2
ARGS=${*:3}

# CREATE OUTPUT_DIR
TIMESTAMP=`date +"%D-%H:%M:%S"`
OUTPUT_DIR=data/${LIB}_$TIMESTAMP
mkdir $OUTPUT_DIR

# SET EXEC PATH
case $1 in
	"fvcd","fvcm") EXE=FVLibs/$1
		;;
	*) EXE=BIC2012/$1
		;;
esac

# if mode is gprof, append _gprof to exec
if [ $RUN_TYPE -eq "gprof" ]; then
	EXE=$EXE_gprof
fi

CMD="$LIB/$EXE $ARGS"
echo "Running: " $CMD
`$CMD`

# if mode is gprof, create gprof data
if [$RUN_TYPE -eq "gprof" ]; then
	echo "Running gprof"
	GPROF_DIR=$OUTPU_DIR/gprof_data
	mkdir $OUTPUT_DIR/gprod_data
	# run gprof
	gprof $EXE > $OUTPUT_DIR/gprof
	# also run call graph generator
	echo "Generating gprof call graph"
	$SCRIPT_DIR/gprof2dot.py < $OUTPUT_DIR/gprof | dot -Tpng -o $OUTPUT_DIR/gprof_dot.png
fi
