#!/bin/bash

if [ $# == "0" ]; then
	echo <<EOF
	Usage:
		simul_callgrind.sh <folder>
		folder - folder name to be created inside simulations to generate input/output
		folder/callgrind will also be created to save gprof results
EOF
	exit 1
fi

MODE=$(cat MODE)
REQUIRED="CALLGRIND"
if [ $MODE != $REQUIRED ]; then
	echo "requires mode: $REQUIRED"
	echo "current  mode: $MODE"
	read -p "recompile? [yn] " yn
	case $yn in
		[Yy])
			(cd $(cat PATH); make clean; make MODE=$REQUIRED)
			;;
		*)
			echo "aborting"
			exit 1
	esac
fi


TIMESTAMP=$(date +%m.%d_%H:%M:%S)

ROOT=.
OUTPUT_DIR=$ROOT/$MODE\_$TIMESTAMP\_$1
CALLGRIND_DIR=$OUTPUT_DIR/callgrind
INPUT_DIR=$ROOT/default_data

#GPROF2DOT=gprof2dot.py

# creates folder structure
mkdir $CALLGRIND_DIR -p

# copy input data
echo " --- Copying data"
cp -fv $INPUT_DIR/{foz.geo,foz.msh} $OUTPUT_DIR/
cp -fv $INPUT_DIR/foz.xml $ROOT


echo
echo " --- generating velocity.xml"
$ROOT/velocity
echo " --- generating polu.xml"
valgrind --tool=callgrind $ROOT/polu

mv foz.xml velocity.xml polution.xml $OUTPUT_DIR
mv -v callgrind.out.* $CALLGRIND_DIR
