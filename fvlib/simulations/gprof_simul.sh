#!/bin/bash

if [ $# -eq 0 ]; then
	echo <<EOF
	Usage:
		gprof_simul <folder>
		folder - folder name to be created inside simulations to generate input/output
		folder/gprof will also be created to save gprof results
EOF
	exit 1
fi

TIMESTAMP=$(date +%m.%d_%H:%M:%S)

ROOT=.
OUTPUT_DIR=$ROOT/$TIMESTAMP\_$1
GPROF_DIR=$OUTPUT_DIR/gprof
INPUT_DIR=$ROOT/default_data

GPROF2DOT=gprof2dot.py

# creates folder structure
mkdir $GPROF_DIR -p

# copy input data
echo " --- Copying data"
cp -fv $INPUT_DIR/{foz.geo,foz.msh} $OUTPUT_DIR/
cp -fv $INPUT_DIR/foz.xml $ROOT

echo
echo " --- generating velocity.xml"
$ROOT/velocity
echo " --- generating polu.xml"
$ROOT/polu

mv foz.xml velocity.xml polution.xml $OUTPUT_DIR

echo " --- profiling velocity"
# logsave $GPROF_DIR/velocity.gprof gprof velocity | $GPROF2DOT | dot -Tpng -o $GPROF_DIR/velocity.png
echo " --- profiling polu"
gprof velocity > $GPROF_DIR/velocity.gprof
cat $GPROF_DIR/velocity.gprof | $GPROF2DOT | dot -Tpng -o $GPROF_DIR/velocity.png
gprof polu > $GPROF_DIR/polu.gprof
cat $GPROF_DIR/polu.gprof | $GPROF2DOT | dot -Tpng -o $GPROF_DIR/polu.png
