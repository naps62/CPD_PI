#!/bin/bash
#	Perform the tests for the Struct-Of-Arrays version

declare -a CASES;
declare -a MEASURES;

CASES=( "huge" );
EXEC="polu.soa"
MEASURES=( "btm" "l1dcm" "l2dcm" "brins" "fpins" "ldins" "srins" "totins" "vecins" "flops" );
RUNS=10

for CASE in "${CASES[@]}";
do
	for MEASURE in "${MEASURES[@]}";
	do
		for (( i = 0 ; i < $RUNS ; ++i ));
		do
			if [ ! -d "data" ];
			then
				mkdir "data";
			fi;
			if [ ! -d "data/out" ];
			then
				mkdir "data/out";
			fi;
			OUTFILE="data/out/${EXEC}_${CASE}_${MEASURE}.csv"
			if [ -f "$OUTFILE" ];
			then
				bin/${EXEC}.${MEASURE} data/xml/${CASE}.param.xml >> "$OUTFILE";
			else
				bin/${EXEC}.${MEASURE} data/xml/${CASE}.param.xml > "$OUTFILE";
			fi;
		done;
	done;
done;
