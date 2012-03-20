#!/bin/bash
#	Perform the tests for the Struct-Of-Arrays version

declare -a CASES;
declare -a MEASURES;

CASES=( "huge" );
EXEC="polu.soa"
#MEASURES=( "btm" "l1dcm" "l2dcm" "brins" "fpins" "ldins" "srins" "totins" "vecins" "flops" );
MEASURES=( "fpins" "vecins" "flops" );
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
			if [ -f "data/out/${EXEC}.${CASE}.${MEASURE}" ];
			then
				bin/${EXEC}.${MEASURE} data/xml/${CASE}.param.xml >> data/out/${EXEC}.${CASE}.${MEASURE};
			else
				bin/${EXEC}.${MEASURE} data/xml/${CASE}.param.xml > data/out/${EXEC}.${CASE}.${MEASURE};
			fi;
		done;
	done;
done;
