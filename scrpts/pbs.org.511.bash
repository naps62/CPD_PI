#!/bin/bash
#
#	SeARCH Job Script
#	Runs OpenMP on the 24 core with 64GB RAM AMD Opt 6174 (group 511)
#
#PBS -l nodes=1:r511:ppn=24
#PBS -l walltime=3:00:00
#
#PBS -M pdrcosta90@gmail.com
#PBS -m bea
#PBS -e out/org.511-2.err
#PBS -o out/org.511-2.out
#
#CASES=( "tiny" "small" "medium" "big" "huge" )
CASES=( "huge" );
#TIMERS=( "main" "iteration" "functions" )
TIMERS=( "functions" );
EXE="polu.orig.time"

cd "$PBS_O_WORKDIR"

for c in ${CASES[@]};
do
	echo "#####    ${c}    #####";
	for t in ${TIMERS[@]}
	do
		echo ">>>>>    ${t}";
		bin/${EXE}.${t} "data/xml/${c}.param.xml";
		echo "<<<<<    ${t}";
	done;
	echo;
done;
