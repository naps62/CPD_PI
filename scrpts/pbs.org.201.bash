#!/bin/bash
#
#	SeARCH Job Script
#	Runs OpenMP on the 4 core with 4GB RAM Intel Xeon 5130 (group 201)
#
#PBS -l nodes=1:r201:ppn=4
#PBS -l walltime=5:00:00
#
#PBS -M pdrcosta90@gmail.com
#PBS -m bea
#PBS -e out/org.201-3.err
#PBS -o out/org.201-3.out
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
		echo ">>>>> ${t}";
		bin/${EXE}.${t} "data/xml/${c}.param.xml";
		echo "<<<<< ${t}";
	done;
	echo;
done;
