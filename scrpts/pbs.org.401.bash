#!/bin/bash
#
#	SeARCH Job Script
#	Runs OpenMP on the 8 core with 8 GB of RAM Intel Xeon E5520 (group 401)
#
#PBS -l nodes=1:r401:ppn=8
#PBS -l walltime=5:00:00
#
#PBS -M pdrcosta90@gmail.com
#PBS -m bea
#PBS -e out/org.401.err
#PBS -o out/org.401.out
#
CASES=( "tiny" "small" "medium" "big" "huge" )
TIMERS=( "main" "iteration" "functions" )
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
