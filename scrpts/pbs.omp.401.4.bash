#!/bin/bash
#
#	SeARCH Job Script
#	Runs OpenMP on the 8 core with 8 GB of RAM Intel Xeon E5520 (group 401)
#	(Halfed)
#
#PBS -l nodes=1:r401:ppn=4
#PBS -l walltime=10:00:00
#
#PBS -M pdrcosta90@gmail.com
#PBS -m bea
#PBS -e out/omp.401.4.err
#PBS -o out/omp.401.4.out
#
CASES=( "tiny" "small" "medium" "big" "huge" "original" )
TIMERS=( "main" "iteration" "functions" )
EXE="polu.omp.time"

cd "$PBS_O_WORKDIR"

for c in ${CASES[@]};
do
	echo "#####    ${c}    #####";
	for t in ${TIMERS[@]}
	do
		echo ">>>>> ${t}";
		for i in {1..10}
		do
			bin/${EXE}.${t} "data/xml/${c}.param.xml";
		done;
		echo "<<<<< ${t}";
	done;
	echo;
done;
