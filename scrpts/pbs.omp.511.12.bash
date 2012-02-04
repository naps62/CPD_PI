#!/bin/bash
#
#	SeARCH Job Script
#	Runs OpenMP on the 24 core with 64GB RAM AMD Opt 6174 (group 511)
#	(Halfed)
#
#PBS -l nodes=1:r511:ppn=12
#PBS -l walltime=8:00:00
#
#PBS -M pdrcosta90@gmail.com
#PBS -m bea
#PBS -e out/omp.511.12.err
#PBS -o out/omp.511.12.out
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
