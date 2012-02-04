#!/bin/bash
#
#	SeARCH Job Script
#	Runs OpenMP on the 24 core with 64GB RAM AMD Opt 6174 (group 511)
#	(Quarted)
#
#PBS -l nodes=1:r511:ppn=6
#PBS -l walltime=10:00:00
#
#PBS -M pdrcosta90@gmail.com
#PBS -m bea
#PBS -e out/omp.511.6.err
#PBS -o out/omp.511.6.out
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
