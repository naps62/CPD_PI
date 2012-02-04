#!/bin/bash
#
#	SeARCH Job Script
#	Runs OpenMP on the 4 core with 4GB RAM Intel Xeon 5130 (group 201)
#	(Halved)
#
#PBS -l nodes=1:r201:ppn=2
#PBS -l walltime=10:00:00
#
#PBS -M pdrcosta90@gmail.com
#PBS -m bea
#PBS -e out/omp.201.2.err
#PBS -o out/omp.201.2.out
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
