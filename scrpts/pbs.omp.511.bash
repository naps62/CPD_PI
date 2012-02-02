#!/bin/bash
#
#	SeARCH Job Script
#	Runs OpenMP on the 24 core with 64GB RAM AMD Opt 6174 (group 511)
#
#PBS -l nodes=1:r511:ppn=24
#PBS -l walltime=5:00:00
#
#PBS -M pdrcosta90@gmail.com
#PBS -m bea
#PBS -e out/omp.511.err
#PBS -o out/omp.511.out
#
CASES=( tiny small medium big huge original )
EXE="polu.openmp"

cd "$PBS_O_WORKDIR"

for c in ${CASES[@]};
do
	echo "#####    ${c}    #####";
	time bin/${EXE} "data/xml/${c}.param.xml";
	echo;
done;