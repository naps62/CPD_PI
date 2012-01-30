#!/bin/bash
#
#	SeARCH Job Script
#	Runs OpenMP on the 24 core with 64GB RAM AMD 6174 (511 group)
#	Runs OpenMP on the 2 core with 2GB RAM Xeon 3.2GHz (101 group)
#
#PBS -l nodes=1:r101
#PBS -l walltime=1:00:00
#
#PBS -M pdrcosta@gmail.com
#PBS -m bea
#PBS -e out/openmp.24.out
#PBS -o out/openmp.24.err
#
CASES=( tiny small medium huge original )
EXE=polu.openmp

cd "$PBS_O_WORKDIR"

for c in $CASES[@];
do
	echo "#####    ${c}    #####";
	bin/${EXE} "data/xml/${c}.params.xml";
	echo;
done;
