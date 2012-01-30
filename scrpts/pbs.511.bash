#!/bin/bash
#
#	SeARCH Job Script
#	Runs OpenMP on the 24 core with 64GB RAM AMD Opt 6174 (group 511)
#
#PBS -l nodes=1:r511
#PBS -l walltime=1:00:00
#
#PBS -M pdrcosta90@gmail.com
#PBS -m bea
#PBS -e out/openmp.511.err
#PBS -o out/openmp.511.out
#
CASES=( tiny small medium huge original )
EXE="polu.openmp"

cd $PBS_O_WORKDIR

for c in ${CASES[@]};
do
	echo "#####    ${c}    #####";
	bin/${EXE} "data/xml/${c}.params.xml";
	echo;
done;
