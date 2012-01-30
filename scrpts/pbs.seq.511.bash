#!/bin/bash
#
#	SeARCH Job Script
#	Runs sequential code on the 24 core with 64GB RAM AMD Opt 6174 (group 311)
#
#PBS -l nodes=1:r511
#PBS -l walltime=3:00:00
#
#PBS -M pdrcosta90@gmail.com
#PBS -m bea
#PBS -e out/seq.511.err
#PBS -o out/seq.511.out
#
CASES=( tiny small medium big huge original )
EXE="polu.clean"

cd "$PBS_O_WORKDIR"

for c in ${CASES[@]};
do
	echo "#####    ${c}    #####";
	time bin/${EXE} "data/xml/${c}.param.xml";
	echo;
done;
