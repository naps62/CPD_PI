#!/bin/bash
#
#	SeARCH Job Script
#	Runs sequential code on the 8 core with 8GB RAM Xeon E5420 (group 311)
#
#PBS -l nodes=1:r311:ppn=1
#PBS -l walltime=5:00:00
#
#PBS -M pdrcosta90@gmail.com
#PBS -m bea
#PBS -e out/seq.311.err
#PBS -o out/seq.311.out
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
