#!/bin/bash
#
#	SeARCH Job Script
#
#PBS -N mpi.polu.time1
#PBS -l nodes=2:r101:ppn=4
#PBS -l walltime=1:00:00
#PBS -M pdrcosta90@gmail.com
#PBS -m bea
#PBS -o data/out/mpi.polu.time1.out
#PBS -e data/out/mpi.polu.time1.err
#PBS -V
#
cd "$PBS_O_WORKDIR";
PROCESSES=`cat $PBS_NODEFILE | wc -l`
mpirun -np $PROCESSES -machinefile "$PBS_NODEFILE" -loadbalance bin/mpi.polu.time1 data/xml/new.huge.param.xml
