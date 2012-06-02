#!/bin/bash
#
#	SeARCH Job Script
#
#PBS -l walltime=0:10:00
#PBS -M pdrcosta90@gmail.com
#PBS -m bea
#PBS -V
#
cd "$PBS_O_WORKDIR";
PROCESSES=`cat $PBS_NODEFILE | wc -l`
mpirun -np $PROCESSES -machinefile "$PBS_NODEFILE" -loadbalance bin/mpi.polu.time0 data/xml/new.huge.param.xml
