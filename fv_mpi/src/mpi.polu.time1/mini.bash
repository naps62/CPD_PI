#!/bin/bash
#
#	SeARCH Job Script
#
#PBS -l nodes=6:r101:ppn=4
#PBS -l walltime=30:00
#PBS -M pdrcosta90@gmail.com
#PBS -m bea
#PBS -V
#
cd "$PBS_O_WORKDIR";
PROCESSES=`cat $PBS_NODEFILE | wc -l`
mpirun -np $PROCESSES -machinefile "$PBS_NODEFILE" -loadbalance bin/mpi.polu.time1 data/xml/new.huge.param.xml
