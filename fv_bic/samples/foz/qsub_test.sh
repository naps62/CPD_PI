#!/bin/bash
#
#PBS -V
#PBS -l nodes=2:ppn=1
#PBS -l walltime=1:00:00
#PBS -N mpi.polu.test
#PBS -m bea
#PBS -e qsub_test.err
#PBS -o qsub_test.out

cd $PBS_O_WORKDIR

rm -rf polution.xml polution.seq.xml
gmsh foz.geo -2
mpi.fvcm foz.msh foz.xml
mpi.velocity

mpi.sequential
mv polution.xml polution.seq.xml
mpirun -loadbalance -n 2 -machinefile $PBS_NODEFILE mpi.polu
