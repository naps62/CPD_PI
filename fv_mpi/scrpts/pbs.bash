#!/bin/bash
#		to use with mpich2
#PBS -l walltime=2:00:00
#PBS -M pdrcosta90@gmail.com
#PBS -m bea
#
source "/etc/profile.d/env-modules.sh";
module purge;
module load gnu/openmpi;

cd "$PBS_O_WORKDIR";

if [ ! "$RUNS" ]; then RUNS="1"; fi;

if [ ! "$PROCESSES" ]
then
	PROCESSES=`cat "$PBS_NODEFILE" | wc -l`;
fi;

#NODES=`sort "$PBS_NODEFILE" | uniq | tee "mpd.hosts" | wc -l`;
#mpdboot -n "$NODES" -f "mpd.hosts";

R="1";
while [ "$R" -le "$RUNS" ];
do
	echo -en "\t${R}..." 1>&2;
	mpirun -n "$PROCESSES" -f "$PBS_NODEFILE" -loadbalance "$EXEC" "data/xml/huge.param";
	echo "DONE" 1>&2;
	R=$(( $R + 1 ));
done;
	

#mpdallexit;
rm "mpd.hosts";
