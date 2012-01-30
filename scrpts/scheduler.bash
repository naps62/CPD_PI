#!/bin/bash
####    SCHEDULER    ####
#	calls the sampler to test each provided case
#
#	-f	FILE		test cases file
#	-o	OUTPUT		destiny file for the gathered samples	[SAMPLER]
#	-nr	RUNS		minimal number of runs					[TESTER]
#	-ns	SAMPLES		number of samples to gather				[SAMPLER]
#	-t	TRUST		trust degree							[TESTER]
#

usage () {
	echo "$0 [-o <OUTPUT>] [-nr <#RUNS>] [-ns <#SAMPLES>] [-t <TRUST>] <FILE>"
	echo -e "\tFILE:\tfile containing the test cases";
	echo -e "\tOUTPUT:\tsamples output file name (see sampler)";
	echo -e "\tRUNS:\tminimal number of runs (see tester)";
	echo -e "\tSAMPLES:\tnumber of samples to gather (see sampler)";
	echo -e "\tTRUST:\ttrust degree (percent, integer value) (see tester)";
	echo;
}

SAMPLER="sampler.bash"

declare -a CASES

#	parse arguments
i=0
o_flg=false;
nr_flg=false
ns_flg=false;
t_flg=false;
for arg;
do
	case $arg in
	"-o")
		o_flg=true;
		;;
	"-nr")
		nr_flg=true;
		;;
	"-ns")
		ns_flg=true;
		;;
	"-t")
		t_flg=true;
		;;
	*)
		if $o_flg;
		then
			OUTPUT="$arg";
			o_flg=false;
		elif $nr_flg;
		then
			RUNS="$arg";
			nr_flg=false;
		elif $ns_flg;
		then
			SAMPLES="$arg";
			ns_flg=false;
		elif $t_flg;
		then
			TRUST="$arg";
			t_flg=false;
		else
			FILE="$arg";
		fi;
		;;
	esac;
done;

exec <${FILE};
i=0;
while read line;
do
	larray=( $line );
	echo "$i";
	echo "${larray[@]}";
	i=$(( $i + 1 ));
done;
