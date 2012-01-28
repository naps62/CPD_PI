####    SAMPLER    ####
#	calls each test M times saving each set of samples in the same file
#
#	-s	SAMPLES		number of samples to gather
#	-o	OUTPUT		destiny file for the gathered samples
#
#	-n	RUNS		minimal number of runs (tester)
#	-t	TRUST		trust degree (tester)

usage () {
	echo "$0 [-o <OUTPUT>] [-nr <#RUNS>] [-ns <#SAMPLES>] [-t <TRUST>] command [arg0 ...]";
	echo "where"
	echo "\tOUTPUT:\tsamples output file name";
	echo "\tRUNS:\tminimal number of runs (see tester)";
	echo "\tSAMPLES:\tnumber of samples to gather";
	echo "\tTRUST:\ttrust degree (percent, integer value)";
	echo;
}

TESTER="tester.bash";
RESULT_D="../results";

#	parse arguments
i=0;
o_flg=false;
nr_flg=false;
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
			command[$i]="$arg";
			i=$(( $i + 1 ));
		fi;
		;;
	esac;
done;

#	there must be a command to test and sample
if [ "${#command[@]}" -lt 1 ];
then
	usage;
	exit 1;
fi;

#	script shrink, solves identity problems
#	script: "who am I?"
SOURCE="${BASH_SOURCE[0]}";
while [ -h "$SOURCE" ];
do
	SOURCE=`readlink "$SOURCE"`;
done;
source_d=`dirname "$SOURCE"`;
SOURCE_D=`cd -P "${source_d}" && pwd`;

#	tester command
t_command[0]="${SOURCE_D}/${TESTER}";
i=1;
if [ ! -z "$RUNS" ];
then
	t_command[$i]="-n ${RUNS}";
	i=$(( $i + 1 ));
fi;
if [ ! -z "$TRUST" ];
then
	t_command[$i]="-t ${TRUST}";
	i=$(( $i + 1 ));
fi;
t_command[$i]="${command[@]}";

#	number of samples
if [ -z "$SAMPLES" ];
then
	SAMPLES=10;
fi;

#	output file
if [ -z "$OUTPUT" ];
then
	if [ -z "$RESULT_D" ];
	then
		RESULT_D=".";
	fi;
	OUTPUT="${SOURCE_D}/${RESULT_D}/${command[0]}.samples";
fi;

#	prepare output file
if [ -f "$OUTPUT" ];
then
	echo >> "$OUTPUT";
else
	touch "$OUTPUT";
fi;
	
timestamp=`date "+%y-%m-%d_%H:%M:%S"`;
echo "##### ##### ##### #####" >> "$OUTPUT";
echo "Sampler job initiated at ${timestamp}" >> "$OUTPUT";
echo >> "$OUTPUT";

i=1;
while [ "$i" -lt "$SAMPLES" ];
do
	echo "SAMPLE #${i} ----- -----" >> "$OUTPUT";
	${SHELL} ${t_command[@]} >> "$OUTPUT";
	echo "----- ----- ----- -----" >> "$OUTPUT";
	echo >> "$OUTPUT";
	i=$(( $i + 1 ));
done;
timestamp=`date "+%y-%m-%d_%H:%M:%S"`;
echo "Sampler job finished at ${timestamp}" >> "$OUTPUT";
echo "##### ##### ##### #####" >> "$OUTPUT";
echo >> "$OUTPUT";
