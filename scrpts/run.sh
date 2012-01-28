#!/bin/sh

resultdir="results"
bindir="bin"

cd "${bindir}";

for arg in $@;
do
	case $arg in
	polu*)
		if [ ! -d "$resultdir" ];
		then
			mkdir "$resultdir";
		fi;
		timestamp=`date "+%y-%m-%d_%H:%M:%S"`
		./$arg "../data/xml/param.xml";
		mv "../data/xml/polution.xml" "../${resultdir}/${arg}.out.${timestamp}"
		;;
	*)
		./$arg
		;;
	esac
done
