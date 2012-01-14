#!/bin/sh

timestamp=`date "+%m-%d_%H:%M:%S"`

resultdir="../results"

cd "bin";

for arg in $@
do
	case $arg in
	polu*)
		./$arg "../data/xml/param.xml";
		if [ ! -d "$resultdir" ];
		then
			mkdir "$resultdir";
		fi;
		mv "../data/xml/polution.xml" "${resultdir}/${arg}.out.${timestamp}"
		;;
	*)
		./$arg
		;;
	esac
done
