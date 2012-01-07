all:
	dirs=`find . -type d -depth 1`;	\
	for dir in $${dirs};	\
		do cd $${dir}; $_; cd ..;	\
	done

%:
	dirs=`find . -type d -depth 1`;	\
	for dir in $${dirs};	\
		do cd $${dir}; $_ $@; cd ..;	\
	done
