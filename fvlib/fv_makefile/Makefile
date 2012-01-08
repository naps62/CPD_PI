

.PHONY: FVLib BIC2012

default: FVLib BIC2012 symlinks

FVLib BIC2012:
	cd $@; $_

#all:
#	dirs=`find . -mindepth 1 -maxdepth 1 -type d`;	\
#	for dir in $${dirs};	\
#		do cd $${dir}; $_; cd ..;	\
#	done

#%:
#	dirs=`find . -mindepth 1 -maxdepth 1 -type d`;	\
#	for dir in $${dirs};	\
#		do cd $${dir}; $_ $@; cd ..;	\
#	done

clean:
	cd FVLib; make clean
	cd BIC2012; make clean

symlink:
	ln -s FVLib/bin/fvcd ../simulations/fvcd
	ln -s FVLib/bin/fvcm ../simulations/fvcm
	ln -s BIC2012/polu ../simulations/polu
	ln -s BIC2012/velocity ../simulations/velocity
