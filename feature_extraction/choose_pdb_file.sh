#!/bin/sh

for i in $(ls *aln_feature);do
	name=`echo $i|cut -d '.' -f 1`
#	egrep -v "^>" ../../CASP13MyDM-Features/$i/${name}_uce3/${name}.a3m | sed 's/[a-z]//g' >./CASP13_aln/${name}.aln
	if [ -s ../../CASP13MyDM-Features/${name}.pdb ];then
		cp ../../CASP13MyDM-Features/${name}.pdb ./
		echo $name >>pdb_list_9_25.txt
	fi
done
