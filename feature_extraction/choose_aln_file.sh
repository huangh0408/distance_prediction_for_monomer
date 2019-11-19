#!/bin/sh

for i in $(ls ../../CASP13MyDM-Features);do
	name=`echo $i|cut -d '_' -f 1`
	egrep -v "^>" ../../CASP13MyDM-Features/$i/${name}_uce3/${name}.a3m | sed 's/[a-z]//g' >./CASP13_aln/${name}.aln
done
