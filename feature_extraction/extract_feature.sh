#!/bin/sh
for i in $(ls CASP13_aln);do
	name=`echo $i|cut -d '_' -f 1`
        perl deepcon-seq-step1_hh.pl ./CASP13_aln/$i ${name}_feature
done

