#!/bin/bash

set -e

EVAL="./eval-using-tmscore.pl"

maxjobs=0
for pdb in `ls ./pdb/*.pdb`;
   do
	id=$(basename $pdb)
	id=${id%.*}
	echo $id
    if [ ! -d "./distfold-jobs_hh/output-$id" ];then
	continue
    fi
#    files=$(ls ./distfold-jobs/output-$id/*.pdb 2>/dev/null |wc -l)
#    if ls ./distfold-jobs/output-$id/*.pdb >/dev/null 2 >&1;then
#	continue
#    fi
    perl $EVAL $pdb ./distfold-jobs_hh/output-$id all header
    maxjobs=$((maxjobs+1))
#    if [[ "$maxjobs" -gt 10 ]]; then
 #       exit 0
 #   fi
   done
