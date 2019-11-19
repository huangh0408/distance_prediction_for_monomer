#!/bin/bash

#DISTFOLD="/home/huanghe/huangh/IEEE-ICMLA-2019-PIDP-Challenge/workspace_8_6/DISTFOLD/distfold.pl"
DISTFOLD="./DISTFOLD/distfold.pl"

maxjobs=0
for rr in `ls ./predictions_hh/*.rr`;
   do
	id=$(basename $rr)
	id=${id%.*}
	echo $id
#    rm -r ./distfold-jobs/output-$id
    # Run 150 jobs in parallel
	perl $DISTFOLD -rr ./predictions_hh/$id.rr -ss ./ss/$id.ss -o ./distfold-jobs_hh/output-$id -mcount 20 -selectrr 1.0L &> ./distfold-jobs_hh/$id.log &
    maxjobs=$((maxjobs+1))
    #if [[ "$maxjobs" -gt 150 ]]; then
     #   exit 0
   # fi
   done
