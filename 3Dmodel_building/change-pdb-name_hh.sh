#!/bin/sh

for i in $(ls ./distfold-jobs_hh);do
	rename 's/_sub_embed/.dist/' ./distfold-jobs_hh/$i/*
done

