#!/bin/sh
N=44
n=1
rm result_hh.txt
echo num pdb_id tmscore rmsd >>result_hh.txt
for i in $(seq 150);do
#	N=$((N+44))
#	n=$((n+44))
	name=$(awk 'NR=="'"$n"'"{print $1}' nohup.out)
	s_tmscore=$(awk 'NR=="'"$N"'"{print $1}' nohup.out)
	s_rmsd=$(awk 'NR=="'"$N"'"{print $2}' nohup.out)
	echo $i $name $s_tmscore $s_rmsd >>result_hh.txt
        n=$((n+44))
        N=$((N+44))
#	echo -e "\n" >>result.txt
done

