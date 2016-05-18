@echo off
set N0=50
rem Old way of exact factorization: 
rem python factorize.py -n 50 -t 28000 -e absentwords.txt top2grams-rcv1.txt
rem New online fashion:
rem 1. Obtain 23000 core embeddings, into 25000-50-EM.vec:
rem python factorize.py -w 23000 -n %N0% top2grams-rcv1.txt
rem 2. Obtain 23409 noncore embeddings, totaling 46409 (23000 core + 23409 noncore), into 25000-46409-50-BLK-2.0.vec:
python factorize.py -v 23000-%N0%-EM.vec  -n %N0% -t2 top2grams-rcv1.txt
