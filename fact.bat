@echo off
rem Old way of exact factorization: 
rem python factorize.py -n 500 -t 28000 -e absentwords.txt top2grams-wiki.txt
rem New online fashion:
rem 1. obtain 70000 embeddings (25000 core + 45000 noncore), into 25000-70000-500-BLKEM.vec
python factorize.py -b 25000 -o 45000 top2grams-wiki.txt
rem 2. incrementally learn other 60000 noncore embeddings, into 25000-130000-500-BLKEM.vec
python factorize.py -v 25000-70000-500-BLKEM.vec -b 25000 -o 60000 top2grams-wiki.txt
rem 3. run more times to get embeddings of the remaining rare words
