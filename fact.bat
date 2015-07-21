@echo off
rem Old way of exact factorization: 
rem python factorize.py -n 500 -t 28000 -e absentwords.txt top2grams-wiki.txt
rem New online fashion:
rem 1. Obtain 25000 core embeddings, into 25000-500-EM.vec:
python factorize.py -w 25000 top2grams-wiki.txt
rem 2. Obtain 45000 noncore embeddings, totaling 70000 (25000 core + 45000 noncore), into 25000-70000-500-BLKEM.vec:
python factorize.py -v 25000-500-EM.vec -o 45000 top2grams-wiki.txt
rem 3. Incrementally learn other 50000 noncore embeddings (based on 25000 core), into 25000-120000-500-BLKEM.vec:
python factorize.py -v 25000-70000-500-BLKEM.vec -b 25000 -o 50000 top2grams-wiki.txt
rem 4. Repeat 3 a few times to get more embeddings of rarer words.
