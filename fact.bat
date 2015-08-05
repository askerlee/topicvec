@echo off
rem Old way of exact factorization: 
rem python factorize.py -n 500 -t 28000 -e absentwords.txt top2grams-wiki.txt
rem New online fashion:
rem 1. Obtain 25000 core embeddings, into 25000-500-EM.vec:
python factorize.py -w 25000 top2grams-wiki.txt
rem 2. Obtain 55000 noncore embeddings, totaling 80000 (25000 core + 55000 noncore), into 25000-80000-500-BLK-2.0.vec:
python factorize.py -v 25000-500-EM.vec -o 55000 -t2 top2grams-wiki.txt
rem 3. Incrementally learn other 50000 noncore embeddings (based on 25000 core), into 25000-130000-500-BLK-4.0.vec:
python factorize.py -v 25000-80000-500-BLK-2.0.vec -b 25000 -o 50000 -t4 top2grams-wiki.txt
rem 4. Repeat 3 again to get more embeddings of rarer words.
python factorize.py -v 25000-130000-500-BLK-4.0.vec -b 25000 -o 50000 -t8 top2grams-wiki.txt
