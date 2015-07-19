@echo off
rem Old way of exact factorization: 
rem python factorize.py -n 500 -t 28000 -e absentwords.txt top2grams-wiki.txt
rem New online fashion:
rem 1. obtain 60000 embeddings (20000 core + 40000 noncore)
python factorize.py -b 20000 -o 40000 top2grams-wiki.txt
rem 2. incrementally learn other 60000 noncore embeddings
python factorize.py -v 60000-500-BLKEM.vec -o 60000 top2grams-wiki.txt
