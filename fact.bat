@echo off
rem Train old embeddings: 
rem python factorize.py -n 500 -t 28000 -e absentwords.txt top2grams-wiki.txt
rem Train new embeddings:
python factorize.py -n 500 -b 20000 -v 100000 top2grams-wiki.txt
