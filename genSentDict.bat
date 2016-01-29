@echo off
set N=200
python topwordsInList.py -c sent-gen-config.txt -l d:\Dropbox\sentiment\positive-words.txt,d:\Dropbox\sentiment\negative-words.txt -n %N% -o topSentWords%N%.txt
