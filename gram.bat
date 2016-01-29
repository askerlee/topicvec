set CORPUS=rcv1clean.txt
set SUFFIX=rcv1
perl gramcount.pl -i %CORPUS% -m1 --f1 top1grams-%SUFFIX%.txt -c --nofilter --thres1 50,0
perl gramcount.pl -i %CORPUS% -m2 --f1 top1grams-%SUFFIX%.txt --nofilter -c --f2 top2grams-%SUFFIX%.txt -w 3 --thres1 50,0
