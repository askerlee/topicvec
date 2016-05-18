set CORPUS=reuters-train-5770.orig.txt
set SUFFIX=reuters
perl gramcount.pl -i %CORPUS% -m1 --f1 top1grams-%SUFFIX%.txt -c --nofilter --thres1 5,0
perl gramcount.pl -i %CORPUS% -m2 --f1 top1grams-%SUFFIX%.txt --nofilter -c --f2 top2grams-%SUFFIX%.txt -w 3 --thres1 5,0
