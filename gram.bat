set CORPUS=cleanwiki.txt
set SUFFIX=wiki
perl gramcount.pl -i %CORPUS% -m1 --f1 top1grams-%SUFFIX%.txt -c --nofilter
perl gramcount.pl -i %CORPUS% -m2 --f1 top1grams-%SUFFIX%.txt --nofilter -c --f2 top2grams-%SUFFIX%.txt -w 3
