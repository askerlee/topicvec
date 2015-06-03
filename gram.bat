set CORPUS=D:\omer\cleanwiki.txt.clean
set SUFFIX=wiki-clean
perl gramcount.pl -i %CORPUS% -m1 --f1 top1grams-%SUFFIX%.txt -c --nofilter
perl gramcount.pl -i %CORPUS% -m2 --f1 top1grams-%SUFFIX%.txt --nofilter -c --f2 top2grams-%SUFFIX%.txt --top1 28000 -e absentwords.txt -w 5 --dyn
