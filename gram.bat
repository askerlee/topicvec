perl gramcount.pl -i cleanwiki.txt -m1 --f1 top1grams-wiki.txt
@rem perl gramcount.pl -i cleanwiki.txt -m2 --f1 top1grams-wiki.txt --f2 top2grams-wiki.txt
perl gramcount.pl -i cleanwiki.txt -m2 --f1 top1grams-wiki.txt --nofilter -c --f2 top2grams-wiki.txt --top1 40000 -e absentwords.txt -w 4 --dyn
