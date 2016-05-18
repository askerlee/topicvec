import sys

CAT = 1
WORDS = 2

FTAB = open(sys.argv[1])
FLIST = open(sys.argv[2], "w")
FLIST.write("WORD\tTRUECLASS\n")

state = CAT
catnum = 0
wordnum = 0

for line in FTAB:
    line = line.strip()

    if not line and state == CAT:
        continue

    if state == CAT:
        cat = line.replace(" ", "-")
        state = WORDS
        catnum += 1
        continue
    if state == WORDS:
        if line:
            words = line.split(", ")
            for word in words:
                word = word.replace(",", "")
                FLIST.write( "%s\t%s\n" %(word, cat) )
                wordnum += 1
            continue
        else:
            state = CAT
            continue

print "%d words in %d categories written into %s" %(wordnum, catnum, sys.argv[2])
