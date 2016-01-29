#!/usr/bin/python
import os
import sys
import re

oldVecFilename = sys.argv[1]
newVecFilename = sys.argv[2]

stream = os.popen( "wc %s" %oldVecFilename )
output = stream.read()
output = output.strip()
linecount, wordcount, charcount, filename = re.split(" +", output)
linecount = int(linecount)
wordcount = int(wordcount)

if wordcount % linecount != 0:
    print "Error: line count %d does not divide word count %d" %(linecount, wordcount)
    sys.exit(1)

veclen = wordcount / linecount - 1
print "%d %d" %(linecount, veclen)
VEC = open(newVecFilename, "w")
VEC.write( "%d %d\n" %(linecount, veclen) )
VEC.close()
os.popen( "cat %s >> %s" %(oldVecFilename, newVecFilename) )

stream = os.popen( "ls -l %s" %oldVecFilename )
print stream.read().strip()
stream = os.popen( "ls -l %s" %newVecFilename )
print stream.read().strip()
