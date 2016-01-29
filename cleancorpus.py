import sys
import os
import gensim.corpora.wikicorpus

# check and process input arguments
if len(sys.argv) < 3:
    print "Usage: cleancorpus.py infile_name outfile_name"
    sys.exit(1)
    
infilename, outfilename = sys.argv[1:3]

if os.path.isfile(outfilename):
    print "Output file %s exists. Change the file name and try again." %outfilename
    sys.exit(1)
    
linecount = 0
bytecount = 0
wordcount = 0

output = open(outfilename, 'w')     
IN = open(infilename)
for line in IN:
    tokens = gensim.corpora.wikicorpus.tokenize(line)
    output.write( "%s\n" %(" ".join(tokens)) )
    linecount += 1
    bytecount += len(line)
    wordcount += len(tokens)
    if linecount % 500 == 0:
        print "\r%d    %d    %d    \r" %(linecount, bytecount/1024/1024, wordcount),
        