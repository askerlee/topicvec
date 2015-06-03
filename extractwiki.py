#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code based on http://textminingonline.com/training-word2vec-model-on-english-wikipedia-by-gensim
 
import logging
import os.path
import sys
 
from gensim.corpora import WikiCorpus
 
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
 
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
 
    # check and process input arguments
    if len(sys.argv) < 3:
        print "Usage: extractwiki.py infile_name outfile_name"
        sys.exit(1)
        
    infilename, outfilename = sys.argv[1:3]
 
    if os.path.isfile(outfilename):
        logger.error("Output file %s exists. Change the file name and try again." %outfilename)
        sys.exit(1)
        
    i = 0
    output = open(outfilename, 'w')
    wiki = WikiCorpus(infilename, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        output.write( " ".join(text) + "\n")
        i = i + 1
        if (i % 10000 == 0):
            logger.info("Saved " + str(i) + " articles")
 
    output.close()
    logger.info("Finished Saved " + str(i) + " articles")
    