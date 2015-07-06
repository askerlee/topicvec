# -*- coding: utf-8 -*-

### change to "\Lib\site-packages\gensim\corpora\wikicorpus.py" ###

    def get_texts(self):
        ....
        # changed to support both compressed and uncompressed formats
        if self.fname[-4:].lower() == ".bz2":
            infile = bz2.BZ2File(self.fname)
        else:
            infile = open(self.fname)
        texts = ((text, self.lemmatize, title, pageid) for title, text, pageid in extract_pages(infile, self.filter_namespaces))
        
# add "keep_poss=True"
def tokenize(content):
    return [token.encode('utf8') for token in utils.tokenize(content, lower=True, errors='ignore', keep_poss=True)
            if 2 <= len(token) <= 15 and not token.startswith('_')]

def get_texts(self):
    # change the maxsize=1 to maxsize=10, to increase parallalism
    for group in utils.chunkize(texts, chunksize=10 * self.processes, maxsize=10):
        

### change to "\Lib\site-packages\gensim\utils.py" ###

# add Possessive 's. In order to match "â€? \xE2\x80\x99, do
# text = text.replace("â€?, "'") before matching
PAT_ALPHABETIC_POSS = re.compile('(((?![\d])\w)+(?:\'s)?)', re.UNICODE)

# add "keep_poss=False", the new switch of whether extracts possessive forms
def tokenize(text, lowercase=False, deacc=False, errors="strict", to_lower=False, lower=False, keep_poss=False):
    
    if keep_poss:
        # utf8 of "¡¯"
        text = text.replace(u"â€?, "'")            # '
        PAT = PAT_ALPHABETIC_POSS
    else:
        PAT = PAT_ALPHABETIC
        
    ...
        
    for match in PAT.finditer(text):
        ....
        
    