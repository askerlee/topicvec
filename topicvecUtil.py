import numpy as np
from utils import *
import pdb

global fileLogger
# should initialize fileLogger before main()
fileLogger = None

def calcTopicResiduals(T, V, u):
    VT = np.dot(V, T.T)
    
    try:
        expVT = np.exp(VT)
    except FloatingPointError:
        pdb.set_trace()
        
    r = -np.log( np.dot(u, expVT) )

    fileLogger.debug("r:")
    fileLogger.debug(r)

    return r

# Pi: L x K
# sum_pi_v: K x N0
def calcSum_pi_v(Pi, V, wids):
    L, K = Pi.shape
    N0 = V.shape[1]
    sum_pi_v = np.zeros( (K, N0) )
    
    for i in xrange(L):
        sum_pi_v += np.outer( Pi[i], V[ wids[i] ] )

    return sum_pi_v

def screen_log_output(s):
    fileLogger.debug(s)
    print s
    
# topicRatioThres: when a topic's proportion Em[k]/L > topicRatioThres/K, print it
# default: the average topic proportion
def printTopWordsInTopic( wids, vocab, Pi, V, T, wid2freq, outputToScreen, topD, topicRatioThres=1.0 ):
    wids2 = wid2freq.keys()
    wids_topics_sim = np.dot( normalizeF( V[wids2] ), normalizeF(T).T )
    wids_topics_dot = np.dot( V[wids2], T.T )
    
    wid2rowID = {}
    for i, wid in enumerate(wids2):
        wid2rowID[wid] = i
    
    row_topicsProp = np.zeros( wids_topics_sim.shape )
        
    L, K = Pi.shape
            
    Em = np.sum( Pi, axis=0 )
    tids = sorted( range(K), key=lambda k: Em[k], reverse=True )
    for i,k in enumerate(tids):
        if Em[k] < topicRatioThres * L / K:
            break
    
    # cut_i is the cut point of tids: tids[:cut_i] will be printed
    # if i==0, no topic has enough proportion to be printed. 
    # this may happen when topitThres is too big. in this case, print the principal topic
    if i == 0:
        cut_i = 1
    else:
        cut_i = i

    for i in xrange(L):
        wid = wids[i]
        rowID = wid2rowID[wid]
        row_topicsProp[rowID] += Pi[i]
    
    W = len(vocab) 
    # number of unique words in the doc       
    W2 = len(wids2)

    if outputToScreen:
        out = screen_log_output
    else:
        out = fileLogger.debug
        
    out("")
    out("Topic magnitudes:")
    
    topicMagnitudes = np.linalg.norm(T, axis=1)

    out(topicMagnitudes)
    out("")
    
    selTids = tids[:cut_i]
    # always output topic 0
    if len( np.where(selTids == 0)[0] ) == 0:
        selTids = np.append( selTids, 0 )
        
    for k in selTids:
        out( "Topic %d (%.2f): %.1f%%" %( k, np.linalg.norm( T[k] ), 100 * Em[k] / L ) )
            
        rowID_sorted = sorted( range(W2), key=lambda rowID: row_topicsProp[rowID, k], reverse=True )
        
        out("Most relevant words in doc:")
        
        for rowID in rowID_sorted[:topD]:
            wid = wids2[rowID]
            topicProp = row_topicsProp[rowID, k]
            sim = wids_topics_sim[rowID, k]
            dotprod = wids_topics_dot[rowID, k]
            
            # multiple words in a line for screen output, so do not use out() here
            fileLogger.debug( "%s (%d,%d): %.3f/%.3f/%.3f" %( vocab[wid], wid, wid2freq[wid], topicProp, sim, dotprod ) )
            if outputToScreen:
                print "%s (%d,%d): %.3f/%.3f/%.3f" %( vocab[wid], wid, wid2freq[wid], topicProp, sim, dotprod ),
        
        if np.linalg.norm( T[k] ) == 0:
            continue
            
        V_topic_dot = np.dot( V, T[k] )
        V_topic_sim = V_topic_dot / np.linalg.norm( V, axis=1 ) / np.linalg.norm( T[k] )
        
        wid_sorted = sorted( range(W), key=lambda wid: V_topic_sim[wid], reverse=True )
            
        if outputToScreen:
            print
        out("Most similar words in vocab:")
        
        for wid in wid_sorted[:topD]:
            sim = V_topic_sim[wid]
            dotprod = V_topic_dot[wid]
            
            # multiple words in a line for screen output, so do not use out() here
            fileLogger.debug( "%s: %.3f/%.3f" %( vocab[wid], sim, dotprod ) )
            if outputToScreen:
                print "%s (%d): %.3f/%.3f" %( vocab[wid], wid, sim, dotprod ),
            
        if outputToScreen:
            print
        out("")

# wid2Pi: a dictionary that maps a wid to its topic distribution   
# N: number of sentences to extract
def getSummary(widInSentences, wid2Pi, N):
    