import numpy as np
import scipy.linalg
from scipy.special import *
import getopt
import sys
from utils import *
import pdb
import time
import re
import os
from scipy.spatial.distance import cdist

# V: W x N0
# T: K x N0
# VT: W x K
# u: W x 1
# r: K x 1
# Pi: L x K
# sum_pi_v: K x N0
# X = Evv

class topicvecDir:
    def __init__(self, **kwargs):
        self.unigramFilename = kwargs.get( 'unigramFilename', "top1grams-wiki.txt" )
        self.word_vec_file = kwargs.get( 'word_vec_file', "25000-500-EM.vec" )
        self.topic_vec_file = kwargs.get( 'topic_vec_file', None )
        self.W = kwargs.get( 'load_embedding_word_count', -1 )
        K = kwargs.get( 'K', 30 )

        self.max_l = kwargs.get( 'max_l', 6 )
        self.init_l = kwargs.get( 'init_l', 1 )
        self.max_grad_norm = kwargs.get( 'max_grad_norm', 0 )
        self.grad_scale_Em_base = kwargs.get( 'grad_scale_Em_base', 0 )
        # number of top words to output
        self.topW = kwargs.get( 'topW', 12 )
        self.topDim = kwargs.get( 'topDim', 10 )
        self.topTopicMassFracPrintThres = kwargs.get( 'topTopicMassFracPrintThres', 1 )
        self.alpha0 = kwargs.get( 'alpha0', 5 )
        self.alpha1 = kwargs.get( 'alpha1', 1 )
        # initial learning rate
        self.delta = self.iniDelta = kwargs.get( 'iniDelta', 0.1 )
        self.MAX_EM_ITERS = kwargs.get( 'MAX_EM_ITERS', 100 )
        self.topicDiff_tolerance = kwargs.get( 'topicDiff_tolerance', 2e-3 )
        self.zero_topic0 = kwargs.get( 'zero_topic0', True )
        self.smoothing_context_size = kwargs.get( 'smoothing_context_size', 0 )
        self.context_weight = kwargs.get( 'context_weight', 0.5 )
        self.appendLogfile = kwargs.get( 'appendLogfile', False )
        self.customStopwords = kwargs.get( 'customStopwords', "" )
        self.remove_stop = kwargs.get( 'remove_stop', True )
        self.seed = kwargs.get( 'seed', 0 )
        self.verbose = kwargs.get( 'verbose', 1 )
        self.printTopic_iterNum = kwargs.get( 'printTopic_iterNum', 20 )
        self.calcSum_pi_v_iterNum = kwargs.get( 'calcSum_pi_v_iterNum', 1 )
        self.VStep_iterNum = kwargs.get( 'VStep_iterNum', 1 )
        self.calcLike_iterNum = kwargs.get( 'calcLike_iterNum', 1 )
        #self.max_theta_to_avg_ratio = kwargs.get( 'max_theta_to_avg_ratio', -1 )
        #self.big_theta_step_ratio = kwargs.get( 'big_theta_step_ratio', 2 )

        self.useDrdtApprox = kwargs.get( 'useDrdtApprox', False )
        self.Mstep_sample_topwords = kwargs.get( 'Mstep_sample_topwords', 0 )
        self.normalize_vecs = kwargs.get( 'normalize_vecs', False )
        self.rebase_vecs = kwargs.get( 'rebase_vecs', False )
        self.rebase_norm_thres = kwargs.get( 'rebase_norm_thres', 0 )
        self.evalKmeans = kwargs.get( 'evalKmeans', False )
        
        self.D = 0
        self.docsName = "Uninitialized"

        #self.alpha = np.array( [ self.alpha1 ] * self.K )
        #if self.zero_topic0:
        #    self.alpha[0] = self.alpha0

        self.vocab_dict = loadUnigramFile(self.unigramFilename)
        
        embedding_npyfile = self.word_vec_file + ".npy"
        if os.path.isfile(embedding_npyfile):
            print "Load embeddings from npy file '%s'" %embedding_npyfile
            embedding_arrays = np.load(embedding_npyfile)
            self.V, self.vocab, self.word2ID, skippedWords_whatever = embedding_arrays
        else:
            self.V, self.vocab, self.word2ID, skippedWords_whatever = load_embeddings(self.word_vec_file, self.W)
            embedding_arrays = np.array( [ self.V, self.vocab, self.word2ID, skippedWords_whatever ] )
            print "Save embeddings to npy file '%s'" %embedding_npyfile
            np.save( embedding_npyfile, embedding_arrays )
            
        # map of word -> id of all words with embeddings
        vocab_dict2 = {}
        
        if self.normalize_vecs:
            self.Vnorm = np.array( [ normF(x) for x in self.V ] )
            for i,w in enumerate(self.vocab):
                if self.Vnorm[i] == 0:
                    print "WARN: %s norm is 0" %w
                    # set to 1 to avoid "divided by 0 exception"
                    self.Vnorm[i] = 1
                
            self.V /= self.Vnorm[:, None]
            
        # dimensionality of topic/word embeddings
        self.N0 = self.V.shape[1]
        # number of all words
        self.vocab_size = self.V.shape[0]
        
        # words both in the embedding vector file and in the unigram file
        vocab2 = []
        # set unigram probs
        u2 = []    
        for wid,w in enumerate(self.vocab):
            if w not in self.vocab_dict:
                continue
            u2.append( self.vocab_dict[w][2] )
            #vocab2.append(w)
            vocab_dict2[w] = wid

        u2 = np.array(u2)
        self.u = normalize(u2)
        #self.vocab = vocab2
        self.vocab_dict = vocab_dict2

        # u2 is the top "Mstep_sample_topwords" words of u, 
        # used for a sampling inference (i.e. only the most 
        # important "Mstep_sample_topwords" words are used) in the M-step
        # if Mstep_sample_topwords == 0, sampling is disabled
        if self.Mstep_sample_topwords == 0:
            self.Mstep_sample_topwords = self.vocab_size
            self.u2 = self.u
            self.V2 = self.V
        else:
            self.u2 = self.u[:self.Mstep_sample_topwords]
            self.u2 = normalize(self.u2)
            self.V2 = self.V[:self.Mstep_sample_topwords]
            
        customStopwordList = re.split( r"\s+", self.customStopwords )
        for stop_w in customStopwordList:
            stopwordDict[stop_w] = 1
        print "Custom stopwords: %s" %( ", ".join(customStopwordList) )
        
        if 'fileLogger' not in kwargs:
            self.logfilename = kwargs.get( 'logfilename', "topicvecDir" )
            self.fileLogger = initFileLogger( self.logfilename, self.appendLogfile )
        else:
            self.fileLogger = kwargs['fileLogger']

        self.fileLogger.debug( "topicvecDir() init at %s", time.ctime() )
        self.precompute()
        self.setK(K)

        self.docs_name = []
        self.docs_idx = []
        self.docs_wids = []
        self.wid2freq = []
        self.wids_freq = []
        self.expVT = None
        self.T = self.r = self.sum_pi_v = None
        self.docs_L = []
        self.docs_Pi = []
        self.docs_theta = []
        self.totalL = 0
        self.kmeans_xtoc = self.kmeans_distances = None
        # current iteration number
        self.it = 0

    def setK(self, K):
        self.K = K
        self.alpha = np.array( [ self.alpha1 ] * self.K )
        if self.zero_topic0:
            self.alpha[0] = self.alpha0
        # K rows of Ev
        # EV: K x N0
        if self.useDrdtApprox:
            self.EV = np.tile( self.Ev, (self.K, 1) )

    def precompute(self):
        print "Precompute matrix u_V"
        # each elem of u multiplies each row of V
        # Pw_V: Mstep_sample_topwords x N0
        self.Pw_V = self.u2[:, np.newaxis] * self.V2

        if self.useDrdtApprox:
            print "Precompute vector Ev"
            self.Ev = np.dot(self.u, self.V)
            print "Precompute matrix Evv...",
            self.Evv = np.zeros( (self.N0, self.N0) )
            for wid in xrange(self.vocab_size):
                self.Evv += self.u[wid] * np.outer( self.V[wid], self.V[wid] )
            print "Done."
                    
    def calcEm(self, docs_Pi):
        Em = np.zeros(self.K)
        for d in xrange( len(docs_Pi) ):
            Em += np.sum( docs_Pi[d], axis=0 )
        return Em

    def calcLoglikelihood(self):
        totalLoglike = 0

        for d in xrange(self.D):
            theta = self.docs_theta[d]
            Pi = self.docs_Pi[d]

            theta0 = np.sum(theta)
            entropy = np.sum( gammaln(theta) ) - gammaln(theta0)
            entropy += (theta0 - self.K) * psi(theta0) - np.sum( (theta - 1) * psi(theta) )
            entropy -= np.sum( Pi * np.log(Pi) )
            # this Em is not the total Em calculated by calcEm()
            # Em[k] = sum_j Pi[j][k]
            Em = np.sum( Pi, axis=0 )
            Em_Ephi = ( Em + self.alpha - 1 ) * ( psi(theta) - psi(theta0) )
            sum_r_pi = np.dot( Em, self.r )
            loglike = entropy + np.sum(Em_Ephi) + np.trace( np.dot( self.T, self.sum_pi_v.T ) ) + sum_r_pi

            totalLoglike += loglike
        return totalLoglike

    def updateTheta(self):
        for d in xrange(self.D):
            L = self.docs_L[d]
            self.docs_theta[d] = np.sum( self.docs_Pi[d], axis=0 ) + self.alpha
            
#            # if this component prior becomes too big, it will absorb irrelevant words -- a vicious circle
#            if self.max_theta_to_avg_ratio > 0:
#                self.fileLogger.debug( "Theta of doc %i:" %d )
#                self.fileLogger.debug( self.docs_theta[d] )
#                theta_sum = np.sum(self.docs_theta[d])
#                theta_thres = self.max_theta_to_avg_ratio * theta_sum / self.K
#                thetas_k = [ [ self.docs_theta[d][k], k ] for k in xrange(self.K) ]
#                thetas_k_sorted = sorted( thetas_k, key=lambda theta_k: theta_k[0] )
#                if thetas_k_sorted[-1][0] > theta_thres:
#                    i = 0
#                    # find the first component that is too big
#                    while i < self.K:
#                        if thetas_k_sorted[i][0] > theta_thres:
#                            break
#                        i += 1
#                    # set to 1 to disable initStepBoost
#                    initStepBoost = 1
#                    while i < self.K:
#                        if thetas_k_sorted[i][0] > thetas_k_sorted[i-1][0] * self.big_theta_step_ratio * initStepBoost:
#                            thetas_k_sorted[i][0] = thetas_k_sorted[i-1][0] * self.big_theta_step_ratio * initStepBoost
#                            k = thetas_k_sorted[i][1]
#                            orig_thetaK = self.docs_theta[d][k]
#                            self.docs_theta[d][k] = thetas_k_sorted[i][0]
#                            self.fileLogger.debug( "theta-%d %.3f -> %.3f", k, orig_thetaK, self.docs_theta[d][k] )
#                        i += 1
#                        initStepBoost = 1

    def updatePi(self, docs_theta):
        docs_Pi = []
        psiDocs_theta = psi(docs_theta)

        for d in xrange(self.D):
            if d % 50 == 49 or d == self.D - 1:
                print "\r%d" %(d+1),

            wids = self.docs_wids[d]
            L = self.docs_L[d]
            
            # faster computation, more memory
            if L <= 20000:
                # Vd: L x N0
                Vd = self.V[wids]
                TV = np.dot( Vd, self.T.T )
                Pi = np.exp( psiDocs_theta[d] + TV + self.r )
                
            else:
                Pi = np.zeros( (L, self.K) )
    
                for i,wid in enumerate(wids):
                    v = self.V[wid]
                    # smooth the current vector using context (preceding) vectors
                    # default: disabled
    #                if self.smoothing_context_size and i > 0:
    #                    j = max( 0, i - self.smoothing_context_size )
    #                    totalWeight = 1
    #                    for x in xrange(j, i):
    #                        v += self.context_weight * self.V[ wids[x] ]
    #                        totalWeight += self.context_weight
    #                    v /= totalWeight
    
                    Tv = np.dot( self.T, v )
                    Pi[i] = np.exp( psiDocs_theta[d] + Tv + self.r )

            Pi = normalize(Pi)
            docs_Pi.append(Pi)

        return docs_Pi

    # T is fed as an argument to provide more flexibility
    def calcTopicResiduals(self, T):
        # VT_{i,j} = v_wi' t_j
        VT = np.dot(self.V2, T.T)

        # expVT_{i,j} = exp(v_wi' t_j)
        # used in the computation of drdt
        # expVT: Mstep_sample_topwords x K
        self.expVT = np.exp(VT)

        r = -np.log( np.dot(self.u2, self.expVT) )

        return r

    def updateTopicEmbeddings(self):
        Em = self.calcEm( self.docs_Pi )
        if self.grad_scale_Em_base > 0 and np.sum(Em) > self.grad_scale_Em_base:
            grad_scale = self.grad_scale_Em_base / np.sum(Em)
        else:
            grad_scale = 1
                        
        # Em: 1 x K vector
        # r: 1 x K vector
        # Em_exp_r: 1 x K vector
        Em_exp_r = Em * np.exp(self.r)

        debug = False

        if debug or self.useDrdtApprox:
            # EV_XT: K x N0
            # Em_drdT_approx: N0 x K
            EV_XT = self.EV + np.dot( self.T, self.Evv )
            Em_drdT_approx = EV_XT.T * Em_exp_r
        if debug or not self.useDrdtApprox:
            # d_EwVT_dT: K x N0
            d_EwVT_dT = np.dot( self.expVT.T, self.Pw_V )
            # Em_drdT_exact: N0 x K
            Em_drdT_exact = d_EwVT_dT.T * Em_exp_r

        #if debug:
        #    pdb.set_trace()

        # Em_drdT: K x N0
        if self.useDrdtApprox:
            Em_drdT = Em_drdT_approx.T
        else:
            Em_drdT = Em_drdT_exact.T

        # diffMat: K x N0
        diffMat = self.sum_pi_v - Em_drdT
        diffMat *= self.delta * grad_scale
        
        maxTStep = np.max( np.linalg.norm( diffMat, axis=1 ) )
        #if self.it >= 50:
        #    pdb.set_trace()
            
        if self.max_grad_norm > 0 and maxTStep > self.max_grad_norm:
            diffMat *= self.max_grad_norm / maxTStep
        T2 = self.T + diffMat

        maxTStep = np.max( np.linalg.norm( diffMat, axis=1 ) )

        # self.max_l == 0: do not do normalization
        if self.max_l > 0:
            for k in xrange( self.K ):
                # do normalization only if the magnitude > self.max_l
                if np.linalg.norm( T2[k] ) > self.max_l:
                    T2[k] = self.max_l * normalizeF( T2[k] )

        if self.zero_topic0:
            T2[0] = np.zeros(self.N0)

        r2 = self.calcTopicResiduals(T2)
        topicDiff = np.linalg.norm( self.T - T2 )
        return T2, r2, topicDiff, maxTStep, diffMat

    # Pi: L x K
    # sum_pi_v: K x N0
    def calcSum_pi_v(self):
        self.sum_pi_v = np.zeros( (self.K, self.N0) )

        for d in xrange(self.D):
            Pi = self.docs_Pi[d]
            wids = self.docs_wids[d]
            #L = self.docs_L[d]
            #for i in xrange(L):
            #    self.sum_pi_v += np.outer( Pi[i], self.V[ wids[i] ] )
            self.sum_pi_v += np.dot( Pi.T, self.V[wids] )
            
    # the returned outputter always output to the log file
    # screenVerboseThres controls when the generated outputter will output to screen
    # when self.verbose >= screenVerboseThres, screen output is enabled
    # in the batch mode for multiple files, typically self.verbose == 0
    # then by default no screen output anyway
    # in the single file mode, typically self.verbose == 1
    # then in printTopWordsInTopic(),
    #   outputToScreen == True => screenVerboseThres == 1
    #       with screen output
    #   outputToScreen == False => screenVerboseThres == 2
    #       no screen output
    # in other places by default screenVerboseThres==1, with screen output
    def genOutputter(self, screenVerboseThres=1):
        def screen_log_output(s):
            self.fileLogger.debug(s)
            if self.verbose >= screenVerboseThres:
                print s
        return screen_log_output

    def genProgressor(self):
        def screen_log_progress(s):
            self.fileLogger.debug(s)
            if self.verbose == 0:
                print "\r%s    \r" %s,
            else:
                print s
        return screen_log_progress

    # topTopicMassFracPrintThres: when a topic's fraction Em[k]/L > topTopicMassFracPrintThres/K, print it
    def printTopWordsInTopic( self, docs_theta, outputToScreen=False ):
        wids2 = self.wid2freq.keys()
        wids_topics_sim = np.dot( normalizeF( self.V[wids2] ), normalizeF(self.T).T )
        wids_topics_dot = np.dot( self.V[wids2], self.T.T )

        # row ID: de-duplicated id, also the row idx in the 
        # matrices wids_topics_sim and wids_topics_dot
        wid2rowID = {}
        for i, wid in enumerate(wids2):
            wid2rowID[wid] = i

        # the topic prop of each word, indexed by the row ID
        row_topicsProp = np.zeros( wids_topics_sim.shape )
        # word occurrences, indexed bythe row ID
        row_wordOccur = np.array( self.wid2freq.values() )

        if self.evalKmeans:
            Em = np.bincount(self.kmeans_xtoc)
        else:
            docs_Pi = self.updatePi(docs_theta)
            Em = self.calcEm(docs_Pi)

        # tids is sorted topic IDs from most frequent to least frequent
        tids = sorted( range(self.K), key=lambda k: Em[k], reverse=True )
        for i,k in enumerate(tids):
            # below the average proportion * topTopicMassFracPrintThres
            if Em[k] < self.topTopicMassFracPrintThres * self.totalL / self.K:
                break

        # cut_i is the cut point of tids: tids[:cut_i] will be printed
        # if i==0, no topic has enough proportion to be printed.
        # this may happen when topicThres is too big. in this case, print the principal topic
        if i == 0:
            cut_i = 1
        else:
            cut_i = i

        for d in xrange(self.D):
            for i in xrange(self.docs_L[d]):
                wid = self.docs_wids[d][i]
                rowID = wid2rowID[wid]
                if self.evalKmeans:
                    k = self.kmeans_xtoc[rowID]
                    row_topicsProp[rowID][k] += 1
                else:
                    row_topicsProp[rowID] += docs_Pi[d][i]
            
        # the topic prop of each word, indexed by the row ID
        # take account of the word freq, but dampen it with sqrt
        # so that more similar, less frequent words have chance to be selected
        # doing average does not consider freq, not good either
        row_topicsDampedProp = row_topicsProp / np.sqrt(row_wordOccur)[:,None]

        W = len(self.vocab)
        # number of unique words in the docs
        W2 = len(wids2)

        if outputToScreen:
            out = self.genOutputter(1)
        else:
            out = self.genOutputter(2)

        out("")
        out( "Em:\n%s\n" %Em )
        out("Topic magnitudes:")

        topicMagnitudes = np.linalg.norm(self.T, axis=1)

        out(topicMagnitudes)
        out("")

        # selected tids to output
        selTids = tids[:cut_i]
        selTids = np.array(selTids)

        # always output topic 0
        # if topic 0 is not in selTids, append it
        if len( np.where(selTids == 0)[0] ) == 0:
            selTids = np.append( selTids, 0 )

        for k in selTids:
            out( "Topic %d (%.2f): %.1f%%" %( k, np.linalg.norm( self.T[k] ), 100 * Em[k] / self.totalL ) )

            rowID_sorted = sorted( range(W2), key=lambda rowID: row_topicsDampedProp[rowID, k], reverse=True )

            out("Most relevant words:")

            line = ""
            for rowID in rowID_sorted[:self.topW]:
                wid = wids2[rowID]
                topicAvgProp = row_topicsDampedProp[rowID, k]
                topicProp = row_topicsProp[rowID, k]
                sim = wids_topics_sim[rowID, k]
                dotprod = wids_topics_dot[rowID, k]

                line += "%s (%d,%d): %.2f/%.2f/%.2f/%.2f " %( self.vocab[wid], wid, self.wid2freq[wid],
                                    topicAvgProp, topicProp, sim, dotprod )

            out(line)

            if np.linalg.norm( self.T[k] ) == 0:
                continue

            V_topic_dot = np.dot( self.V2, self.T[k] )
            V_topic_sim = V_topic_dot / np.linalg.norm( self.V2, axis=1 ) / np.linalg.norm( self.T[k] )

            wid_sorted = sorted( xrange(self.Mstep_sample_topwords), 
                                key=lambda wid: V_topic_sim[wid], reverse=True )

            out("Most similar words in vocab:")

            line = ""
            for wid in wid_sorted[:self.topW]:
                sim = V_topic_sim[wid]
                dotprod = V_topic_dot[wid]
                line += "%s: %.2f/%.2f " %( self.vocab[wid], sim, dotprod )

            out(line)
            out("")

    def docSentences2wids( self, docs_wordsInSentences ):
        docs_wids = []
        docs_idx = []
        countedWC = 0
        outvocWC = 0
        stopwordWC = 0
        wid2freq = {}
        wids_freq = np.zeros(self.vocab_size)

        for d, wordsInSentences in enumerate(docs_wordsInSentences):
            wids = []
            for sentence in wordsInSentences:
                for w in sentence:
                    w = w.lower()
                    if self.remove_stop and w in stopwordDict:
                        stopwordWC += 1
                        continue

                    if w in self.vocab_dict:
                        wid = self.vocab_dict[w]
                        wids.append(wid)
                        wids_freq[wid] += 1

                        if wid not in wid2freq:
                            wid2freq[wid] = 1
                        else:
                            wid2freq[wid] += 1
                        countedWC += 1
                    else:
                        outvocWC += 1

            # skip empty documents
            if len(wids) > 0:
                docs_wids.append(wids)
                docs_idx.append(d)

        # out0 prints both to screen and to log file, regardless of the verbose level
        out0 = self.genOutputter(0)
        out1 = self.genOutputter(1)

        out0( "%d docs scanned, %d kept. %d words kept, %d unique. %d stop words, %d out voc" %( len(docs_wordsInSentences),
                                                            len(docs_idx), countedWC, len(wid2freq), stopwordWC, outvocWC ) )

        wid_freqs = sorted( wid2freq.items(), key=lambda kv: kv[1], reverse=True )
        out1("Top words:")
        line = ""
        for wid, freq in wid_freqs[:30]:
            line += "%s(%d): %d " %( self.vocab[wid], wid, freq )
        out1(line)
        return docs_idx, docs_wids, wid2freq, wids_freq

    def setDocs( self, docs_wordsInSentences, docs_name ):
        self.totalL = 0
        self.docs_L = []
        self.docs_name = []

        self.docs_idx, self.docs_wids, self.wid2freq, self.wids_freq = \
                                    self.docSentences2wids(docs_wordsInSentences)

        for doc_idx in self.docs_idx:
            self.docs_name.append( docs_name[doc_idx] )
        for wids in self.docs_wids:
            self.docs_L.append( len(wids) )
        self.totalL = sum(self.docs_L)

        avgV = np.zeros(self.N0)
        sum_freq = 0
        for wid, freq in self.wid2freq.iteritems():
            avgV += self.V[wid] * freq
            sum_freq += freq
        avgV /= sum_freq
        norm_avgV = np.linalg.norm(avgV)
        print "Norm of avg vector: %.2f" %norm_avgV
        if self.rebase_vecs and norm_avgV >= self.rebase_norm_thres:
            self.V -= avgV
            # update the precomputed matrices/vectors
            self.precompute()
            
#        if self.useLocalU:
#            self.local_u = self.wids_freq / self.totalL
#            assert abs( np.sum(self.local_u) - 1 ) < 1e-5, \
#                "Local unigram empirical prob vector local_u wrongly normalized: sum=%.3f != 1" %np.sum(self.local_u)

        self.D = len(self.docs_name)
        if self.D == 0:
            print "WARN: Document set is empty after preprocessing."
        if self.D == 1:
            self.docsName = "'%s'" %(docs_name[0])
        else:
            self.docsName = "'%s'...(%d docs)" %( docs_name[0], self.D )

        return self.docs_idx
        
    def kmeans( self, maxiter=10 ):
        """ centers, Xtocentre, distances = topicvec.kmeans( ... )
        in:
            X: M x N0
            centers K x N0: initial centers, e.g. random.sample( X, K )
            iterate until the change of the average distance to centers
                is within topicDiff_tolerance of the previous average distance
            maxiter
            metric: cosine
            self.verbose: 0 silent, 2 prints running distances
        out:
            centers, K x N0
            Xtocentre: each X -> its nearest center, ints M -> K
            distances, M
        see also: kmeanssample below, class Kmeans below.
        """
    
        wids2 = self.wid2freq.keys()
        weights = np.array( self.wid2freq.values() )
        
        X = normalizeF( self.V[wids2] )
        centers = randomsample( X, self.K )
        
        if self.verbose:
            print "kmeans: X %s  centers %s  tolerance=%.2g  maxiter=%d" %(
                X.shape, centers.shape, self.topicDiff_tolerance, maxiter )
        
        M = X.shape[0]
        allx = np.arange(M)
        prevdist = 0
        
        for jiter in range( 1, maxiter+1 ):
            D = cdist( X, centers, metric='cosine' )  # |X| x |centers|
            xtoc = D.argmin(axis=1)  # X -> nearest center
            distances = D[allx,xtoc]
            #avdist = distances.mean()  # median ?
            avdist = (distances * weights).sum() / weights.sum()
            
            if self.verbose >= 2:
                print "kmeans: av |X - nearest center| = %.4g" % avdist
                
            if (1 - self.topicDiff_tolerance) * prevdist <= avdist <= prevdist \
            or jiter == maxiter:
                break
                
            prevdist = avdist
            
            for jc in range(self.K):  # (1 pass in C)
                c = np.where( xtoc == jc )[0]
                if len(c) > 0:
                    centers[jc] = ( X[c] * weights[c, None] ).mean( axis=0 )
                    
        if self.verbose:
            print "kmeans: %d iterations  cluster sizes:" % jiter, np.bincount(xtoc)
            
        if self.verbose >= 2:
            r50 = np.zeros(self.K)
            r90 = np.zeros(self.K)
            for j in range(self.K):
                dist = distances[ xtoc == j ]
                if len(dist) > 0:
                    r50[j], r90[j] = np.percentile( dist, (50, 90) )
            print "kmeans: cluster 50% radius", r50.astype(int)
            print "kmeans: cluster 90% radius", r90.astype(int)
        
        self.T = centers
        self.kmeans_xtoc = xtoc
        self.kmeans_distances = distances    
    
    def inferTopicProps( self, T, MAX_ITERS=5 ):

        self.T = T
        self.r = self.calcTopicResiduals(T)
        # uniform prior
        self.docs_theta = np.ones( (self.D, self.K) )
        it_loglikeer = 0
        loglike = 0

        for i in xrange(MAX_ITERS):
            iterStartTime = time.time()
            docs_Pi2 = self.docs_Pi
            self.docs_Pi = self.updatePi( self.docs_theta )
            self.updateTheta()

            if i > 0:
                docs_Pi_diff = np.zeros(self.D)
                for d in xrange(self.D):
                    docs_Pi_diff[d] = np.linalg.norm( self.docs_Pi[d] - docs_Pi2[d] )
                max_Pi_diff = np.max(docs_Pi_diff)
                total_Pi_diff = np.sum(docs_Pi_diff)
            else:
                max_Pi_diff = 0
                total_Pi_diff = 0

            """if i % 5 == 0:
                self.calcSum_pi_v()
                loglike = self.calcLoglikelihood()
                it_loglikeer = i
            """
            iterDur = time.time() - iterStartTime
            print "Iter %d loglike(%d) %.2f, Pi diff total %.3f, max %.3f. %.1fs" %( i, it_loglikeer,
                                                loglike, total_Pi_diff, max_Pi_diff, iterDur )

        docs_Em = np.zeros( (self.D, self.K) )
        for d, Pi in enumerate(self.docs_Pi):
            docs_Em[d] = np.sum( Pi, axis=0 )

        return docs_Em, self.docs_Pi

    def inference(self):
        if self.D == 0:
            print "document set is empty or uninitialized"
            return None, None, None, None

        startTime = time.time()
        startTimeStr = timeToStr(startTime)

        # out0 prints both to screen and to log file, regardless of the verbose level
        out0 = self.genOutputter(0)
        out1 = self.genOutputter(1)

        out0( "%d topics." %(self.K) )
        out0( "%s inference starts at %s" %( self.docsName, startTimeStr ) )

        progress = self.genProgressor()

        self.T = np.zeros( ( self.K, self.N0 ) )

        if self.seed != 0:
            np.random.seed(self.seed)
            out0( "Seed: %d" %self.seed )

        for k in xrange(0, self.K):
            self.T[k] = np.random.multivariate_normal( np.zeros(self.N0), np.eye(self.N0) )
            if self.init_l > 0:
                self.T[k] = self.init_l * normalizeF(self.T[k])

        if self.zero_topic0:
            self.T[0] = np.zeros(self.N0)

    #    sum_v = np.zeros(N0)
    #    for wid in wids:
    #        sum_v += V[wid]
    #
    #    T[0] = self.max_l * normalizeF(sum_v)
        #self.fileLogger.debug("avg_v:")
        #self.fileLogger.debug(T[0])

        self.r = self.calcTopicResiduals(self.T)
        # initialized as uniform over topics
        self.docs_theta = np.ones( (self.D, self.K) )

        lastIterEndTime = time.time()
        print "\rInitial learning rate: %.2f" %(self.iniDelta)

        self.docs_Pi = self.updatePi( self.docs_theta )
        self.updateTheta()

        self.calcSum_pi_v()
        loglike = self.calcLoglikelihood()
        loglike2 = 0

        self.it = 0

        iterDur = time.time() - lastIterEndTime
        lastIterEndTime = time.time()

        print "\rIter %d: loglike %.2f, %.1fs" %( self.it, loglike, iterDur )

        # an arbitrary number to satisfy pylint
        topicDiff = 100000
        it_loglike = 0

        unif_docs_theta = np.ones( (self.D, self.K) )
        Ts_loglikes = []

        while self.it == 0 or ( self.it < self.MAX_EM_ITERS and topicDiff > self.topicDiff_tolerance ):
        #while self.it == 0 or ( self.it < self.MAX_EM_ITERS and abs(loglike - loglike2) > loglike_tolerance ):
            self.it += 1
            self.fileLogger.debug( "EM Iter %d:", self.it )

            self.delta = self.iniDelta / ( self.it + 1 )
            # T, r not updated inside updateTopicEmbeddings()
            # because sometimes we want to keep the original T, r
            self.T, self.r, topicDiff, maxTStep, diffMat = self.updateTopicEmbeddings()
            
            if self.it % self.VStep_iterNum == 0:
                self.updateTheta()
                self.docs_Pi = self.updatePi( self.docs_theta )

            # calcSum_pi_v() takes a long time on a large corpus
            # so it can be done once every a few iters, with slight loss of performance
            # on 20news and reuters, calcSum_pi_v() is fast enough and this acceleration is not necessary
            if self.it <= 5 or self.it == self.MAX_EM_ITERS or self.it % self.calcSum_pi_v_iterNum == 0:
                self.calcSum_pi_v()

            loglike2 = loglike
            loglike = self.calcLoglikelihood()
            it_loglike = self.it

            iterDur = time.time() - lastIterEndTime
            lastIterEndTime = time.time()

            iterStatusMsg = "\rIter %d: loglike(%d) %.2f, topicDiff %.4f, maxTStep %.3f, %.1fs" %( self.it,
                                           it_loglike, loglike, topicDiff, maxTStep, iterDur )

            if self.it % self.printTopic_iterNum == 0:
                out0(iterStatusMsg)

                if self.verbose >= 2:
                    self.fileLogger.debug( "T[:,%d]:", self.topDim )
                    self.fileLogger.debug( self.T[ :, :self.topDim ] )

                    self.fileLogger.debug("r:")
                    self.fileLogger.debug(self.r)

                #self.printTopWordsInTopic(unif_docs_theta, False)
                self.printTopWordsInTopic(self.docs_theta, False)
            else:
                print "%s  \r" %iterStatusMsg,
                self.fileLogger.debug(iterStatusMsg)
                Em = self.calcEm( self.docs_Pi )
                self.fileLogger.debug( "Em:\n%s\n", Em )
                
            Ts_loglikes.append( [ self.it, self.T, loglike ] )
            
        if self.verbose >= 1:
            # if == 0, topics has just been printed in the while loop
            if self.it % self.printTopic_iterNum != 0:
                #self.printTopWordsInTopic(unif_docs_theta, False)
                self.printTopWordsInTopic(self.docs_theta, False)

        endTime = time.time()
        endTimeStr = timeToStr(endTime)
        inferDur = int(endTime - startTime)
        
        print
        out0( "%s inference ends at %s. %d iters, %d seconds." %( self.docsName, endTimeStr, self.it, inferDur ) )

        Em = self.calcEm( self.docs_Pi )
        docs_Em = np.zeros( (self.D, self.K) )
        for d, Pi in enumerate(self.docs_Pi):
            docs_Em[d] = np.sum( Pi, axis=0 )

        # sort according to loglike
        Ts_loglikes_sorted = sorted( Ts_loglikes, key=lambda T_loglike: T_loglike[2], reverse=True )
        # best T is the last T
        if Ts_loglikes_sorted[0][0] == self.it:
            best_last_Ts = [ Ts_loglikes_sorted[0], None ]
        else:
            best_last_Ts = [ Ts_loglikes_sorted[0], Ts_loglikes[-1] ]

        return best_last_Ts, Em, docs_Em, self.docs_Pi

