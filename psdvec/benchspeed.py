import numpy as np
import time

class Timer(object):
    def __init__(self, name=None):
        self.name = name
        self.tstart = time.time()
        self.tlast = self.tstart
        self.firstCall = True

    def getElapseTime(self, isStr=True):
        totalElapsed = time.time() - self.tstart
        # elapsed time since last call
        interElapsed = time.time() - self.tlast
        self.tlast = time.time()

        firstCall = self.firstCall
        self.firstCall = False

        if isStr:
            if self.name:
                if firstCall:
                    return '%s elapsed: %.2f' % ( self.name, totalElapsed )
                return '%s elapsed: %.2f/%.2f' % ( self.name, totalElapsed, interElapsed )
            else:
                if firstCall:
                    return 'Elapsed: %.2f' % ( totalElapsed )
                return 'Elapsed: %.2f/%.2f' % ( totalElapsed, interElapsed )
        else:
            return totalElapsed, interElapsed

    def printElapseTime(self):
        print self.getElapseTime()

def timeToStr(timeNum, fmt="%H:%M:%S"):
    timeStr = time.strftime(fmt, time.localtime(timeNum))
    return timeStr

def block_factorize( core_size, noncore_size, N0, tikhonovCoeff ):
    # new WGsum: noncore_size * core_size
    WGsum = np.random.random((noncore_size,core_size))
    Wsum = np.random.random((noncore_size,core_size))
    Wsum[ np.isclose(Wsum,0) ] = 0.001
    Gwmean = WGsum

    V1 = np.random.random((core_size,N0))
    # embeddings of noncore words
    # new V2: noncore_size * N0
    V2 = np.zeros( ( noncore_size, N0 ), dtype=np.float32 )
    Tikhonov = np.identity(N0) * tikhonovCoeff

    timer = Timer()

    print "Begin finding embeddings of non-core words"

    # Find each noncore word's embedding
    for i in xrange(noncore_size):
        # core_size
        wi = Wsum[i]
        # new VW: N0 * core_size
        VW = V1.T * wi
        # new VWV: N0 * N0
        VWV = VW.dot(V1)
        if False:
            VWV_Tik = VWV + Tikhonov
            V2[i] = np.linalg.inv(VWV_Tik).dot( VW.dot(Gwmean[i]) )
        if i >= 0 and i % 100 == 99:
            print "\r%d / %d." %(i+1,noncore_size),
            print timer.getElapseTime(), "\r",

    print

block_factorize(15000, 1000, 500, 2)
#block_factorize(15000, 10000, 50, 2)
