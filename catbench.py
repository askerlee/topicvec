import os

testsetNames = [ "ap", "battig", "esslli" ]
testsetCatNums = [ 13, 10, 6 ]
algNames = [ "PSDVec", "word2vec", "CCA" ]
CLmethods = [ "rbr", "direct", "graph" ]
vclusterPath = "D:\\cluto-2.1.2\\MSWIN-x86_64-openmp\\vcluster.exe"
testsetDir = "./concept categorization"

for CLmethod in CLmethods:
    for i, testsetName in enumerate(testsetNames):
        for algName in algNames:
            vecFilename = testsetDir + "/" + testsetName + "-" + algName + ".vec"
            labelFilename = testsetDir + "/" + testsetName + "-" + algName + ".label"
            catNum = testsetCatNums[i]
            print "%s on %s using %s:" %( algName, testsetName, CLmethod )
            stream = os.popen( '%s -rclassfile="%s" -clmethod=%s "%s" %d' %( vclusterPath, 
                               labelFilename, CLmethod, vecFilename, catNum ) )
            output = stream.read()
            lines = output.split("\n")
            for line in lines:
                if line.find("way clustering") >= 0:
                    print line
    print
    