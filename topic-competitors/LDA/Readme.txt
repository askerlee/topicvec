输入语料: 采用sklearn自带的20 newsgroups (~20000篇文档)和nltk自带的reuters(10788篇文档)

ldaExp.py: 用gensim从指定语料(20newsgroup 或者reuters，通过命令行指定二者之一)中学习doc-topic分布并保存为 '语料名-train-文档数.svm-lda.txt' 和 '语料名-test-文档数.svm-lda.txt'

classEval.py: 用ldaExp.py生成的 '语料名-train-文档数.svm-lda.txt' 作为特征文件，进行训练，在 '语料名-test-文档数.svm-lda.txt' 上测试分类效果。

corpusLoader.py: 把sklearn的20newsgroups和nltk的reuters统一的语料访问接口。

运行命令实例:
- python ldaExp.py 20news
生成 '20news-train-11314.svm-lda.txt' 和 '20news-test-7532.svm-lda.txt'
- python classEval.py 20news lda
训练并评估模型效果
