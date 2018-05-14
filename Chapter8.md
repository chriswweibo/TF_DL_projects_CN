

---

## 第8章 检测重复的Quora问题

Quora (www.quora.com)是一个社区驱动的问答网站。用户可以在上边公开的或者匿名的提出问题和回答问题。2017年1月， Quora第一次发布了一个包含问题对的数据集，其中的问题有可能是重复的。重复问题对在语义上是类似的。或者说，尽管两个问题使用不同的词汇，但是传达了相同的意思。为了给用户提供更好的答案集合展示以便尽快找出需要的信息，Quora需要为每一个问题都准备一个页面。这个工程量是非常大。主持人机制对于避免网站上的重复内容是很有帮助的，但是一旦每天回答的问题增多以及历史存量问题的扩大，这种机制就不容易扩展了。这种情况下，基于**自然语言理解（Natural Language Processing，NLP）**和深度学习的自动化项目就成了合适的方案。

本章会介绍如何构建基于TensorFlow的项目，以便理解Quora数据集中句子之间的相似性问题。本章的内容基于Abhishek Thakur （https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur/）的工作。他基于Keras库开发了一套方案 。给出的处理技术也可以用在其它有关语义相似性的问题。在这个项目中，我们会介绍： 

* 文本数据的特征工程Feature engineering on text data
* TF-IDF和SVD
* 基于特征的Word2vec和GloVe算法
* 传统的机器学习模型，例如logistic回归，和使用`xgboost`的梯度加速
* 深度学习模型，包括LSTM，GRU和1D-CNN

学完本章，读者可以训练自己的深度学习模型，来解决类似的问题。首先，让我们看一下Quora数据集。

### 展示数据集
这个数据集仅面向非商业目的 （https://www.quora.com/about/tos），可以在Kaggle竞赛（https://www.kaggle.com/c/quora-question-pairs）和Quora的博客上（https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs）获取。它包含404,351个问题对，其中255,045个负样本（非重复的）和149,306个正样本（重复的）。此数据集中，正样本的比例大约是40%。这说明存在轻度的数据不均衡，但是并不需要专门的处理。事实上，正如Quora博客所公布的，采用初始的采样策略，数据集中的重复样本要比非重复样本多得多。为了构建更加均衡的数据集，负样本需要通过相关问题进行升采样。这些问题是关于相同的主题，但是事实上并不相似。

开始项目之前，读者可以从亚马逊的S3仓库 http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv下载数据集到工作目录中，大约55MB。

下载完成后，读者可以挑几行检查一下数据情况。下图展示了数据集前几行的真实截图：

!["Quora数据集的前几行"](figures\174_1.png)

*Quora数据集的前几行*

进一步查看数据，我们可以发现表示相同含义的问题对，也就是重复问题，如下：

| Quora如何快速的标记需要修改的问题？          | 为什么Quora在我具体写出之前，几秒内就把我的问题标记为需要修改/澄清？ |
| -------------------------------------------- | ------------------------------------------------------------ |
| 为什么特伦普赢得了总统选举？                 | 特朗普是如何赢得2016年总统选举的？                           |
| 从希格斯玻色子发现开始，实际应用有哪些进展？ | 希格斯玻色子的发现有哪些实际益处？                           |

首先看到，重复问题通常包含相同的词语，但是问题长度不同。

另一方面，非重复问题的例子如下：

| 如果我要申请一个诸如Mozilla的大公司，我应该把求职信发给谁？  | 从安全的角度讲，什么车比较好？                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 黑客军团（电视剧）：《黑客军团》是现实生活中黑客和黑客文化的完美演绎吗？对于黑客组织的描绘是真的吗？ | 和真实的网络安全渗透或者一般技术手段相比，《黑客军团》中对黑客的描绘有哪些错误？ |
| 我应该如何搭建网上购物（电子商务）网站？                     | 对于搭建一个大型电子商务网站，哪些网络技术最合适？           |

例子中有些问题很明显不是重复问题，它们的用词也不同，但是其他的一些就很难判断是否相关。例如，第二个例子 对有些人来说就很费时，甚至人类判断也不甚明晰。这两个问题表达了不同的内容：*为什么*和*如何*。粗略看一下，它们两个可能被当做同一个问题。查看更多的例子后，读者可以发现更加可疑的例子，甚至一些数据错误。数据集中当然会有一些异常点（Quora在数据集中也做了提醒）。但是，如果数据是从真实世界的问题中得到的，我们只能接受这种瑕疵，力争有效的健壮的解决方案。


现在，数据探查变得更加量化，一些问题对的统计数据如下：

| 问题1的平均字符数 | 59.57 |
| ----------------- | ----- |
| 问题1的最少字符数 | 1     |
| 问题1的最多字符数 | 623   |
| 问题2的平均字符数 | 60.14 |
| 问题2的最少字符数 | 1     |
| 问题2的最多字符数 | 1169  |


尽管问题2的最值大些，问题1和问题2基本上拥有同样的平均字符数。数据中肯定会有些噪音，因为我们不可能用一个字符构成一个问题。

读者甚至可以通过绘制词云来得到完全不同的视角。数据集中的高频词可以在图中高亮展示：

![](figures\176_1.png)

*图1：Quora数据集中高频词词云*

​        一些词语的存在，例如Hillary Clinton和Donald Trump，提示我们数据是在特定历史时期收集的。其中的许多问题也是阶段性的，只在数据收集的时间点上有意义。其他主题，例如programming language，World War或者earn money，不论是人们的兴趣还是答案的有效性性上将，也许会持久一些。

查看数据之后，现在我们可以确定项目中要争取优化的目标。在这一章中，我们会使用 Throughout the chapter, we will be using accuracy as a metric to evaluate the performance of our models. Accuracy as a measure is simply focused on the effectiveness of the prediction, and it may miss some important differences between alternative models, such as discrimination power (is the model more able to detect duplicates or not?) or the exactness of probability scores (how much margin is there between being a duplicate and not being one?). We chose accuracy based on the fact that this metric was the one decided on by Quora's engineering team to create a benchmark for this dataset (as stated in this blog post of theirs: https:// engineering	.quora	.com	/ Semantic-Question	-Matching	-with	-Deep	-Learnin	g). Using accuracy as the metric makes it easier for us to evaluate and compare our models with the one from Quora's engineering team, and also several other research papers. In addition, in a real-world application, our work may simply be evaluated on the basis of how many times it is just right or wrong, regardless of other considerations.
We can now proceed furthermore in our projects with some very basic feature engineering to start with.

### 基础特征工程

Before starting to code, we have to load the dataset in Python and also provide Python with all the necessary packages for our project. We will need to have these packages installed on our system (the latest versions should suffice, no need for any specific package version):

* `Numpy` 
* `pandas `
* `fuzzywuzzy `
* `python-Levenshtein `
* `scikit-learn `
* `gensim`
* ` pyemd `
* `NLTK`

As we will be using each one of these packages in the project, we will provide specific instructions and tips to install them.
For all dataset operations, we will be using pandas (and Numpy will come in handy, too). To install numpy and pandas:

```
pip install numpy 
pip install pandas
```
The dataset can be loaded into memory easily by using pandas and a specialized data structure, the pandas dataframe (we expect the dataset to be in the same directory as your script or Jupyter notebook):
```python
import pandas as pd 
import numpy as np
data = pd.read_csv('quora_duplicate_questions.tsv', sep='\t') 
data = data.drop(['id', 'qid1', 'qid2'], axis=1)
```
We will be using the pandas dataframe denoted by data throughout this chapter, and also when we work with our TensorFlow model and provide input to it.
We can now start by creating some very basic features. These basic features include lengthbased features and string-based features:

1. Length of question1

2. Length of question2

3. Difference between the two lengths

4. Character length of question1 without spaces

5. Character length of question2 without spaces

6. Number of words in question1

7. Number of words in question2

8. Number of common words in question1 and question2

These features are dealt with one-liners transforming the original input using the pandas package in Python and its method apply:

```python
# length based features
data['len_q1'] = data.question1.apply(lambda x: len(str(x))) 
data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
# difference in lengths of two questions 
data['diff_len'] = data.len_q1 - data.len_q2 
# character length based features 
data['len_char_q1'] = data.question1.apply(lambda x:
                  len(''.join(set(str(x).replace(' ', ''))))) data['len_char_q2'] = data.question2.apply(lambda x:                                             len(''.join(set(str(x).replace(' ', '')))))
# word length based features 
data['len_word_q1'] = data.question1.apply(lambda x:
                                         len(str(x).split())) data['len_word_q2'] = data.question2.apply(lambda x:                                                                    len(str(x).split()))
# common words in the two questions 
data['common_words'] = data.apply(lambda x: 
                                  len(set(str(x['question1'])
                                      .lower().split())
                                      .intersection(set(str(x['question2'])
                                      .lower().split()))), axis=1)
                        
```
For future reference, we will mark this set of features as feature set-1 or fs_1:
```python
fs_1 = ['len_q1', 'len_q2', 'diff_len', 'len_char_q1', 'len_char_q2', 
        'len_word_q1', 'len_word_q2','common_words']
```
This simple approach will help you to easily recall and combine a different set of features in the machine learning models we are going to build, turning comparing different models run by different feature sets into a piece of cake.

### 创建模糊特征

The next set of features are based on fuzzy string matching. Fuzzy string matching is also known as approximate string matching and is the process of finding strings that approximately match a given pattern. The closeness of a match is defined by the number of primitive operations necessary to convert the string into an exact match. These primitive operations include insertion (to insert a character at a given position), deletion (to delete a particular character), and substitution (to replace a character with a new one).
Fuzzy string matching is typically used for spell checking, plagiarism detection, DNA sequence matching, spam filtering, and so on and it is part of the larger family of edit distances, distances based on the idea that a string can be transformed into another one. It is frequently used in natural language processing and other applications in order to ascertain the grade of difference between two strings of characters. 
It is also known as Levenshtein distance, from the name of the Russian scientist, Vladimir Levenshtein, who introduced it in 1965. 
These features were created using the fuzzywuzzy package available for Python (https://pypi.python.org/pypi/fuzzywuzzy). This package uses Levenshtein distance to calculate the differences in two sequences, which in our case are the pair of questions.
The fuzzywuzzy package can be installed using pip3: 

```
pip install fuzzywuzzy
```


As an important dependency, fuzzywuzzy requires the Python-Levenshtein package
(https:// github .com /ztane /python -Levenshtein /) , which is a blazingly fast implementation of this classic algorithm, powered by compiled C code. To make the calculations much faster using fuzzywuzzy, we also need to install the PythonLevenshtein package: 

```
pip install python-Levenshtein
```


The fuzzywuzzy package offers many different types of ratio, but we will be using only the following:

1. QRatio

2. WRatio

3. Partial ratio

4. Partial token set ratio

5. Partial token sort ratio

6. Token set ratio

7. Token sort ratio

Examples of fuzzywuzzy features on Quora data: 

```python
from fuzzywuzzy import fuzz
fuzz.QRatio("Why did Trump win the Presidency?", 
            "How did Donald Trump win the 2016 Presidential Election")
```


  This code snippet will result in the value of 67 being returned:

```python
fuzz.QRatio("How can I start an online shopping (e-commerce) website?",
            "Which web technology is best suitable for building a big E-                   Commerce website?")
```


  In this comparison, the returned value will be 60. Given these examples, we notice that although the values of QRatio are close to each other, the value for the similar question pair from the dataset is higher than the pair with no similarity. Let's take a look at another feature from fuzzywuzzy for these same pairs of questions:


```python
fuzz.partial_ratio("Why did Trump win the Presidency?", 
                   "How did Donald Trump win the 2016 Presidential Election")
```


  In this case, the returned value is 73:


```python
 fuzz.partial_ratio("How can I start an online shopping (e-commerce) website?", 
                    "Which web technology is best suitable for building a big                        E-Commerce website?")
```


  Now the returned value is 57.
  Using the partial_ratio method, we can observe how the difference in scores for these two pairs of questions increases notably, allowing an easier discrimination between being a duplicate pair or not. We assume that these features might add value to our models.
  By using pandas and the fuzzywuzzy package in Python, we can again apply these features as simple one-liners:

```python
data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(
    str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio( 
    str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_ratio'] = data.apply(lambda x:
                                        fuzz.partial_ratio(str(x['question1']),
                                        str(x['question2'])), axis=1)
data['fuzz_partial_token_set_ratio'] = data.apply(lambda x:                                                fuzz.partial_token_set_ratio(str(x['question1']),           str(x['question2'])), axis=1)
data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x:                   fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_token_set_ratio'] = data.apply(lambda x:                                    fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_token_sort_ratio'] = data.apply(lambda x:                               fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
```


  This set of features are henceforth denoted as feature set-2 or fs_2:

```python
fs_2 = ['fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio',
       'fuzz_partial_token_set_ratio', 'fuzz_partial_token_sort_ratio',
       'fuzz_token_set_ratio', 'fuzz_token_sort_ratio']
```


  Again, we will store our work and save it for later use when modeling.


### 借助TF-IDF和SVD特征

  The next few sets of features are based on TF-IDF and SVD. Term Frequency-Inverse Document Frequency (TF-IDF). Is one of the algorithms at the foundation of information retrieval. Here, the algorithm is explained using a formula:

$$TF(t)=C(t)/N$$

$$IDF(t)=log(ND/ND_t)$$

You can understand the formula using this notation: C(t) is the number of times a term t appears in a document, N is the total number of terms in the document, this results in the Term Frequency (TF).  ND is the total number of documents and NDt is the number of documents containing the term t, this provides the Inverse Document Frequency (IDF).  TF-IDF for a term t is a multiplication of Term Frequency and Inverse Document Frequency for the given term t:

$$TFIDF(t)=TF(t)*IDF(t)$$

Without any prior knowledge, other than about the documents themselves, such a score will highlight all the terms that could easily discriminate a document from the others, down-weighting the common words that won't tell you much, such as the common parts of speech (such as articles, for instance).

> If you need a more hands-on explanation of TFIDF, this great online tutorial will help you try coding the algorithm yourself and testing it on some text data: https://stevenloria.com/tf-idf/

For convenience and speed of execution, we resorted to the scikit-learn implementation of TFIDF.  If you don't already have scikit-learn installed, you can install it using pip: pip install -U scikit-learn
We create TFIDF features for both question1 and question2 separately (in order to type less, we just deep copy the question1 TfidfVectorizer):

```python
from sklearn.feature_extraction.text import TfidfVectorizer 
from copy import deepcopy
tfv_q1 = TfidfVectorizer(min_df=3,
                         max_features=None, 
                         strip_accents='unicode', 
                         analyzer='word',   
                         token_pattern=r'\w{1,}', 
                         ngram_range=(1, 2),    
                         use_idf=1,        
                         smooth_idf=1,  
                         sublinear_tf=1, 
                         stop_words='english') 
tfv_q2 = deepcopy(tfv_q1)
```


It must be noted that the parameters shown here have been selected after quite a lot of experiments. These parameters generally work pretty well with all other problems concerning natural language processing, specifically text classification. One might need to change the stop word list to the language in question.
We can now obtain the TFIDF matrices for question1 and question2 separately:

```python
q1_tfidf = tfv_q1.fit_transform(data.question1.fillna("")) 
q2_tfidf = tfv_q2.fit_transform(data.question2.fillna(""))
```


> In our TFIDF processing, we computed the TFIDF matrices based on all the data available (we used the fit_transform method). This is quite a common approach in Kaggle competitions because it helps to score higher on the leaderboard. However, if you are working in a real setting, you may want to exclude a part of the data as a training or validation set in order to be sure that your TFIDF processing helps your model to generalize to a new, unseen dataset. 

After we have the TFIDF features, we move to SVD features. SVD is a feature decomposition method and it stands for singular value decomposition. It is largely used in NLP because of a technique called Latent Semantic Analysis (LSA).

> A detailed discussion of SVD and LSA is beyond the scope of this chapter, but you can get an idea of their workings by trying these two approachable and clear online tutorials: https://alyssaq.github.io/2015/singular-value-decomposition-visualisation/ and 	https://technowiki.wordpress.com/2011/08/27/latent-semantic-analysis-lsa-tutorial/

To create the SVD features, we again use scikit-learn implementation. This implementation is a variation of traditional SVD and is known as TruncatedSVD.

> A TruncatedSVD is an approximate SVD method that can provide you with reliable yet computationally fast SVD matrix decomposition. You can find more hints about how this technique works and it can be applied by consulting this web page: http://langvillea.people.cofc.edu/DISSECTION-LAB/Emmie'sLSI-SVDModule/p5module.html 

```python
from sklearn.decomposition import TruncatedSVD 
svd_q1 = TruncatedSVD(n_components=180) 
svd_q2 = TruncatedSVD(n_components=180)
```

We chose 180 components for SVD decomposition and these features are calculated on a TFIDF matrix:
```python
question1_vectors = svd_q1.fit_transform(q1_tfidf) 
question2_vectors = svd_q2.fit_transform(q2_tfidf)
```
Feature set-3 is derived from a combination of these TF-IDF and SVD features. For example, we can have only the TF-IDF features for the two questions separately going into the model, or we can have the TF-IDF of the two questions combined with an SVD on top of them, and then the model kicks in, and so on. These features are explained as follows.
Feature set-3(1) or fs3_1 is created using two different TF-IDFs for the two questions, which are then stacked together horizontally and passed to a machine learning model:

![](figures\184_1.png)

This can be coded as:

```python
from scipy import sparse
# obtain features by stacking the sparse matrices together 
fs3_1 = sparse.hstack((q1_tfidf, q2_tfidf))
```
Feature set-3(2), or fs3_2, is created by combining the two questions and using a single TFIDF: 

![](figures\185_1.png)

```python
tfv = TfidfVectorizer(min_df=3, 
                      max_features=None, 
                      strip_accents='unicode',  
                      analyzer='word',
                      token_pattern=r'\w{1,}',
                      ngram_range=(1, 2),
                      use_idf=1,  
                      smooth_idf=1,  
                      sublinear_tf=1, 
                      stop_words='english')

# combine questions and calculate tf-idf 
q1q2 = data.question1.fillna("") 
q1q2 += " " + data.question2.fillna("")
fs3_2 = tfv.fit_transform(q1q2)
```

The next subset of features in this feature set, feature set-3(3) or fs3_3, consists of separate TF-IDFs and SVDs for both questions:

![](figures\185_2.png)

This can be coded as follows:
```python
# obtain features by stacking the matrices together 
fs3_3 = np.hstack((question1_vectors, question2_vectors))
```

We can similarly create a couple more combinations using TF-IDF and SVD, and call them fs3-4 and fs3-5, respectively. These are depicted in the following diagrams, but the code is left as an exercise for the reader.
Feature set-3(4) or fs3-4:

![](figures\186_1.png)

Feature set-3(5) or fs3-5:

![](figures\186_2.png)

After the basic feature set and some TF-IDF and SVD features, we can now move to more complicated features before diving into the machine learning and deep learning models. 

### 使用Word2vec词嵌入映射

Very broadly, Word2vec models are two-layer neural networks that take a text corpus as input and output a vector for every word in that corpus. After fitting, the words with similar meaning have their vectors close to each other, that is, the distance between them is small compared to the distance between the vectors for words that have very different meanings.
Nowadays, Word2vec has become a standard in natural language processing problems and often it provides very useful insights into information retrieval tasks. For this particular problem, we will be using the Google news vectors. This is a pretrained Word2vec model trained on the Google News corpus.
Every word, when represented by its Word2vec vector, gets a position in space, as depicted in the following diagram:

![](figures\187_1.png)

All the words in this example, such as Germany, Berlin, France, and Paris, can be represented by a 300-dimensional vector, if we are using the pretrained vectors from the Google news corpus. When we use Word2vec representations for these words and we subtract the vector of Germany from the vector of Berlin and add the vector of France to it, we will get a vector that is very similar to the vector of Paris. The Word2vec model thus carries the meaning of words in the vectors. The information carried by these vectors constitutes a very useful feature for our task. 
> For a user-friendly, yet more in-depth, explanation and description of possible applications of Word2vec, we suggest reading https://www.distilled.net/resources/a-beginners-guide-to-Word2vec-aka-whats-the-opposite-of-canada/, or if you need a more mathematically defined explanation, we recommend reading this paper: http://www.1-4-5.net/~dmm/ml/how_does_Word2vec_work.pdf

To load the Word2vec features, we will be using Gensim. If you don't have Gensim, you can install it easily using pip. At this time, it is suggested you also install the pyemd package, which will be used by the WMD distance function, a function that will help us to relate two Word2vec vectors:
```
pip install gensim 
pip install pyemd
```


To load the Word2vec model, we download the GoogleNews-vectorsnegative300.bin.gz binary and use Gensim's load_Word2vec_format function to load it into memory. You can easily download the binary from an Amazon AWS repository using the wget command from a shell:
```
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors	negative300.bin.gz"
```


After downloading and decompressing the file, you can use it with the Gensim KeyedVectors functions: 

```python
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin.gz', binary=True)
```


Now, we can easily get the vector of a word by calling model[word]. However, a problem arises when we are dealing with sentences instead of individual words. In our case, we need vectors for all of question1 and question2 in order to come up with some kind of comparison. For this, we can use the following code snippet. The snippet basically adds the vectors for all words in a sentence that are available in the Google news vectors and gives a normalized vector at the end. We can call this sentence to vector, or Sent2Vec.
Make sure that you have Natural Language Tool Kit (NLTK) installed before running the preceding function:
```
$ pip install nltk
```


It is also suggested that you download the punkt and stopwords packages, as they are part of NLTK:
```python
import nltk 
nltk.download('punkt') 
nltk.download('stopwords')
```


If NLTK is now available, you just have to run the following snippet and define the sent2vec function:
```python
from nltk.corpus import stopwords 
from nltk import word_tokenize 
stop_words = set(stopwords.words('english'))
def sent2vec(s, model):
    M = []
    words = word_tokenize(str(s).lower())     
    for word in words:        
        #It shouldn't be a stopword      
        if word not in stop_words:    
            #nor contain numbers    
            if word.isalpha():     
                #and be part of word2vec    
                if word in model:                    
                    M.append(model[word])   
    M = np.array(M)    
    if len(M) > 0:      
        v = M.sum(axis=0)
        return v / np.sqrt((v ** 2).sum())     
    else:        
        return np.zeros(300)
```


When the phrase is null, we arbitrarily decide to give back a standard vector of zero values.
To calculate the similarity between the questions, another feature that we created was word mover's distance. Word mover's distance uses Word2vec embeddings and works on a principle similar to that of earth mover's distance to give a distance between two text documents. Simply put, word mover's distance provides the minimum distance needed to move all the words from one document to an other document. 
> The WMD has been introduced by this paper: KUSNER, Matt, et al. From word embeddings to document distances. In: International Conference on Machine Learning. 2015. p. 957-966 which can be found at http://proceedings.mlr.press/v37/kusnerb15.pdf. For a hands-on tutorial on the distance, you can also refer to this tutorial based on the Gensim implementation of the distance: https://markroxor.github.io/gensim/static/notebooks/WMD_tutorial.html

Final Word2vec (w2v) features also include other distances, more usual ones such as the Euclidean or cosine distance. We complete the sequence of features with some measurement of the distribution of the two document vectors:
1. Word mover distance

2. Normalized word mover distance

3. Cosine distance between vectors of question1 and question2

4. Manhattan distance between vectors of question1 and question2

5. Jaccard similarity between vectors of question1 and question2

6. Canberra distance between vectors of question1 and question2

7. Euclidean distance between vectors of question1 and question2

8. Minkowski distance between vectors of question1 and question2

9. Braycurtis distance between vectors of question1 and question2

10. The skew of the vector for question1

11. The skew of the vector for question2

12. The kurtosis of the vector for question1

13. The kurtosis of the vector for question2

   
All the Word2vec features are denoted by fs4.

A separate set of w2v features consists in the matrices of Word2vec vectors themselves:

1.    Word2vec vector for question1

2.    Word2vec vector for question2

   These will be represented by fs5:
   

```python
w2v_q1 = np.array([sent2vec(q, model) 
                   for q in data.question1]) 
w2v_q2 = np.array([sent2vec(q, model) 
                   for q in data.question2])
```


   In order to easily implement all the different distance measures between the vectors of the Word2vec embeddings of the Quora questions, we use the implementations found in the scipy.spatial.distance module:

```python
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
data['cosine_distance'] = [cosine(x,y)
                           for (x,y) in zip(w2v_q1, w2v_q2)]
data['cityblock_distance'] = [cityblock(x,y)
                              for (x,y) in zip(w2v_q1, w2v_q2)]
data['jaccard_distance'] = [jaccard(x,y)
                            for (x,y) in zip(w2v_q1, w2v_q2)]
data['canberra_distance'] = [canberra(x,y)
                             for (x,y) in zip(w2v_q1, w2v_q2)]
data['euclidean_distance'] = [euclidean(x,y)
                              for (x,y) in zip(w2v_q1, w2v_q2)]
data['minkowski_distance'] = [minkowski(x,y,3)
                              for (x,y) in zip(w2v_q1, w2v_q2)]
data['braycurtis_distance'] = [braycurtis(x,y)            
                               for (x,y) in zip(w2v_q1, w2v_q2)]
```


   All the features names related to distances are gathered under the list fs4_1:


```python
fs4_1 = ['cosine_distance', 'cityblock_distance','jaccard_distance', 
         'canberra_distance','euclidean_distance', 'minkowski_distance',
         'braycurtis_distance']
```


   The Word2vec matrices for the two questions are instead horizontally stacked and stored away in the w2v variable for later usage: 

```python
w2v = np.hstack((w2v_q1, w2v_q2))
```


   The Word Mover's Distance is implemented using a function that returns the distance between two questions, after having transformed them into lowercase and after removing any stopwords. Moreover, we also calculate a normalized version of the distance, after transforming all the Word2vec vectors into L2-normalized vectors (each vector is transformed to the unit norm, that is, if we squared each element in the vector and summed all of them, the result would be equal to one) using the init_sims method:


```python
def wmd(s1, s2, model):     
    s1 = str(s1).lower().split()   
    s2 = str(s2).lower().split()   
    stop_words = stopwords.words('english')  
    s1 = [w for w in s1 if w not in stop_words]  
    s2 = [w for w in s2 if w not in stop_words]  
    return model.wmdistance(s1, s2)
data['wmd'] = data.apply(lambda x: wmd(x['question1'], 
                                       x['question2'], model), axis=1)
model.init_sims(replace=True)
data['norm_wmd'] = data.apply(lambda x: wmd(x['question1'], 
                                            x['question2'], model), axis=1) 
fs4_2 = ['wmd', 'norm_wmd']
```


   After these last computations, we now have most of the important features that are needed to create some basic machine learning models, which will serve as a benchmark for our deep learning models. The following table displays a snapshot of the available features:

![](figures\192_1.png)

Let's train some machine learning models on these and other Word2vec based features.

### 测试机器学习模型

Before proceeding, depending on your system, you may need to clean up the memory a bit and free space for machine learning models from previously used data structures. This is done using gc.collect, after deleting any past variables not required anymore, and then checking the available memory by exact reporting from the psutil.virtualmemory function:

```python
import gc
import psutil
del([tfv_q1, tfv_q2, tfv, q1q2,
     question1_vectors, question2_vectors, svd_q1,
     svd_q2, q1_tfidf, q2_tfidf])
del([w2v_q1, w2v_q2]) del([model]) 
gc.collect() 
psutil.virtual_memory()
```


At this point, we simply recap the different features created up to now, and their meaning in terms of generated features:

- fs_1: List of basic features 
- fs_2: List of fuzzy features 
- fs3_1: Sparse data matrix of TFIDF for separated questions 
- fs3_2: Sparse data matrix of TFIDF for combined questions
-  fs3_3: Sparse data matrix of SVD 
- fs3_4: List of SVD statistics 
- fs4_1: List of w2vec distances 
- fs4_2: List of wmd distances
- w2v: A matrix of transformed phrase's Word2vec vectors by means of the Sent2Vec function

We evaluate two basic and very popular models in machine learning, namely logistic regression and gradient boosting using the xgboost package in Python. The following table provides the performance of the logistic regression and xgboost algorithms on different sets of features created earlier, as obtained during the Kaggle competition:

|      |      |      |      |      |      |
| ---- | ---- | ---- | ---- | ---- | ---- |
|      |      |      |      |      |      |
|      |      |      |      |      |      |
|      |      |      |      |      |      |
|      |      |      |      |      |      |
|      |      |      |      |      |      |
|      |      |      |      |      |      |
|      |      |      |      |      |      |
|      |      |      |      |      |      |
|      |      |      |      |      |      |
|      |      |      |      |      |      |

Feature set	Logistic regression accuracy	xgboost accuracy
Basic features (fs1)	0.658	0.721
Basic features + fuzzy features (fs1 + fs2)	0.660	0.738
Basic features + fuzzy features + w2v features (fs1 + fs2 + fs4)	0.676	0.766
W2v vector features (fs5)	*	0.78
Basic features + fuzzy features + w2v features + w2v vector features (fs1 + fs2 + fs4 + fs5)	*	0.814
TFIDF-SVD features (fs3-1)	0.777	0.749
TFIDF-SVD features (fs3-2)	0.804	0.748
TFIDF-SVD features (fs3-3)	0.706	0.763
TFIDF-SVD features (fs3-4)	0.700	0.753
TFIDF-SVD features (fs3-5)	0.714	0.759

\* These models were not trained due to high memory requirements.

We can treat the performances achieved as benchmarks or baseline numbers before starting with deep learning models, but we won't limit ourselves to that and we will be trying to replicate some of them.
We are going to start by importing all the necessary packages. As for as the logistic regression, we will be using the scikit-learn implementation.
The xgboost is a scalable, portable, and distributed gradient boosting library (a tree ensemble machine learning algorithm). Initially created by Tianqi Chen from Washington University, it has been enriched with a Python wrapper by Bing Xu, and an R interface by Tong He (you can read the story behind xgboost directly from its principal creator at homes.cs.washington.edu/~tqchen/2016/03/10/story-and-lessons-behind-theevolution-of-xgboost.html ). The xgboost is available for Python, R, Java, Scala, Julia, and C++, and it can work both on a single machine (leveraging multithreading) and in Hadoop and Spark clusters.

> Detailed instruction for installing xgboost on your system can be found on this page: github.com/dmlc/xgboost/blob/master/doc/build.md
> The installation of xgboost on both Linux and macOS is quite straightforward, whereas it is a little bit trickier for Windows users.
> For this reason, we provide specific installation steps for having xgboost working on Windows:
>
> 1. First, download and install Git for Windows (git-forwindows.github.io)
> 2. Then, you need a MINGW compiler present on your system. You can download it from www.mingw.org according to the characteristics of your system
> 3. From the command line, execute:
>
> ```
> $> git clone --recursive https://github.com/dmlc/xgboost 
> $> cd xgboost
> $> git submodule init
> $> git submodule update
> ```
>
> 4. Then, always from the command line, you copy the configuration for 64-byte systems to be the default one:
>
> ```
> $> copy make\mingw64.mk config.mk
> ```
>
> ​      Alternatively, you just copy the plain 32-byte version:
>
> ```
> $> copy make\mingw.mk config.mk
> ```
>
> 
>
> 5. After copying the configuration file, you can run the compiler, setting it to use four threads in order to speed up the compiling process:
>
> ```
> $> mingw32-make -j4
> ```
>
> 6. In MinGW, the make command comes with the name mingw32make; if you are using a different compiler, the previous command may not work, but you can simply try:
>
> ```
> $> make -j4
> ```
>
> 7. Finally, if the compiler completed its work without errors, you can install the package in Python with:
>
> ```
> $> cd python-package
> $> python setup.py install
> ```


If xgboost has also been properly installed on your system, you can proceed with importing both machine learning algorithms:
```python
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler 
import xgboost as xgb
```


Since we will be using a logistic regression solver that is sensitive to the scale of the features
it is the sag solver from https:// github	.com	/EpistasisLab	/tpot	/issues	/29 2, which requires a linear computational time in respect to the size of the data), we will start by standardizing the data using the scaler function in scikit-learn:

```python
scaler = StandardScaler() y = data.is_duplicate.values y = y.astype('float32').reshape(-1, 1) 
X = data[fs_1+fs_2+fs3_4+fs4_1+fs4_2]
X = X.replace([np.inf, -np.inf], np.nan).fillna(0).values
X = scaler.fit_transform(X)
X = np.hstack((X, fs3_3))
```



We also select the data for the training by first filtering the fs_1, fs_2, fs3_4, fs4_1, and fs4_2 set of variables, and then stacking the fs3_3 sparse SVD data matrix. We also provide a random split, separating 1/10 of the data for validation purposes (in order to effectively assess the quality of the created model):

```python
np.random.seed(42) 
n_all, _ = y.shape 
idx = np.arange(n_all) 
np.random.shuffle(idx) 
n_split = n_all // 10 
idx_val = idx[:n_split] 
idx_train = idx[n_split:] 
x_train = X[idx_train] 
y_train = np.ravel(y[idx_train])
x_val = X[idx_val] 
y_val = np.ravel(y[idx_val])
```



As a first model, we try logistic regression, setting the regularization l2 parameter C to 0.1
(modest regularization). Once the model is ready, we test its efficacy on the validation set (x_val for the training matrix, y_val for the correct answers). The results are assessed on accuracy, that is the proportion of exact guesses on the validation set:

```python
logres = linear_model.LogisticRegression(C=0.1,
                                         solver='sag', max_iter=1000)
logres.fit(x_train, y_train) 
lr_preds = logres.predict(x_val)
log_res_accuracy = np.sum(lr_preds == y_val) / len(y_val)
print("Logistic regr accuracy: %0.3f" % log_res_accuracy)
```


After a while (the solver has a maximum of 1,000 iterations before giving up converging the results), the resulting accuracy on the validation set will be 0.743, which will be our starting
baseline.
Now, we try to predict using the xgboost algorithm. Being a gradient boosting algorithm, this learning algorithm has more variance (ability to fit complex predictive functions, but also to overfit) than a simple logistic regression afflicted by greater bias (in the end, it is a summation of coefficients) and so we expect much better results. We fix the max depth of its decision trees to 4 (a shallow number, which should prevent overfitting) and we use an eta of 0.02 (it will need to grow many trees because the learning is a bit slow). We also set up a watchlist, keeping an eye on the validation set for an early stop if the expected error on the validation doesn't decrease for over 50 steps.

> It is not best practice to stop early on the same set (the validation set in our case) we use for reporting the final results. In a real-world setting, ideally, we should set up a validation set for tuning operations, such as early stopping, and a test set for reporting the expected results when generalizing to new data.

After setting all this, we run the algorithm. This time, we will have to wait for longer than we when running the logistic regression:

```python
params = dict()
params['objective'] = 'binary:logistic' 
params['eval_metric'] = ['logloss', 'error']
params['eta'] = 0.02 
params['max_depth'] = 4
d_train = xgb.DMatrix(x_train, label=y_train) 
d_valid = xgb.DMatrix(x_val, label=y_val) 
watchlist = [(d_train, 'train'), (d_valid, 'valid')] 
bst = xgb.train(params, d_train, 5000, watchlist, 
                early_stopping_rounds=50, verbose_eval=100) 
xgb_preds = (bst.predict(d_valid) >= 0.5).astype(int) 
xgb_accuracy = np.sum(xgb_preds == y_val) / len(y_val) 
print("Xgb accuracy: %0.3f" % xgb_accuracy)
```


The final result reported by xgboost is 0.803 accuracy on the validation set.

###  搭建TensorFlow模型

he deep learning models in this chapter are built using TensorFlow, based on the original script written by Abhishek Thakur using Keras (you can read the original code at https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question). Keras is a Python library that provides an easy interface to TensorFlow. Tensorflow has official support for Keras, and the models trained using Keras can easily be converted to TensorFlow models. Keras enables the very fast prototyping and testing of deep learning models. In our project, we rewrote the solution entirely in TensorFlow from scratch anyway.
To start, let's import the necessary libraries, in particular TensorFlow, and let's check its version by printing it:

```python
import zipfile
from tqdm import tqdm_notebook as tqdm
import tensorflow as tf
print("TensorFlow version %s" % tf.__version__)
```


At this point, we simply load the data into the df pandas dataframe or we load it from disk. We replace the missing values with an empty string and we set the y variable containing the target answer encoded as 1 (duplicated) or 0 (not duplicated):

```python
try:
    df = data[['question1', 'question2', 'is_duplicate']] 
except:     
    df = pd.read_csv('data/quora_duplicate_questions.tsv', sep='\t')
df = df.drop(['id', 'qid1', 'qid2'], axis=1)
df = df.fillna('') y = df.is_duplicate.values
y = y.astype('float32').reshape(-1, 1)
```


We can now dive into deep neural network models for this dataset.

### 深度神经网络之前的处理

Before feeding data into any neural network, we must first tokenize the data and then convert the data to sequences. For this purpose, we use the Keras Tokenizer provided with TensorFlow, setting it using a maximum number of words limit of 200,000 and a maximum sequence length of 40. Any sentence with more than 40 words is consequently cut off to its first 40 words:

```python
Tokenizer = tf.keras.preprocessing.text.Tokenizer 
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences 
tk = Tokenizer(num_words=200000) max_len = 40
```


After setting the Tokenizer, tk, this is fitted on the concatenated list of the first and second questions, thus learning all the possible word terms present in the learning corpus:

```python
tk.fit_on_texts(list(df.question1) + list(df.question2))
x1 = tk.texts_to_sequences(df.question1) 
x1 = pad_sequences(x1, maxlen=max_len) 
x2 = tk.texts_to_sequences(df.question2) 
x2 = pad_sequences(x2, maxlen=max_len) 
word_index = tk.word_index
```


In order to keep track of the work of the tokenizer, word_index is a dictionary containing all the tokenized words paired with an index assigned to them.
Using the GloVe embeddings, we must load them into memory, as previously seen when discussing how to get the Word2vec embeddings.
The GloVe embeddings can be easily recovered using this command from a shell: 

```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
```


he GloVe embeddings are similar to Word2vec in the sense that they encode words into a complex multidimensional space based on their co-occurrence. However, as explained by the paper http://clic.cimec.unitn.it/marco/publications/acl2014/baroni-etal-countpredict-acl2014.pdf  —BARONI, Marco; DINU, Georgiana; KRUSZEWSKI, Germán. Don't count, predict! A systematic comparison of context-counting vs. context-predicting semantic vectors. In: Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2014. p. 238-247.
GloVe is not derived from a neural network optimization that strives to predict a word from its context, as Word2vec is. Instead, GloVe is generated starting from a co-occurrence count matrix (where we count how many times a word in a row co-occurs with the words in the columns) that underwent a dimensionality reduction (a factorization just like SVD, as we mentioned before when preparing our data). 

> hy are we now using GloVe instead of Word2vec? In practice, the main difference between the two simply boils down to the empirical fact that GloVe embeddings work better on some problems, whereas Word2vec embeddings perform better on others. In our case, after experimenting, we found GloVe embeddings working better with deep learning algorithms. You can read more information about GloVe and its uses from its official page at Stanford University: https://nlp.stanford.edu/projects/glove/

Having got a hold of the GloVe embeddings, we can now proceed to create an embedding_matrix by filling the rows of the embedding_matrix array with the embedding vectors (sized at 300 elements each) extracted from the GloVe file.
The following code reads the glove embeddings file and stores them into our embedding matrix, which in the end will consist of all the tokenized words in the dataset with their respective vectors:

```python
embedding_matrix = np.zeros((len(word_index) + 1, 300), dtype='float32')
glove_zip = zipfile.ZipFile('data/glove.840B.300d.zip') 
glove_file = glove_zip.filelist[0]
f_in = glove_zip.open(glove_file) 
for line in tqdm(f_in):
    values = line.split(b' ')    
    word = values[0].decode()     
    if word not in word_index:       
        continue     
    i = word_index[word]
    coefs = np.asarray(values[1:], dtype='float32')     
    embedding_matrix[i, :] = coefs
    
f_in.close() 
glove_zip.close()
```


Starting from an empty embedding_matrix, each row vector is placed on the precise row number of the matrix that is expected to represent its corresponding wording. Such correspondence between words and rows has previously been defined by the encoding process completed by the tokenizer and is now available for consultation in the word_index dictionary. 
After the embedding_matrix has completed loading the embeddings, it is time to start building some deep learning models.

### 深度神经网络搭建模块

In this section, we are going to present the key functions that will allow our deep learning project to work. Starting from batch feeding (providing chunks of data to learn to the deep neural network) we will prepare the building blocks of a complex LSTM architecture. 
The LSTM architecture is presented in a hands-on and detailed way in Chapter 7, Stock Price Prediction with LSTM, inside the Long short-term memory – LSTM 101 section
The first function we start working with is the prepare_batches one. This function takes the question sequences and based on a step value (the batch size), returns a list of lists, where the internal lists are the sequence batches to be learned:

```python
def prepare_batches(seq, step):    
    n = len(seq)     
    res = []     
    for i in range(0, n, step):  
        res.append(seq[i:i+step])
    return res
```


The dense function will create a dense layer of neurons based on the provided size and activate and initialize them with random normally distributed numbers that have a mean of zero, and as a standard deviation, the square root of 2 divided by the number of input features.
A proper initialization helps back-propagating the input derivative deep inside the network. In fact:

- If you initialize the weights in a network too small, then the derivative shrinks as it passes through each layer until it's too faint to trigger the activation functions.
-  If the weights in a network are initialized too large, then the derivative simply grows (the so-called exploding gradient problem) as it traverses through each layer, the network won't converge to a proper solution and it will break because of handling numbers that are too large.

The initialization procedure makes sure the weights are just right by setting a reasonable starting point where the derivative can propagate through many layers. There are quite a few initialization procedures for deep learning networks, such as Xavier by Glorot and Bengio (Xavier is Glorot's first name), and the one proposed by He, Rang, Zhen, and Sun, and built on the Glorot and Bengio one, which is commonly referred to as He.

> eight initialization is quite a technical aspect of building a neural network architecture, yet a relevant one. If you want to know more about it, you can start by consulting this post, which also delves into more mathematical explanations of the topic: http://deepdish.io/2015/02/24/network-initialization/ 

n this project, we opted for the He initialization, since it works quite well for rectified units. Rectified units, or ReLu, are the powerhouse of deep learning because they allow signals to propagate and avoid the exploding or vanishing gradient problems, yet neurons activated by the ReLU,  from a practical point of view, are actually most of the time just firing a zero value. Keeping the variance large enough in order to have a constant variance of the input and output gradient passing through the layer really helps this kind of activation to work best, as explained in this paper: HE, Kaiming, et al. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In: Proceedings of the IEEE international conference on computer vision. 2015. p. 1026-1034 which can be found and read at https:// arxiv.org /abs	/1502	.0185	2:

```python
def dense(X, size, activation=None):
he_std = np.sqrt(2 / int(X.shape[1]))     
out = tf.layers.dense(X, units=size,           
                      activation=activation,  
                      kernel_initializer=\ 
                      tf.random_normal_initializer(stddev=he_std))     
return out
```


Next, we work on another kind of layer, the time distributed dense layer.
This kind of layer is used on recurrent neural networks in order to keep a one-to-one relationship between the input and the output. An RNN (with a certain number of cells providing channel outputs), fed by a standard dense layer, receives matrices whose dimensions are rows (examples) by columns (sequences) and it produces as output a matrix whose dimensions are rows by the number of channels (cells). If you feed it using the time distributed dense layer, its output will instead be dimensionality shaped as rows by columns by channels. In fact, it happens that a dense neural network is applied to timestamp (each column).
A time distributed dense layer is commonly used when you have, for instance, a sequence of inputs and you want to label each one of them, taking into account the sequence that arrived. This is a common scenario for tagging tasks, such as multilabel classification or Part-Of-Speech tagging. In our project, we will be using it just after the GloVe embedding in order to process how each GloVe vector changes by passing from a word to another in the question sequence.
As an example, let's say you have a sequence of two cases (a couple of question examples), and each one has three sequences (some words), each of which is made of four elements (their embeddings). If we have such a dataset passed through the time distributed dense layer with five hidden units, we will obtain a tensor of size (2, 3, 5). In fact, passing through the time distributed layer, each example retains the sequences, but the embeddings are replaced by the result of the five hidden units. Passing them through a reduction on the 1 axis, we will simply have a tensor of size (2,5), that is a result vector for each since example.

> If you want to replicate the previous example:
>
> ```python
> print("Tensor's shape:", X.shape)
> tensor = tf.convert_to_tensor(X, dtype=tf.float32)
> dense_size = 5
> i = time_distributed_dense(tensor, dense_size)print("Shape of time distributed output:", i)
> j = tf.reduce_sum(i, axis=1) 
> print("Shape of reduced output:", j)
> ```



> he concept of a time distributed dense layer could be a bit trickier to grasp than others and there is much discussion online about it. You can also read this thread from the Keras issues to get more insight into the topic: https://github.com/keras-team/keras/issues/1029

```python
def time_distributed_dense(X, dense_size):     
    shape = X.shape.as_list()     
    assert len(shape) == 3
    _, w, d = shape
    X_reshaped = tf.reshape(X, [-1, d])
    H = dense(X_reshaped, dense_size,
              tf.nn.relu)     
    return tf.reshape(H, [-1, w, dense_size])
```


he conv1d and maxpool1d_global functions are in the end wrappers of the TensorFlow functions tf.layers.conv1d (https://www.tensorflow.org/api_docs/python/tf/layers/conv1d), which is a convolution layer, and tf.reduce_max (https://www.tensorflow.org/api_docs/python/tf/reduce_max), which computes the maximum value of elements across the dimensions of an input tensor. In natural language processing, this kind of pooling (called global max pooling) is more frequently used than the standard max pooling that is commonly found in deep learning applications for computer vision. As explained by a Q&A on cross-validated (https://stats.stackexchange.com/a/257325/49130) global max pooling simply takes the maximum value of an input vector, whereas standard max pooling returns a new vector made of the maximum values found in different pools of the input vector given a certain pool size:

```python
def conv1d(inputs, num_filters, filter_size, padding='same'):     
    he_std = np.sqrt(2 / (filter_size * num_filters))
    out = tf.layers.conv1d(
    inputs=inputs, filters=num_filters, padding=padding,
    kernel_size=filter_size, 
    activation=tf.nn.relu,
    kernel_initializer=tf.random_normal_initializer(stddev=he_std))     
return out
def maxpool1d_global(X):     
    out = tf.reduce_max(X, axis=1)
return out
```


ur core lstm function is initialized by a different scope at every run due to a random integer number generator, initialized by He initialization (as seen before), and it is a wrapper of the TensorFlow tf.contrib.rnn.BasicLSTMCell for the layer of Basic LSTM recurrent network cells (https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell) and tf.contrib.rnn.static_rnn for creating a recurrent neural network specified by the layer of cells (https://www.tensorflow.org/versions/r1.1/api_docs/python/tf/contrib/rnn/static_rnn).

> he implementation of the Basic LSTM recurrent network cells is based on the paper ZAREMBA, Wojciech; SUTSKEVER, Ilya; VINYALS, Oriol. Recurrent neural network regularization. arXiv preprint arXiv:1409.2329, 2014 found at https://arxiv.org/abs/1409.2329.

```python
def lstm(X, size_hidden, size_out):     
    with tf.variable_scope('lstm_%d'
                           % np.random.randint(0, 100)):         
        he_std = np.sqrt(2 / (size_hidden * size_out))
        W = tf.Variable(tf.random_normal([size_hidden, size_out], 
                                         stddev=he_std))
        b = tf.Variable(tf.zeros([size_out]))
        size_time = int(X.shape[1])
        X = tf.unstack(X, size_time, axis=1)
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(size_hidden,
                                                 forget_bias=1.0) 
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, X, 
                                                    dtype='float32')
        out = tf.matmul(outputs[-1], W) + b        
        return out
```


At this stage of our project, we have gathered all the building blocks necessary to define the architecture of the neural network that will be learning to distinguish duplicated questions.

### 设计学习架构

We start defining our architecture by fixing some parameters such as the number of features considered by the GloVe embeddings, the number and length of filters, the length of maxpools, and the learning rate:

```python
max_features = 200000 
filter_length = 5 
nb_filter = 64 
pool_length = 4 
learning_rate = 0.001
```


Managing to grasp the different semantic meanings of less or more different phrases in order to spot possible duplicated questions is indeed a hard task that requires a complex architecture. For this purpose, after various experimentation, we create a deeper model consisting of LSTM, time-distributed dense layers, and 1d-cnn. Such a model has six heads, which are merged into one by concatenation. After concatenation, the architecture is completed by five dense layers and an output layer with sigmoid activation. 
The full model is shown in the following diagram:

![](figures\205_1.png)

The first head consists of an embedding layer initialized by GloVe embeddings, followed by a time-distributed dense layer. The second head consists of 1D convolutional layers on top of embeddings initialized by the GloVe model, and the third head is an LSTM model on the embeddings learned from scratch. The other three heads follow the same pattern for the other question in the pair of questions.
We start defining the six models and concatenating them. In the end, the models are merged by concatenation, that is, the vectors from the six models are stacked together horizontally.
Even if the following code chunk is quite long, following it is straightforward. Everything starts at the three input placeholders, place_q1, place_q2, and place_y, which feed all six models with the first questions, the second questions, and the target response respectively. The questions are embedded using GloVe (q1_glove_lookup and q2_glove_lookup) and a random uniform embedding. Both embeddings have 300 dimensions.
The first two models, model_1 and model_2, acquire the GloVe embeddings and they apply a time distributed dense layer.
The following two models, model_3 and model_4, acquire the GloVe embeddings and process them by a series of convolutions, dropouts, and maxpools. The final output vector is batch normalized in order to keep stable variance between the produced batches.

> If you want to know about the nuts and bolts of batch normalization, this Quora answer by Abhishek Shivkumar clearly provides all the key points you need to know about what batch normalization is and why it is effective in neural network architecture: https://www.quora.com/In-layman%E2%80%99s-terms-what-is-batch-normalisation-what-does-it-do-and-why-does-it-work-so-well/answer/Abhishek-Shivkumar

Finally, model_5 and model_6 acquire the uniform random embedding and process it with an LSTM. The results of all six models are concatenated together and batch normalized:

```python
graph = tf.Graph() 
graph.seed = 1

with graph.as_default():    
    place_q1 = tf.placeholder(tf.int32, shape=(None, max_len))    
    place_q2 = tf.placeholder(tf.int32, shape=(None, max_len))    
    place_y = tf.placeholder(tf.float32, shape=(None, 1))   
    place_training = tf.placeholder(tf.bool, shape=())   
    
    glove = tf.Variable(embedding_matrix, trainable=False)  
    q1_glove_lookup = tf.nn.embedding_lookup(glove, place_q1)     
    q2_glove_lookup = tf.nn.embedding_lookup(glove, place_q2)
    emb_size = len(word_index) + 1
    
    emb_dim = 300
    emb_std = np.sqrt(2 / emb_dim)
    emb = tf.Variable(tf.random_uniform([emb_size, emb_dim],
                                        -emb_std, emb_std))     
    q1_emb_lookup = tf.nn.embedding_lookup(emb, place_q1)     
    q2_emb_lookup = tf.nn.embedding_lookup(emb, place_q2)
    model1 = q1_glove_lookup
    model1 = time_distributed_dense(model1, 300)     
    model1 = tf.reduce_sum(model1, axis=1)
    
    model2 = q2_glove_lookup
    model2 = time_distributed_dense(model2, 300)     
    model2 = tf.reduce_sum(model2, axis=1)
    
    model3 = q1_glove_lookup
    model3 = conv1d(model3, nb_filter, filter_length, padding='valid')     
    model3 = tf.layers.dropout(model3, rate=0.2, training=place_training)
    model3 = conv1d(model3, nb_filter, filter_length, padding='valid')
    model3 = maxpool1d_global(model3)     
    model3 = tf.layers.dropout(model3, rate=0.2, training=place_training)     
    model3 = dense(model3, 300)
    model3 = tf.layers.dropout(model3, rate=0.2, training=place_training)
    model3 = tf.layers.batch_normalization(model3, training=place_training)
    
    model4 = q2_glove_lookup
    model4 = conv1d(model4, nb_filter, filter_length, padding='valid')
    model4 = tf.layers.dropout(model4, rate=0.2, training=place_training)
    model4 = conv1d(model4, nb_filter, filter_length, padding='valid')
    model4 = maxpool1d_global(model4)     
    model4 = tf.layers.dropout(model4, rate=0.2, training=place_training)     
    model4 = dense(model4, 300)
    model4 = tf.layers.dropout(model4, rate=0.2, training=place_training)
    model4 = tf.layers.batch_normalization(model4, training=place_training)
    model5 = q1_emb_lookup
    model5 = tf.layers.dropout(model5, rate=0.2, training=place_training)
    model5 = lstm(model5, size_hidden=300, size_out=300)
    model6 = q2_emb_lookup
    model6 = tf.layers.dropout(model6, rate=0.2, training=place_training)
    model6 = lstm(model6, size_hidden=300, size_out=300)
    merged = tf.concat([model1, model2, model3, model4, model5, model6],
                       axis=1)
    merged = tf.layers.batch_normalization(merged, training=place_training)
```


We then complete the architecture by adding five dense layers with dropout and batch normalization. Then, there is an output layer with sigmoid activation. The model is optimized using an AdamOptimizer based on log-loss:
    

```python
for i in range(5):         
    merged = dense(merged, 300, activation=tf.nn.relu)      
    merged = tf.layers.dropout(merged, rate=0.2, training=place_training)
    merged = tf.layers.batch_normalization(merged, training=place_training)
merged = dense(merged, 1, activation=tf.nn.sigmoid)    
loss = tf.losses.log_loss(place_y, merged)
prediction = tf.round(merged)
accuracy = tf.reduce_mean(tf.cast(tf.equal(place_y, prediction), 'float32'))
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
# for batchnorm
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)     
with tf.control_dependencies(extra_update_ops):         
    step = opt.minimize(loss)    
init = tf.global_variables_initializer()
    
session = tf.Session(config=None, graph=graph)
session.run(init)
```


After defining the architecture, we initialize the sessions and we are ready for learning. As a good practice, we split the available data into a training part (9/10) and a testing one (1/10). Fixing a random seed allows replicability of the results:

```python
np.random.seed(1)
n_all, _ = y.shape 
idx = np.arange(n_all) 
np.random.shuffle(idx)

n_split = n_all // 10 
idx_val = idx[:n_split] 
idx_train = idx[n_split:]

x1_train = x1[idx_train] 
x2_train = x2[idx_train] 
y_train = y[idx_train]

x1_val = x1[idx_val] 
x2_val = x2[idx_val] 
y_val = y[idx_val]
```


If you run the following code snippet, the training will start and you will notice that the model accuracy increases with the increase in the number of epochs. However, the model will take a lot of time to train, depending on the number of batches you decide to iterate through. On an NVIDIA Titan X, the model takes over 300 seconds per epoch. As a good balance between obtained accuracy and training time, we opt for running 10 epochs:

```python
val_idx = np.arange(y_val.shape[0]) 
val_batches = prepare_batches(val_idx, 5000) 

no_epochs = 10

# see https://github.com/tqdm/tqdm/issues/481 tqdm.monitor_interval = 0

for i in range(no_epochs):
    np.random.seed(i)
    train_idx_shuffle = np.arange(y_train.shape[0])
    np.random.shuffle(train_idx_shuffle)     
    batches = prepare_batches(train_idx_shuffle, 384)
    progress = tqdm(total=len(batches))     
    for idx in batches:         
        feed_dict = {
            place_q1: x1_train[idx],    
            place_q2: x2_train[idx],      
            place_y: y_train[idx],      
            place_training: True,
        }
        _, acc, l = session.run([step, accuracy, loss], feed_dict)
        progress.update(1)
        progress.set_description('%.3f / %.3f' % (acc, l))
        
    y_pred = np.zeros_like(y_val)     
    for idx in val_batches:
        feed_dict = {  
            place_q1: x1_val[idx],     
            place_q2: x2_val[idx],   
            place_y: y_val[idx], 
            place_training: False,
        }
        y_pred[idx, :] = session.run(prediction, feed_dict)
    print('batch %02d, accuracy: %0.3f' % (i, np.mean(y_val == y_pred)))
```

Trained for 10 epochs, the model produces an accuracy of  82.5%. This is much higher than the benchmarks we had before. Of course, the model could be improved further by using better preprocessing and tokenization. More epochs (up to 200) could also help raise the accuracy a bit more. Stemming and lemmatization may also definitely help to get near the state-of-the-art accuracy of 88% reported by Quora on its blog. 
Having completed the training, we can use the in-memory session to test some question evaluations. We try with two questions about the duplicated questions on Quora, but the procedure works with any pair of questions you would like to test the algorithm on.

> As with many machine learning algorithms, this one depends on the distribution that it has learned. Questions completely different from the ones it has been trained on could prove difficult for the algorithm to guess.

```python
def convert_text(txt, tokenizer, padder):    
    x = tokenizer.texts_to_sequences(txt)    
    x = padder(x, maxlen=max_len)     
    return x

def evaluate_questions(a, b, tokenizer, padder, pred):    
    feed_dict = {
        place_q1: convert_text([a], tk, pad_sequences), 
        place_q2: convert_text([b], tk, pad_sequences),
        place_y: np.zeros((1,1)),
        place_training: False,
        }
    return session.run(pred, feed_dict)
isduplicated = lambda a, b: evaluate_questions(a, b, tk, pad_sequences, prediction)
a = "Why are there so many duplicated questions on Quora?" 
b = "Why do people ask similar questions on Quora multiple times?" 

print("Answer: %0.2f" % isduplicated(a, b))
```


After running the code, the answer should reveal that the questions are duplicated (answer:
1.0).

### 小结

In this chapter, we built a very deep neural network with the help of TensorFlow in order to detect duplicated questions from the Quora dataset. The project allowed us to discuss, revise, and practice plenty of different topics previously seen in other chapters: TF-IDF, SVD, classic machine learning algorithms,  Word2vec and GloVe embeddings, and LSTM models.
In the end, we obtained a model whose achieved accuracy is about 82.5%, a figure that is higher than traditional machine learning approaches and is also near other state-of-the-art deep learning solutions, as reported by the Quora blog.  
It should also be noted that the models and approaches discussed in this chapter can easily be applied to any semantic matching problem.

---