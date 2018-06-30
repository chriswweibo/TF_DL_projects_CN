# 第8章 检测重复的Quora问题

Quora (www.quora.com)是一个社区驱动的问答网站。用户可以在上边公开的或者匿名的提出问题和回答问题。2017年1月， Quora第一次发布了一个包含问题对的数据集，其中的问题有可能是重复的。重复问题对在语义上是类似的。或者说，尽管两个问题使用不同的词汇，但是传达了相同的意思。为了给用户提供更好的答案集合展示以便尽快找出需要的信息，Quora需要为每一个问题都准备一个页面。这个工程量是非常大。主持人机制对于避免网站上的重复内容是很有帮助的，但是一旦每天回答的问题增多以及历史存量问题的扩大，这种机制就不容易扩展了。这种情况下，基于**自然语言理解（Natural Language Processing，NLP）**和深度学习的自动化项目就成了合适的方案。

本章会介绍如何构建基于TensorFlow的项目，以便理解Quora数据集中句子之间的相似性问题。本章的内容基于Abhishek Thakur （https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur/）的工作。他基于Keras库开发了一套方案 。给出的处理技术也可以用在其它有关语义相似性的问题。在这个项目中，我们会介绍： 

* 文本数据的特征工程Feature engineering on text data
* TF-IDF和SVD
* 基于特征的Word2vec和GloVe算法
* 传统的机器学习模型，例如logistic回归，和使用`xgboost`的梯度提升
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

查看数据之后，现在我们可以确定项目中要争取优化的目标。在这一章中，我们会使准确率作为度量来评估模型的性能。准确率关注于预测的有效程度，可能会丢失不同模型之间的重要差异，例如鉴别能力（模型检测重复问题的能力更强）或者概率分数的正确度（重复问题和非重复问题的有多大区别？）。 我们选择准确率是基于以下事实：这个度量是被Quora工程团队拿来确定数据集的基准表现（其博客提到了这一点： https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning）。使用准确率可以让我们的结果更易于和Quora工程团队的结果，以及其他科研文献的结果进行评估和比较。另外，在实际应用中，本书的工作可以基于预测正确错误与否进行快速评估，而不用涉及其他考量。

现在，我们首先介绍基础的特征工程。

### 基础特征工程

开始编码之前，我们需要使用Python加载数据集，同时给Python环境提供项目必须的程序库。我们需要给系统安装一些库（最新版的就可以满足需求，不需要指定具体的版本号：

* `Numpy` 
* `pandas `
* `fuzzywuzzy `
* `python-Levenshtein `
* `scikit-learn `
* `gensim`
* ` pyemd `
* `NLTK`

因为我们会在项目中使用到每一个库，所以我们会提供具体的安装说明和建议。

对于数据集操作，我们会使用`pandas` （ `Numpy` 也会使用）。要安装`numpy`和`pandas`：

```
pip install numpy 
pip install pandas
```
数据集可以通过使用`pandas`和具体的数据结构`pandas dataframe`加载到内存中（我们假设数据集和你的脚本或者Jupyter notebook位于同一个目录下）：
```python
import pandas as pd 
import numpy as np
data = pd.read_csv('quora_duplicate_questions.tsv', sep='\t') 
data = data.drop(['id', 'qid1', 'qid2'], axis=1)
```
我们会在本章中使用`data` 表示`pandas dataframe` 。使用TensorFlow模型的时候，我们也会给它提供输入。

首先，我们可以构造一些基本的特征。这些基础特征包括基于长度的特征和基于字符串的特征：

1. 问题1的长度

2. 问题2的长度

3. 两个长度的差异

4. 去除空格后，问题1的字符串长度

5. 去除空格后，问题2的字符串长度

6. 问题1的词数

7. 问题2的词数

8. 问题1和问题2中相同词的数量

这些特征都可以通过一行代码得到。使用Python中的`pandas库`和`apply`方法转换原始输入:

```python
# length based features
data['len_q1'] = data.question1.apply(lambda x: len(str(x))) 
data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
# difference in lengths of two questions 
data['diff_len'] = data.len_q1 - data.len_q2 
# character length based features 
data['len_char_q1'] = data.question1.apply(lambda x:
                  len(''.join(set(str(x).replace(' ', ''))))) 
data['len_char_q2'] = data.question2.apply(lambda x:                                             len(''.join(set(str(x).replace(' ', '')))))
# word length based features 
data['len_word_q1'] = data.question1.apply(lambda x:
                                         len(str(x).split())) 
data['len_word_q2'] = data.question2.apply(lambda x:                                                                    len(str(x).split()))
# common words in the two questions 
data['common_words'] = data.apply(lambda x: 
                                  len(set(str(x['question1'])
                                      .lower().split())
                                      .intersection(set(str(x['question2'])
                                      .lower().split()))), axis=1)
                        
```
为了后续引用方便，我们把这些特征标记为特征集-1或 `fs_1`:
```python
fs_1 = ['len_q1', 'len_q2', 'diff_len', 'len_char_q1', 'len_char_q2', 
        'len_word_q1', 'len_word_q2','common_words']
```
这个简单的方法可以便于后边机器学习模型中调用和合并不同特征集合，使得不同特征集合上的模型比较易于操作。

### 创建模糊特征

​        下一个特征集合基于模糊字符串匹配。模糊字符串匹配也叫作近似字符串匹配，是找出近似匹配给定模式的字符串的过程。匹配的完全程度定义为，把字符串转换为精确匹配的原始操作步数。这些原始操作包括插入（在给定位置插入一个字符），删除（删除特定字符）和替换（用新的字符替换旧的字符）。

模糊字符串匹配经常用在拼写检查，抄袭检测，DNA序列匹配，垃圾邮件过滤等方面。它是编辑距离大家族中一部分。这个距离是基于字符串转换为另一个字符串的思想。它经常用在自然语言处理，和其他应用中，以便断定不同字符串之间的差异程度。

它也叫做Levenshtein距离。它是由俄国科学家Vladimir Levenshtein于1965年发明的。


这些特征可以使用 Python的`fuzzywuzzy` 库（https://pypi.python.org/pypi/fuzzywuzzy）生成。这个库使用Levenshtein distance计算两个不同序列的差异，即对应本书项目中问题对的差异。

`fuzzywuzzy`库可以使用`pip3`安装： 

```
pip install fuzzywuzzy
```


​       `fuzzywuzzy`需要`Python-Levenshtein`库（https://github.com/ztane/python-Levenshtein/）作为依赖库，它是使用C代码构造，是经典算法的快速实现。因此要使用`fuzzywuzzy`做更快的计算,，我们还需要安装`PythonLevenshtein`库： 

```
pip install python-Levenshtein
```


`fuzzywuzzy`库提供了需要不同的比率，但是我们只会用到下面几种：

1. QRatio

2. WRatio

3. 部分比率

4. 部分令牌集比率

5. 部分令牌排序比率

6. 令牌集比率

7. 令牌排序比率

下面是Quora数据的 `fuzzywuzzy` 例子特征： 

```python
from fuzzywuzzy import fuzz
fuzz.QRatio("Why did Trump win the Presidency?", 
            "How did Donald Trump win the 2016 Presidential Election")
```


  这个代码片段会返回67：

```python
fuzz.QRatio("How can I start an online shopping (e-commerce) website?",
            "Which web technology is best suitable for building a big E-                   Commerce website?")
```


另外，这个返回值是60。我们注意到，尽管这些 `QRatio`值都很接近，但是数据集中相似问题对的值要比非相似问题对的值要高。让我们再看同一个问题对的另一个特征：


```python
fuzz.partial_ratio("Why did Trump win the Presidency?", 
                   "How did Donald Trump win the 2016 Presidential Election")
```


 这时，返回值是73：


```python
 fuzz.partial_ratio("How can I start an online shopping (e-commerce) website?", 
                    "Which web technology is best suitable for building a big                        E-Commerce website?")
```

  这时，返回值是57。

​    使用`partial_ratio`方法，我们可以观察到两个问题对的分数差异显著升高。这意味着重复问题对和非重复问题对之间的鉴别更加容易。我们认为这个特征可以给模型加分不少。

 借助Python 的`pandas` 和 `fuzzywuzzy` 库，我们可以通过一行代码，使用这些特征：

```python
data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(
    str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio( 
    str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_ratio'] = data.apply(lambda x:
                                        fuzz.partial_ratio(str(x['question1']),
                                        str(x['question2'])), axis=1)
data['fuzz_partial_token_set_ratio'] = data.apply(lambda x:                                                fuzz.partial_token_set_ratio(str(x['question1']),           str(x['question2'])), axis=1)
data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x:                   fuzz.partial_token_sort_ratio(str(x['question1']),str(x['question2'])), axis=1)
data['fuzz_token_set_ratio'] = data.apply(lambda x:                                    fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_token_sort_ratio'] = data.apply(lambda x:                               fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
```


  这个特征集合标记为特征集-2或`fs_2`：

```python
fs_2 = ['fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio',
       'fuzz_partial_token_set_ratio', 'fuzz_partial_token_sort_ratio',
       'fuzz_token_set_ratio', 'fuzz_token_sort_ratio']
```


  我们再一次把结果存储起来以便建模中使用。


### 借助TF-IDF和SVD特征

 接下来的特征集合基于TF-IDF和SVD。**词频逆文档频率（Term Frequency-Inverse Document Frequency ，TF-IDF）**是信息检索基础中的一个算法。这个算法可以使用下面的公式解释：

$$TF(t)=C(t)/N$$

$$IDF(t)=log(ND/ND_t)$$

公式中的记号： $C(t)$ 是词项$t$出现在文档中的次数，$N$是文档中词项的总数，可以由此计算**词频（Term Frequency ，TF）。** $ND$是文档总数，$ND_t $是包含词频$t$的文档数，可以由此计算**逆文档频率（Inverse Document Frequency ，IDF）。**  一个词项的TF-IDF是词频和逆文档频率的成绩：

$$TFIDF(t)=TF(t)*IDF(t)$$

除了文档本身之外，不用借助任何先验知识，通过减少信息较少的公共词项（例如冠词）的权重，这个分数就可以给出区分不同文档的所有词项。

> 如果需要更多关于TFIDF的实操解释，下面的在线教程可以帮助读者编码实现算法，并使用文本数据进行测试： https://stevenloria.com/tf-idf/

为了方便快速实现，我们借助于TFIDF的`scikit-learn实现`。如果还没有安装`scikit-learn`，读者可以使用`pip`安装：

```
pip install -U scikit-learn
```
我们可以分别为问题1和问题2构造TFIDF特征2（为了减少打字，我们会深度复制问题1） :

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

需要注意的是，这里的参数并没有经过大量的实验验证。这些参数通常在其他自然语言处理问题，特别是文本分类上都有不错的效果。读者可能需要修改对应语言中的停用词列表。

现在我们可以分别得到问题1和问题2的TFIDF矩阵：

```python
q1_tfidf = tfv_q1.fit_transform(data.question1.fillna("")) 
q2_tfidf = tfv_q2.fit_transform(data.question2.fillna(""))
```


> 在TFIDF处理工程中，我们根据所以可用数据计算出TFIDF矩阵（我们用了`fit_transform`方法）。这个Kaggle竞赛中的常用手段，可以帮助我们拿到高分。但是，如果面对真实场景，读者可能希望排除训练集和验证集中的一些数据，以便保证TFIDF处理可以泛化到新的未知数据集中。

有了TFIDF特征后，我们继续介绍SVD特征。SVD是一种特征分解方法，即奇异值分解。它广泛应用于自然语言处理中，其中一个技术叫做**潜在语义分析（Latent Semantic Analysis ，LSA）**。

> SVD和LSA的具体讨论超出了本书的范畴。读者可以参考下面两个简单明了的网上教程，了解其中的思想： https://alyssaq.github.io/2015/singular-value-decomposition-visualisation/ 和https://technowiki.wordpress.com/2011/08/27/latent-semantic-analysis-lsa-tutorial/

要创建SVD特征，我们再次使用`scikit-learn`实现。这个实现是传统的SVD的变种，叫做`TruncatedSVD`。

>  `TruncatedSVD` 是SVD方法的近似，提供了可靠而迅速的SVD矩阵分解。读者可以参考下面的网站，找到更多有关技术实现和应用的建议：http://langvillea.people.cofc.edu/DISSECTION-LAB/Emmie'sLSI-SVDModule/p5module.html 

```python
from sklearn.decomposition import TruncatedSVD 
svd_q1 = TruncatedSVD(n_components=180) 
svd_q2 = TruncatedSVD(n_components=180)
```

我们选择180个成分，进行SVD分解。这些特征都是在TFIDF矩阵上计算的：
```python
question1_vectors = svd_q1.fit_transform(q1_tfidf) 
question2_vectors = svd_q2.fit_transform(q2_tfidf)
```
特征集-3来自于TF-IDF特征和SVD特征。例如，我们可以只用TF-IDF特征建模两个问题，或者使用TF-IDF特征加上SVD特征，然后加入模型进行学习。这些特征的解释如下：

特征集-3(1) 或者记作`fs3_1`包含两个问题的TF-IDF，通过水平堆砌生成，最终传给机器学习模型： 

![](figures\184_1.png)

这个过程可以编码为：

```python
from scipy import sparse
# obtain features by stacking the sparse matrices together 
fs3_1 = sparse.hstack((q1_tfidf, q2_tfidf))
```
特征集-3(2), 或者记作`fs3_2`，通过合并两个问题生成一个TFIDF而实现： 

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

这个特征集合的子集，特征集-3(3)或者记作`fs3_3`，包括两个TF-IDFs和两个SVD：

![](figures\185_2.png)

This can be coded as follows:
```python
# obtain features by stacking the matrices together 
fs3_3 = np.hstack((question1_vectors, question2_vectors))
```

我们可以类似创建更多TF-IDF和SVD组合特征，把它们分别记作`fs3-4` 和`fs3-5`。 下图给出了构造过程，读者可以练习着尝试编码。

特征集-3(4)或者记作`fs3_4`：

![](figures\186_1.png)

特征集-3(5)或者记作`fs3_5`：

![](figures\186_2.png)

有了基础的特征集，以及TF-IDF和SVD特征，我们可以继续构造更加复杂的特征，进而支持后面的机器学习和深度学习模型。

### 使用Word2vec词嵌入映射

简单的讲，Word2vec模型就是两层的神经网络，它接收文本语料作为输入，输出语料库中每个词语的向量。通过拟合，意思相近的词语的向量也会彼此靠近。相比意思不同的词语而言，它们之间的距离也更小。

如今，Word2vec已经变成了自然语言处理问题中的标准流程，为信息检索任务提供非常有用的理解。对于具体的问题，我们会使用谷歌新闻的词向量。它是通过在谷歌新闻语料库上预训练得出的Word2vec模型，

当每一个单词使用Word2vec的向量表示时，都会在空间中对应一个位置，如下图所示：

![](figures\187_1.png)

如果我们使用谷歌新闻语料库上的预训练词向量，上面例子的所有词语，例如Germany，Berlin，France和Paris都可以表示为300维的向量。借助这些词语的Word2vec表示，把Berlin的向量减去Germany的向量，再加上France的向量，我们会得到一个与Paris向量非常接近的向量。因此Word2vec模型中的向量保留了词语的含义。这些向量所蕴含的信息会给我们的任务带来非常有用的特征。
> 想要获得关于Word2vec应用的易懂且更多详尽的介绍，建议阅读https://www.distilled.net/resources/a-beginners-guide-to-Word2vec-aka-whats-the-opposite-of-canada/，或者如果需要更多严格数学定义的解释，推荐阅读这篇文章：http://www.1-4-5.net/~dmm/ml/how_does_Word2vec_work.pdf

要加载Word2vec特征我们输赢。如果没有安装Gensim，读者可以使用`pip`安装。同时，我们也建议安装`pyemd`库，因为会在WMD距离计算中用到。WDM函数可以帮助我们把两个Word2vec向量关联起来：
```
pip install gensim 
pip install pyemd
```


要加载Word2vec模型，下载`GoogleNews-vectorsnegative300.bin.gz`的二进制文件，使用 Gensim's的`load_Word2vec_format`函数加载到内存中。读者也可以从亚马逊的AWS仓库中下载二进制文件。使用shell中的`wget`命令：
```
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors	negative300.bin.gz"
```


完成文件下载和解压之后，使用Gensim的`KeyedVectors`函数： 

```python
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin.gz', binary=True)
```

现在，我们可以通过调用`model[word]`获取每一个词语的向量。但是，要想处理句子的时候，还会有新的问题。在这个例子中，我们需要问题1和问题2中所有词的向量，以便后续比较。我们可以使用以下代码。这段代码可以给出一条谷歌新闻句子中所有词语的向量，并最终给出归一化的向量。我们称之为句向量或者Sent2Vec。

运行下面的函数前，确定已经安装了 **自然语言工具包（Natural Language Tool Kit ，NLTK）** ：

```
$ pip install nltk
```


建议下载`punkt`和`stopwords`程序包，它们也是NLTK的一部分：
```python
import nltk 
nltk.download('punkt') 
nltk.download('stopwords')
```


NLTK可用之后，我们只需要运行下列代码，定义`sent2vec`函数：
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

当句子为空时，我们可以返回一个标准的零值向量。

要计算问题间的相似度，另外一个特征是词语移动距离（ Word mover's distance）。词语移动距离使用Word2vec词嵌入技术，其找工作原理与给出两篇文档距离的测地距（earth mover's distance）类似。简单的说，词语移动距离提供了从一个文档到另一个文档移动所有词语所需的最小距离。

> 词语移动距离由以下文章提出： *KUSNER, Matt, et al. From word embeddings to document distances. In: International Conference on Machine Learning. 2015. p. 957-966*。该文章的下载地址：http://proceedings.mlr.press/v37/kusnerb15.pdf。需要实际操作介绍，读者可以参考Gensim的距离实现教程：https://markroxor.github.io/gensim/static/notebooks/WMD_tutorial.html

最终的Word2vec （w2v）特征也包括其他距离，例如更常见的欧式距离或余弦距离。完整的特特征集合也包含对两个文档向量分布的一些度量：
1. 词语移动距离

2. 归一化词语移动距离

3. 问题1向量和问题2向量的余弦距离

4. 问题1向量和问题2向量的曼哈顿距离

5. 问题1向量和问题2向量的杰拉德相似度

6. 问题1向量和问题2向量的堪培拉距离

7. 问题1向量和问题2向量的欧氏距离

8. 问题1向量和问题2向量的闵可夫斯基距离

9. 问题1向量和问题2向量的布雷克蒂斯距离

10. 问题1向量的偏度

11. 问题2向量的偏度

12. 问题1向量的峰度

13. 问题2向量的峰度

   
所以这些Word2vec特征记作`fs4`。

另一个w2v特征集合包含Word2vec向量自身的矩阵：

1. 问题1的Word2vec向量

2. 问题2的Word2vec向量


它们记作`fs5`：

```python
w2v_q1 = np.array([sent2vec(q, model) 
                   for q in data.question1]) 
w2v_q2 = np.array([sent2vec(q, model) 
                   for q in data.question2])
```


 为了快速实现Quora问题的Word2vec词向量之间的不同距离度量，我们使用`scipy.spatial.distance`模块：

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


 所有这些和距离相关的特征名都保存在`fs4_1`列中：


```python
fs4_1 = ['cosine_distance', 'cityblock_distance','jaccard_distance', 
         'canberra_distance','euclidean_distance', 'minkowski_distance',
         'braycurtis_distance']
```


两个问题的Word2vec矩阵通过水平堆叠，存储在变量`w2v`中，以便后用：

```python
w2v = np.hstack((w2v_q1, w2v_q2))
```


词语移动距离的函数首先对两个问题进行小写转换，并去除停用词，然后返回距离。然而，我们也可以把所有Word2vec向量转换为L2-归一化向量（每一个向量都转换为单元范式，即对向量的元素取平方和，结果的和为1），计算距离的归一化值  。使用`init_sims`方法：


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


完成最终计算，我们就结束了大部分重要特征的创建。这些特征是基本的机器学习模型所需的   ，也会作为深度学习模型的基线标准。下表展示了备用特征的截图：

![](figures\192_1.png)

现在让我们训练一些机器学习模型，以及其他Word2vec特征。

### 测试机器学习模型

测试之前，根据系统情况不同，读者可能需要清理内存和释放空间，以防机器学习模型使用之前的数据结构。这个过程可以通过`gc.collect`实现，它可以删除所有过去的不再需要的变量，并检查可用内存，使用`psutil.virtualmemory`函数返回准确的结果：

```python
import gc
import psutil
del([tfv_q1, tfv_q2, tfv, q1q2,
     question1_vectors, question2_vectors, svd_q1,
     svd_q2, q1_tfidf, q2_tfidf])
del([w2v_q1, w2v_q2]) 
del([model]) 
gc.collect() 
psutil.virtual_memory()
```


现在，我们可以汇总出到目前为止，所有创建的特征及其含义：

- fs_1: 基础特征列表
- fs_2: 模糊特征列表
- fs3_1: 不同问题的TFIDF稀疏数据矩阵
- fs3_2: 合并问题的TFIDF稀疏数据矩阵
-  fs3_3: SVD稀疏数据矩阵
- fs3_4: SVD统计特征列表
- fs4_1:  Word2vec距离特征列表 
- fs4_2: 词语移动距离特征列表
- w2v:  使用`Sent2Vec`函数后的转换表述的Word2vec向量矩阵

我们会评估两个基础并且常见的机器学习模型，即logistic回归和`xgboost`程序包中的梯度下降。下表给出了Kaggle竞赛中，logistic回归和`xgboost`算法在不同特征集合上的表现：

| 特征集合                                                 | logistic回归准确度 | `xgboost`准确度 |
| -------------------------------------------------------- | ------------------ | --------------- |
| 基础特征（fs1）                                          | 0.658              | 0.721           |
| 基础特征+模糊特征（fs1+fs2）                             | 0.660              | 0.738           |
| 基础特征+模糊特征+w2v特征（fs1+fs2+fs4）                 | 0.676              | 0.766           |
| w2v向量特征（fs5）                                       | *                  | 0.780           |
| 基础特征+模糊特征+w2v特征+w2v向量特征（fs1+fs2+fs4+fs5） | *                  | 0.814           |
| TFIDF-SVD（fs3-1）                                       | 0.777              | 0.749           |
| TFIDF-SVD（fs3-2）                                       | 0.804              | 0.748           |
| TFIDF-SVD（fs3-3）                                       | 0.706              | 0.763           |
| TFIDF-SVD（fs3-4）                                       | 0.700              | 0.753           |

\* *由于对内存需求太高，这个模型没有训练。*

我们可以把这些性能作为深度学习模型的基线或者最低标准，但是也不会完全照搬其中的工作。

接下来，我们会引入所有必须的程序包。对于logistic回归，我们会使用scikit-learn实现。

`xgboost`是一个可扩展的和可移植的分布式梯度提升库（基于树结构的机器学习集成模型）。它是由华盛顿大学的陈天奇发明的，后来由Bing Xu实现了Python的封装，由Tong He实现了R的接口（读者可以通过主要发明者的网站了解到xgboost背后的故事 homes.cs.washington.edu/~tqchen/2016/03/10/story-and-lessons-behind-the-evolution-of-xgboost.html ）。`xgboost`可以在Python，R，Java，Scala，Julia和C++中使用，并且可以在单机上使用（利用多线程），也可以部署在Hadoop和Spark集群上。

> 安装`xgboost`的具体说明可以在这里找到： github.com/dmlc/xgboost/blob/master/doc/build.md
>
> Linux和macOS系统上的`xgboost`安装很简单，但是Windows系统上的安装要稍微注意一下。 
>
> 我们给出Windows系统上的安装`xgboost`的具体步骤：
>
> 1. 首先下载安装Git的Windows版本（git-forwindows.github.io）
> 2. 然后，需要在系统中安装MinGW编译器，可以根据系统的配置在 www.mingw.org 下载
> 3. 在命令行中，执行：
>
> ```
> $> git clone --recursive https://github.com/dmlc/xgboost 
> $> cd xgboost
> $> git submodule init
> $> git submodule update
> 
> ```
>
> 4.  在命令行中，复制64位系统的配置信息作为默认配置：
>
> ```
> $> copy make\mingw64.mk config.mk
> ```
>
> ​      也可以复制32位版本：
>
> ```
> $> copy make\mingw.mk config.mk
> ```
>
> 5.  配置文件复制完成之后，可以运行编译器，设置4个线程以加速编译过程：
>
> ```
> $> mingw32-make -j4
> 
> ```
>
> 6.  在MinGW中，有`make`命令和`mingw32make`；如果读者使用不同的编译器，之前的命令可能不会起作用，读者可以尝试：
>
> ```
> $> make -j4
> 
> ```
>
> 7.  最后，如果编译器顺利完成编译，就可以使用Python安装程序包：
>
> ```
> $> cd python-package
> $> python setup.py install
> ```


`xgboost`正确安装之后，我们可以引入机器学习算法模型： 
```python
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler 
import xgboost as xgb
```


由于我们会用到logistic回归求解器（它是一个https://github.com/EpistasisLab/tpot/issues/292中的`sag`求解器，根据数据的规模需要线性计算时间）：

```python
scaler = StandardScaler() 
y = data.is_duplicate.values 
y = y.astype('float32').reshape(-1, 1) 
X = data[fs_1+fs_2+fs3_4+fs4_1+fs4_2]
X = X.replace([np.inf, -np.inf], np.nan).fillna(0).values
X = scaler.fit_transform(X)
X = np.hstack((X, fs3_3))
```

我们选定一些数据用于训练，首先过滤`fs_1`, `fs_2`, `fs3_4`, `fs4_1`和`fs4_2`集合中的变量，然后把`fs3_3`的稀疏SVD矩阵堆叠放置。我们也可以随机划分，分出1/10的数据用于验证（这样可以有效的评估模型的质量）：

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

在第一个模型中，我们尝试使用logistic回归，并设定L2正则化参数C为0.1
（最保守的正则化）。模型准备好后，我们在验证集合上测试效果（`x_val`是训练矩阵，`y_val`是正确答案）。结果通过准确率，即测试集上正确预测的占比，进行评估：

```python
logres = linear_model.LogisticRegression(C=0.1,
                                         solver='sag', max_iter=1000)
logres.fit(x_train, y_train) 
lr_preds = logres.predict(x_val)
log_res_accuracy = np.sum(lr_preds == y_val) / len(y_val)
print("Logistic regr accuracy: %0.3f" % log_res_accuracy)
```

等待片刻（求解器最多迭代1,000次，就会停止收敛），最终的测试集准确率是0.743。它会作为我们的第一个基线。

现在，我们试着使用`xgboost`算法做预测。作为梯度提升算法的一种，它的方差（可以拟合复杂的预测函数，但要注意过拟合）比logistic回归要大， 而logistic回归的偏差较大（最后我们可以看到，它是系数的和）。我们可以从`xgboost`中得到更好的结果。我们可以把树的最大深度固定位为4（这个深度较浅，可以避免过拟合），并设置学习率为0.02（模型需要生成许多树，因为学习速率比较慢）。我们还可以设置一个监测列表，查看测试集上的表现。一旦误差超过50步不再减少，就尽早停止训练。

> 在同一个集合上（比如例子中的验证集）尽早停止训练返回最终结果并不是好的尝试。在现实世界中，我们应该设置一个验证集以便调整模型运行，例如尽早停止，以及一个测试集来报告泛化情形下的结果。

设置完成之后，我们运行算法。这一次，训练的时间要比logistic回归的时间长：

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


`xgboost`在验证集上的准确率是0.803。

###  搭建TensorFlow模型

本章的深度学习模型会使用TensorFlow搭建，并在Abhishek Thakur的Keras代码上修改（读者可以阅读源代码：https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question）。Keras是一个Python库，提供TensorFlow的简单接口。Tensorflow有Keras的官方支持。使用Keras训练的模型可以轻松的转换为TensorFlow模型。Keras支持深度学习模型的快捷原型搭建和测试。在我们的项目中，我们会从头开始，完整的编写TensorFlow方案。

首先，引入必要的库，特别是TensorFlow，打印信息，检查版本：

```python
import zipfile
from tqdm import tqdm_notebook as tqdm
import tensorflow as tf
print("TensorFlow version %s" % tf.__version__)
```


现在，我们可以把数据加载到pandas数据库`df`中，也可以共本地磁盘加载。 我们把缺失值替换为空字符串，并把包含答案的`y`变量编码为1（重复问题）或0（非重复问题）：

```python
try:
    df = data[['question1', 'question2', 'is_duplicate']] 
except:     
    df = pd.read_csv('data/quora_duplicate_questions.tsv', sep='\t')
df = df.drop(['id', 'qid1', 'qid2'], axis=1)
df = df.fillna('') y = df.is_duplicate.values
y = y.astype('float32').reshape(-1, 1)
```


现在我们可以继续构建深度神经网络。

### 深度神经网络之前的处理

给神经网络输入数据之前，我们必须对数据进行切分，并转化为序列。因此，我们使用Keras的`Tokenizer`函数，并设置词语的最多数量为200,000 ，序列最长为40。任何多于40个词语的句子都会被切开保留前40个词语：

```python
Tokenizer = tf.keras.preprocessing.text.Tokenizer 
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences 
tk = Tokenizer(num_words=200000) max_len = 40
```


切词器`tk`设置完成后，就可以在两个问题的拼接列表上使用了，并学习语料库中所有可能的词语：

```python
tk.fit_on_texts(list(df.question1) + list(df.question2))
x1 = tk.texts_to_sequences(df.question1) 
x1 = pad_sequences(x1, maxlen=max_len) 
x2 = tk.texts_to_sequences(df.question2) 
x2 = pad_sequences(x2, maxlen=max_len) 
word_index = tk.word_index
```

`word_index`是一个词典，包含所有被切出的词语及其所对应的索引，以便跟踪切词器的运行效果。

使用GloVe词嵌入算法的时候，我们必须加载到内存中，和之前获取Word2vec词嵌入方法类似。

GloVe词嵌入模型可以通过shell命令方便的获取：

```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
```

GloVe词嵌入算法与Word2vec算法类似，都可以根据词语共现把词语编码到复杂的多维空间上。但是，正如下面文章所介绍的，http://clic.cimec.unitn.it/marco/publications/acl2014/baroni-etal-countpredict-acl2014.pdf  —BARONI，Marco；DINU，Georgiana；KRUSZEWSKI，Germán，不要计数，要预测！（*Don't count, predict! A systematic comparison of context-counting vs. context-predicting semantic vectors. In: Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2014. p. 238-247.*）

GloVe并不是从试图通过背景词来预测中心词语的神经网络优化中得来的，但是Word2vec却是。GloVe来自于经过降维（例如在准备数据中提到的SVD矩阵分解）的词语共现计数矩阵（矩阵记录了每一行的一个词语与每一列的一个词语共同出现了几次）。

> 那么我们为什么使用GloVe，而不用Word2vec？事实上，二者的最大区别是GloVe词嵌入在某些问题上效果更好，而Word2vec词嵌入在其他问题上表现更好。在我们的项目中，经过试验发现GloVe词嵌入与深度学习结合效果更好。读者可以通过斯坦福大学的官方网页了解更多GloVe的信息和使用：https://nlp.stanford.edu/projects/glove/

理解GloVe词嵌入后，我们可以创建`embedding_matrix`。 `embedding_matrix`数组来自GloVe文件的词嵌入向量（每一个向量300维）。

下面的代码读入词向量文件，并存储在词嵌入矩阵中。这个矩阵最终会包含所有切分的词语，以及对应的向量：

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

首先创建空值`embedding_matrix`，然后每一个行向量准确对应矩阵的具体行数。词语和行之间的这种对应已经在切词器的编码过程中定义好，现在可以通过`word_index`词典调用。 

`embedding_matrix`完成词嵌入加载后，我们就可以开始构建深度学习模型了。

### 深度神经网络搭建模块

在这一节中，我们会给出深度学习的关键函数。首先，我们会进行批输入（提供数据分块进行深度神经网络学习），然后准备复杂的LSTM结构的构建模块。

> LSTM的结构在第7章使用LSTM进行股票价格预测，长短期记忆网络——LSTM101一节中给出。

我们用的第一个函数是`prepare_batches`。这个函数接收问题序列，并根据`step`值（批的多少）返回问题列表的列，其中里边的列是要学习的成批序列：

```python
def prepare_batches(seq, step):    
    n = len(seq)     
    res = []     
    for i in range(0, n, step):  
        res.append(seq[i:i+step])
    return res
```

`dense`函数会根据提供的规模创建一个全连接的神经网络层，并用均值为0，2的平方根除以输入的特征数作为标准差的随机正态分布数字激活和初始化.

一个恰当的初始化可以帮助输入的导数后向传播到较深的网络。事实上：

- 如果给网络的初始化权重太小，导数会随着传播逐渐衰减，直到变得很微弱而无法触动激活函数。
-  如果给网络的初始化权重太大，导数会随着传播逐渐增大（即所谓的梯度爆炸）。网络就不会收敛到一个合适的解，并且会因为处理的数值过大而中断。

初始化过程要确保权重通过合理的设置而支持导数可以传播很多层。深度学习中有许多初始化的过程，例如Glorot和Bengio的Xavier（其实，Xavier也是Glorot的姓），以及He，Rang，Zhen和Sun在二人工作上提出的另一种方法，通常叫做He。

> 权重初始化是神经网络架构中一个非常技术性的操作，也是很相关的一环。如果读者想要了解更多，可以首先阅读这篇博客，它也涉及到一些更加数学的解释：http://deepdish.io/2015/02/24/network-initialization/ 

在这个项目中，我们倾向于使用He初始化，因为它对整流单元的效果很好。整流单元，即ReLu，是深度学习的动力源，因为它支持梯度信号传播的同时可以避免梯度散失和梯度爆炸问题。然而从实际角度讲，用ReLu激活的神经元在大多说情况下抑制了零值。保证方差足够大进而有持续的输入和输出梯度通过每一层，就可以使得激活过程奏效，正如在下面的文章中介绍的：HE, Kaiming, et al. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In: Proceedings of the IEEE international conference on computer vision. 2015. p. 1026-1034。这篇文章可以在 https://arxiv.org/abs/1502.01852上找到:

```python
def dense(X, size, activation=None):
he_std = np.sqrt(2 / int(X.shape[1]))     
out = tf.layers.dense(X, units=size,           
                      activation=activation,  
                      kernel_initializer=\ 
                      tf.random_normal_initializer(stddev=he_std))     
return out
```

接下来，我们开始另一种神经网络层，时间分布的全连接层。

这种神经网络层用在递归神经网络中，保持输入和输出之间的一对一关系。一个RNN结构（带有一定数量的细胞单元提供通道输出），始于标准的全连接层，接收行（样本）列（序列）矩阵，其结果矩阵的维度是行乘以通道（细胞单元）数量。如果使用时间分布的全连接层，它的输出维度会是行乘以列乘以通道数。事实上，一个全连接神经网络分配给了每一个时间戳（每一列）。

时间分布的全连接层经常用在诸如输入序列已知，并希望根据序列的出现，标记每一个输入的情形。这是一个标记任务的常见场景，例如多标签分类或者词性标注。在我们的项目中，我们会在GloVe词嵌入后使用时间分布的全连接层，以便处理每一个GloVe词向量随着问题序列中词语的出现而改变的现象。

例如，假设有两个例子的序列（一对问题），每一个都有3个序列（词语），每一个序列都由4个元素（词嵌入）构成。如果我们有这样的数据集，并传给带有5个隐藏单元的时间分布全连接层，我们会得到大小为 (2, 3, 5)的张量。当输入通过时间分布全连接层时，每一个例子都保留了序列，但是词嵌入结果被5个隐藏单元替换掉了。继续把结果传给一个1轴上的维度约减过程，我们可以得到大小为 (2,5)的张量，这就是最终的向量。

> 如果读者想重复之前的例子，如下：
>
> ```python
> print("Tensor's shape:", X.shape)
> tensor = tf.convert_to_tensor(X, dtype=tf.float32)
> dense_size = 5
> i = time_distributed_dense(tensor, dense_size)
> print("Shape of time distributed output:", i)
> j = tf.reduce_sum(i, axis=1) 
> print("Shape of reduced output:", j)
> ```



> 和其他神经网络层相比，时间分布全连接层的概念可能会有点不容易理解。网络上有一个关于它的讨论。读者可以通过Keras问题的一些介绍中获得更多的认识：https://github.com/keras-team/keras/issues/1029

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


`conv1d`和`maxpool1d_globa`函数分别下面TensorFlow函数的封装：`tf.layers.conv1d`（https://www.tensorflow.org/api_docs/python/tf/layers/conv1d），它是一个卷积层；`tf.reduce_max` (https://www.tensorflow.org/api_docs/python/tf/reduce_max)，它计算输入张量所有维度上最大值。在自然语言处理中，这种池化（也叫全局最大池化）比标准池化用的多，它也经常用在计算机视觉的深度学习实践中。 正如在交叉验证的问答中所提到的（https://stats.stackexchange.com/a/257325/49130），全局最大池化使用输入向量的最大值，然而标准池化根据池的大小返回由输入向量在不同池化下的最大值构成新向量：

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


核心函数`lstm`通过随机数生成器在每一次运行时都会被不同的初始化。而随机数生成器使用He初始化策略。`lstm`是TensorFlow中两个模块的封装：`tf.contrib.rnn.BasicLSTMCell`，对应基础的LSTM递归网络层（https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell），`tf.contrib.rnn.static_rnn`，负责创建由单元层刻画的递归神经网络（https://www.tensorflow.org/versions/r1.1/api_docs/python/tf/contrib/rnn/static_rnn）。

> 基础的LSTM递归网络层实现基于文章：*ZAREMBA，Wojciech；SUTSKEVER，Ilya；VINYALS，Oriol. Recurrent neural network regularization*. arXiv preprint arXiv:1409.2329, 2014，可以查看https://arxiv.org/abs/1409.2329。

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


现在，我们已经有了所有必需的搭建模块，进而定义神经网路的架构以区分重复问题。

### 设计学习架构

我们首先设置一些参数，例如GloVe词嵌入的特征数，滤波器的数量和长度，最大池化策略的长度和学习率：

```python
max_features = 200000 
filter_length = 5 
nb_filter = 64 
pool_length = 4 
learning_rate = 0.001
```

试图掌握不同词语数量的语义差别，最终检测出重复问题，这个任务确实很难，肯定需要复杂的架构。因此，经过多次试验，我们创建了一个包含LSTM，时间分布的全连接层，以及1d-卷积神经网络的更深的模型。这个模型有6类输入，可以通过拼接合成1个。拼接之后，整个架构由5个全连接层和1个带有sigmoid激活函数的输出层构成。

下图展示了完整的模型：

![](figures\205_1.png)

第一类输入包含一个有Glove算法初始化的词嵌入层，已经后续的时间分布全连接层。第二类输入包含1D卷积神经网络和GloVe模型词嵌入层。第三类输入是一个基于词嵌入的LSTM模型。其他三类输入与上述三个的模式相同，只不过是对应第二个问题。 

我们首先定义了6个模型，然后把6种模型拼接起来。最终，6个模型拼接成1个，即来自6个模型的向量水平堆叠在一起。

尽管下面的代码很长，但是不难理解。每一步都从3个输入占位符开始：`place_q1`， `place_q2`和`place_y`，它们负责给所有6个模型传入第一个问题，第二个问题和目标响应。这些问题都通过GloVe（ `q1_glove_lookup` 和`q2_glove_lookup` ）以及随机均匀策略进行词嵌入。两种词嵌入都是300维。 

前两个模型`model_1` 和`model_2`，获取GloVe词嵌入结果，并使用时间分布全连接层。

接着两个模型，`model_3`和`model_4`，获取GloVe词嵌入结果并按照卷积，dropout，最大池化等一系列操作处理。最终输出向量通过批归一化，以便保证不同批之间稳定的方差。

> 如果读者想知道批归一化的具体细节，Abhishek Shivkumar在Quora上的回答清楚地提供了所有关于批处理的重要理解，以及为什么其在神经网络中会有效：https://www.quora.com/In-layman%E2%80%99s-terms-what-is-batch-normalisation-what-does-it-do-and-why-does-it-work-so-well/answer/Abhishek-Shivkumar

最后，`model_5`和`model_6`获取均匀随机词嵌入结果，并使用LSTM处理。所有6个模型的结果拼接在一起，并经过批归一化：

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


然后，我们添加5个带有dropout和批归一化的全连接层，完成整个架构。最后是一个但又sigmoid函数的输出层。整个模型使用基于对数损失的`AdamOptimizer`进行优化：
    

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


定义好架构后，我们初始化会话，并准备学习。一个好的习惯是，我们把可用数据分成训练集（9/10）和测试集（1/10）。设置随机种子可以支持重现结果：

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


如果读者运行下列代码片段，训练就开始了。可以看到，模型的准确率随着论数的增多而增长。但是，根据需要迭代的批的数量不同，模型会花费很长的的时间来训练。在NVIDIA Titan X上，这个模型每轮需要300秒。为了平衡准确率和训练时间，我们选择训练10轮：

```python
val_idx = np.arange(y_val.shape[0]) 
val_batches = prepare_batches(val_idx, 5000) 

no_epochs = 10

# see https://github.com/tqdm/tqdm/issues/481 
tqdm.monitor_interval = 0

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

经过10轮训练，模型给出了82.5%的准确率。这比之前的基准表现提高了不少。当然，模型还可以通过更好的预处理和切词进一步提升效果。多训练几轮（至200轮）也可以提升准确率。词干提取和词形还原也可以把效果提升到Quora博客中当前最好的88%的效果。

训练完成后，我们可以使用内存会话测试一些问题的评估。我们使用两个重复问题，但是处理过程对于任何一对问题都是有效的。

> 对于许多机器学习算法，其依赖于训练集上的数据分布，而真实问题的分布可能和训练集上的分布完全不同，这样使得算法预测变得更加困难。

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


运行上面的代码，结果会给出这是重复问题（answer : 1.0)。

### 小结

在这一章里，我们通过TensorFlow构建了一个深度神经网络，以便监测Quora数据集上的重复问题。这个项目带着我们讨论，修改，实操了之前章节中学到的各个操作：TF-IDF，SVD，经典机器学习算法，Word2vec和GloVe词嵌入和LSTM模型。

最后，我们得到了一个准确率在82.5%的模型，这比传统的机器学习方法眼好，也很接近当前Quora博客中报出的最好结果。

需要注意的是，本章的模型和方法可以应用到任何语义匹配问题上。

---