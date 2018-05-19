

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