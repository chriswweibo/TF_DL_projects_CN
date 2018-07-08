## 第7章 训练像人类一样的聊天机器人 

本章会介绍如何训练一个自动回答简单的通用问题的聊天机器人，以及如何创建HTTP端点借助API提供答案。具体说来，我们会介绍：

- 什么是语料库，如何做语料库预处理
- 如何训练聊天机器人，以及如何测试
- 入股创建HTTP断点提供API

### 项目简介

如今，聊天机器人在帮助用户的应用上越来越广泛。许多公司，包括银行，移动通信公司，以及大型电子商务平台都使用聊天机器人帮助客户，以及售前支持。网站的问答页面不再能够满足用户：如今每个用户都希望得到针对自己问题的回答，这个问题可能并没有在问答页面中涉及。而且，聊天机器人可以帮助公司省去回答各种平凡问题的额外服务：这俨然是个双赢的方案。

随着深度学习的盛行，聊天机器人已经变成了非常流行的工具。借助胜读学习，我们可以训练机器人提供更好的更个性化的回答，在最终实现的时候，保留每个用户的信息。

简单的说，主要有两种方式的聊天机器人：一种是简单的，试图理解主题，给所有同一主题的问题提供相同的答案。例如，在火车站网站上，问题“我在那里可以找到城市A到城市B的服务时间表？”，以及“离开城市A的下一班火车是什么时候？”很可能会得到同样的答案，可能是“服务网络的时间表在这个网页上：链接地址。”。

这种聊天机器人本质上是使用分类算法理解主题（本例中，两个问题都是关于时间表的）。给定一个主题，机器人总是提供相同的答案。通常，这种聊天机器人有一个包含N个主题的列表，已经N个答案；而且如果分到的主题概率很低（问题太模糊，或者暗含的主题不在列表中），机器人通常会问一些更加具体的问题并让用户重复问题，最终给出问题的其他解决途径（例如，发送邮件或者给客服中心打电话）。

第二种类型的聊天机器人更加先进，也更加复杂。其回答是使用RNN生成的，方式与机器翻译的一样（可以查看之前的章节）。这种聊天机器人可以提供更加个性化的回答，以及更加具体的回复。事实上，它们不会猜测聊天的主题，相反使用RNN就可以更好的理解用户的问题，给出最可能的答案。使用这种类型的聊天机器人回答两个不同问题，用户几乎不可能得到同一个答案。

在本章中，我们会使用RNN构建第二种类型的聊天机器人。构建方法与之前章节的机器翻译系统类似。而且，我们会介绍如何把聊天机器人配置到HTTP的端点上，以便在网站上作为服务调用聊天机器人，甚至支持更简单的命令行调用。

### 输入语料库

不幸的是，我们并没有在互联网上找到开源的免费使用的面向客户的数据集。因此，我们会使用通用数据集，而不是真正关注客户服务领域的数据集，来训练机器人。具体地说，我们会使用康纳尔大学的康奈尔电影对话语料库。这个语料库包含许多原始电影脚本中的对话。因此聊天机器人会为虚拟问题提供答案，而不是为真实问题提供答案。康奈尔的语料库包括来自617部电影的，1万多个电影角色的，超过20万条对话记录。

> 数据集的链接：https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html。我们要感谢公开这个语料库的作者：让实验，可重复性研究和知识共享更加容易。

这个数据集是一个`.zip`压缩文件。解压之后，文件包含以下几个子文件：

- `README.txt` 包含三数据集的介绍，语料库文件的格式，收集过程的具体信息和作者的联系方式。

- `Chameleons.pdf` 是语料库的原始文件。尽管这个文件对聊天机器人没有多大意义，但是它介绍了对话中使用的语言，是深入理解语料库的优质信息。

- `movie_conversations.txt` 包含所有对话结构。对于每一组对话，它包括两个对话角色的ID，电影的ID，和以时间顺序排列的语句（或者准确的说是，语言表达）ID列表。例如，文件的第一行是：

  `u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']`

  它的意思是电影`m0`中的用户`u0`与用户`u2`发生了对话，对话有4条语言表达：“`L194`”, “`L195`”, “`L196`” 和“`L197`”。

- `movie_lines.txt` 包含每一个语言表达ID的实际文本，以及发出的人。例如，语言表达`L195`的形式如下：

  `L195 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ Well, I thought we'd start with pronunciation, if that's okay with you`.

  因此，语言表达`L195`的内容是“Well, I thought we'd start with pronunciation, if that's okay with you”。它是由角色`u2`发出的，名字是电影`m0`中的`CAMERON`。

- `movie_titles_metadata.txt` 包含电影的信息，例如电影名，年份，IMDB评分，IMDB投票数和流派。例如，电影`m0`的描述如下：

  `m0 +++$+++ 10 things i hate about you +++$+++ 1999 +++$+++ 6.90 +++$+++ 62847 +++$+++ ['comedy', 'romance']`

  所以ID为`m0`的电影名称为“10 things i hate about you”，拍摄于1999年，是一部浪漫喜剧。它拥有6.3万IMDB的投票，平均分为6.9（满分为10分）。

* `movie_characters_metadata.txt` 包含电影角色的信息，例如角色出现的电影名称，性别（如果知道的话）和影片字幕中的人物位置（如果知道的话）。例如，角色`u2`在文中的描述如下：

  `u2 +++$+++ CAMERON +++$+++ m0 +++$+++ 10 things i hate about you +++$+++ m +++$+++ 3`

  角色`u2`的名字为`CAMERON`，它出现在电影`m0`中，名称为“10 things i hate about you"。角色的性别为男性，影片字幕中的人物位置为第三个。

- `raw_script_urls.txt` 包含每一个电影对话的原始URL链接。例如，电影`m0`如下：

  `m0 +++$+++ 10 things i hate about you +++$+++`

  `http://www.dailyscript.com/scripts/10Things.html`

  可以看到，大部分文件都使用标记`+++$+++`来分割字段。另外，文件的格式看起来也很简单，利于解析。需要在解析文件的时候注意，文件格式不是UTF-8的，而是ISO-8859-1的。

### 创建训练集

现在我们创建用于聊天机器人的训练集。我们需要以正确顺序组织角色间的所有对话。而上述语料库可以满足我们的需求。如果本地没有所需的`zip`文档，我们要先下载下来，然后把文件解压到一个临时文件夹中（Windows环境下，应该放在`C:\Temp`），我们只需读取 `movie_lines.txt和`movie_conversations.txt` 文件，用于创建连续的语言表达数据集。

现在我们可以在`corpora_downloader.py`中逐步创建几个函数。第一个函数是用于本地磁盘没有保存的情况下，从因特网上获取文件。
Internet, if not available on disk.

```python
def download_and_decompress(url, storage_path, storage_dir):
    import os.path
    directory = storage_path + "/" + storage_dir
    zip_file = directory + ".zip"
    a_file = directory + "/cornell movie-dialogs corpus/README.txt"
    if not os.path.isfile(a_file):
        import urllib.request
        import zipfile
        urllib.request.urlretrieve(url, zip_file)
        with zipfile.ZipFile(zip_file, "r") as zfh:
            zfh.extractall(directory)
return
```

这个函数的功能如下：它首先检查本地是否有“`README.txt`”文件；如果没有，下载文件（借助于`urllib.request`模块的`urlretrieve`函数）,并解压`zip`文件（使用`zipfile`模块）。

接着，读取对话文件，抽取语言表达ID的列表。注意，格式为：`u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']`，我们需要的是经过分隔符`+++$+++`分割后的第四个元素。而且，我们需要清除方括号和单引号，拿到纯粹的ID列。 因此，我们首先加载`re`模块，函数如下：

```python
import re
def read_conversations(storage_path, storage_dir):
    filename = storage_path + "/" + storage_dir + "/cornell movie-dialogs corpus/movie_conversations.txt"
    with open(filename, "r", encoding="ISO-8859-1") as fh:
        conversations_chunks = [line.split(" +++$+++ ") for line in fh]
    return [re.sub('[\[\]\']', '', el[3].strip()).split(", ") for el in
            conversations_chunks]
```


正如之前所说，记住使用正确的编码格式读取文件，否则会报错。函数的输出是包含列的列。每一个列都包含角色间对话的语言表达ID。然后，读取和解析`movie_lines.txt`文件，抽取语言表达的文本。记住这个文件的形式如下：
`L195 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ Well, I thought we'd start with pronunciation, if that's okay with you`.
这里，我们需要第一个和最后一个部分。

```python
def read_lines(storage_path, storage_dir):
    filename = storage_path + "/" + storage_dir + "/cornell movie-dialogs corpus/movie_lines.txt"
    with open(filename, "r", encoding="ISO-8859-1") as fh:
        lines_chunks = [line.split(" +++$+++ ") for line in fh]
    return {line[0]: line[-1].strip() for line in lines_chunks}
```

最后一步是关于分割和对齐。我们希望一个集合内的结果有两次语言表达的序列。这样，我们可以在给定第一次语言表达的情况下，训练聊天机器人，以便给出下一句话。最终，我们希望得到一个智能聊天机器人，可以回复多个问题。函数如下：

```python
def get_tokenized_sequencial_sentences(list_of_lines, line_text):
    for line in list_of_lines:
        for i in range(len(line) - 1):
            yield (line_text[line[i]].split(" "),
                   line_text[line[i+1]].split(" "))
```

它的输出是一个生成器，其包含两次语句表达（右边的跟着左边的）的元组。而且每次语言表达都用空格分割。

最后，我们可以把所有逻辑都封装在一个函数中，包括（如果本地没有的话）下载文件和解压文件，解析对话和文件行，格式化数据集为生成器。最终，我们可以把文件存在/tmp目录下：

```python
def retrieve_cornell_corpora(storage_path="/tmp",
                             storage_dir="cornell_movie_dialogs_corpus"):    download_and_decompress("http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip", storage_path, storage_dir)
    conversations = read_conversations(storage_path, storage_dir)
    lines = read_lines(storage_path, storage_dir)
    return tuple(zip(*list(get_tokenized_sequencial_sentences(conversations, 
                                                              lines))))
```

现在，我们的训练集看起来和机器翻译项目中的训练集很类似。事实上，它们不仅仅相似，而且目的相同，格式相同。因此我们可以使用之前章节中相同的代码。例如，`corpora_tools.py`文件可以不加修改直接用在这里（而且，它需要`data_utils.py`）。

有了这些文件，我们可以使用一些脚本检查聊天机器人的输入，再深入研究一下语料库。

我们可以使用之前章节中的`corpora_tools.py`查看语料库。获取康奈尔电影对话语料库，格式化语料，并打印出一个例子和它的长度：

```python
from corpora_tools import *
from corpora_downloader import retrieve_cornell_corpora 
sen_l1, sen_l2 = retrieve_cornell_corpora()
print("# Two consecutive sentences in a conversation")
print("Q:", sen_l1[0])
print("A:", sen_l2[0])
print("# Corpora length (i.e. number of sentences)")
print(len(sen_l1))
assert len(sen_l1) == len(sen_l2)
```

这些代码打印出两条被分割的连续语言表达的句子，以及数据集中的例子数量，超过22万：

```
# Two consecutive sentences in a conversation
Q: ['Can', 'we', 'make', 'this', 'quick?', '', 'Roxanne', 'Korrine', 'and',
'Andrew', 'Barrett', 'are', 'having', 'an', 'incredibly', 'horrendous',
'public', 'break-', 'up', 'on', 'the', 'quad.', '', 'Again.']
A: ['Well,', 'I', 'thought', "we'd", 'start', 'with', 'pronunciation,',
'if', "that's", 'okay', 'with', 'you.']
# Corpora length (i.e. number of sentences)
221616
```

清除句子中的停顿字符，把句子全部转成小写，并且限定每条句子最多20个单词（也就是说，对话序列只要包含超过20个单词的句子，就会被舍弃。）。后续的标准化分隔符如下：
```python
clean_sen_l1 = [clean_sentence(s) for s in sen_l1]
clean_sen_l2 = [clean_sentence(s) for s in sen_l2]
filt_clean_sen_l1, filt_clean_sen_l2 = filter_sentence_length(clean_sen_l1,
                                                              clean_sen_l2)
print("# Filtered Corpora length (i.e. number of sentences)")
print(len(filt_clean_sen_l1))
assert len(filt_clean_sen_l1) == len(filt_clean_sen_l2)
```


结果输出超过14万个例子：
```
# Filtered Corpora length (i.e. number of sentences)
140261
```


然后，创建两组句子集合的词典。实际中可以发现，两个词典看起来一样（因为出现在左边的句子也会出现在右边。），除非第一条句子和最后一条句子（都是只出现一次）使用了独特的词语。为了充分使用语料库，构建两个词典，然后使用词典对所有词语编码：
```python
dict_l1 = create_indexed_dictionary(filt_clean_sen_l1, dict_size=15000,
                                    storage_path="/tmp/l1_dict.p")
dict_l2 = create_indexed_dictionary(filt_clean_sen_l2, dict_size=15000,
                                    storage_path="/tmp/l2_dict.p")
idx_sentences_l1 = sentences_to_indexes(filt_clean_sen_l1, dict_l1)
idx_sentences_l2 = sentences_to_indexes(filt_clean_sen_l2, dict_l2)
print("# Same sentences as before, with their dictionary ID")
print("Q:", list(zip(filt_clean_sen_l1[0], idx_sentences_l1[0])))
print("A:", list(zip(filt_clean_sen_l2[0], idx_sentences_l2[0])))
```


打印出的结果如下。我们注意到词典包含1万5千个词语，但是并没有包含所有词语，而且超过1万6千个（出现较少的）词语并没有包含其中：
```
[sentences_to_indexes] Did not find 16823 words
[sentences_to_indexes] Did not find 16649 words
# Same sentences as before, with their dictionary ID
Q: [('well', 68), (',', 8), ('i', 9), ('thought', 141), ('we', 23), ("'",
5), ('d', 83), ('start', 370), ('with', 46), ('pronunciation', 3), (',',
8), ('if', 78), ('that', 18), ("'", 5), ('s', 12), ('okay', 92), ('with',
46), ('you', 7), ('.', 4)]
A: [('not', 31), ('the', 10), ('hacking', 7309), ('and', 23), ('gagging',
8761), ('and', 23), ('spitting', 6354), ('part', 437), ('.', 4), ('please',
145), ('.', 4)]
```

最后一步，给句子添加补全和标记信息：
```python
data_set = prepare_sentences(idx_sentences_l1, idx_sentences_l2,
                             max_length_l1, max_length_l2)
print("# Prepared minibatch with paddings and extra stuff")
print("Q:", data_set[0][0])
print("A:", data_set[0][1])
print("# The sentence pass from X to Y tokens")
print("Q:", len(idx_sentences_l1[0]), "->", len(data_set[0][0]))
print("A:", len(idx_sentences_l2[0]), "->", len(data_set[0][1]))
```


和预期的一样，打印如下：
```
# Prepared minibatch with paddings and extra stuff
Q: [0, 68, 8, 9, 141, 23, 5, 83, 370, 46, 3, 8, 78, 18, 5, 12, 92, 46, 7, 4]
A: [1, 31, 10, 7309, 23, 8761, 23, 6354, 437, 4, 145, 4, 2, 0, 0, 0, 0, 0,
0, 0, 0, 0]
# The sentence pass from X to Y tokens
Q: 19 -> 20
A: 11 -> 22
```

### 训练聊天机器人

处理完语料库后，我们可以开始构建模型。这个项目需要一个序列生成的模型，因此我们可以使用RNN模型。而且，我们可以重用上个项目的代码：只需要修改一下数据集构建的方法，和模型的参数。然后复制上一章中的训练脚本，修改`build_dataset`函数，方便使用康奈尔的数据集。

需要注意的是，本章使用的数据集比上一章的要大，因此我们需要限制语料库大小至几万行。在一台RAM为8G的4年老笔记本上，我们需要用前3万行，否则程序会耗尽内存，一直交换空间。使用少量例子的另一个好处是，字典也会变小，每个不超过1万个词语。

```python
def build_dataset(use_stored_dictionary=False):
    sen_l1, sen_l2 = retrieve_cornell_corpora()
    clean_sen_l1 = [clean_sentence(s) for s in sen_l1][:30000] ### OTHERWISE IT DOES NOT RUN ON MY LAPTOP
    clean_sen_l2 = [clean_sentence(s) for s in sen_l2][:30000] ### OTHERWISE IT DOES NOT RUN ON MY LAPTOP
    filt_clean_sen_l1, filt_clean_sen_l2 =
    filter_sentence_length(clean_sen_l1, clean_sen_l2, max_len=10)
    if not use_stored_dictionary:
        dict_l1 = create_indexed_dictionary(filt_clean_sen_l1,
                                            dict_size=10000, 
                                            torage_path=path_l1_dict)
        dict_l2 = create_indexed_dictionary(filt_clean_sen_l2,
                                            dict_size=10000, 
                                            torage_path=path_l2_dict)
    else:
        dict_l1 = pickle.load(open(path_l1_dict, "rb"))
        dict_l2 = pickle.load(open(path_l2_dict, "rb"))
    dict_l1_length = len(dict_l1)
    dict_l2_length = len(dict_l2)
    idx_sentences_l1 = sentences_to_indexes(filt_clean_sen_l1, dict_l1)
    idx_sentences_l2 = sentences_to_indexes(filt_clean_sen_l2, dict_l2)
    max_length_l1 = extract_max_length(idx_sentences_l1)
    max_length_l2 = extract_max_length(idx_sentences_l2)
    data_set = prepare_sentences(idx_sentences_l1, idx_sentences_l2,
                                 max_length_l1, max_length_l2)
    return (filt_clean_sen_l1, filt_clean_sen_l2), \ 
            data_set, \
            (max_length_l1, max_length_l2), \
            (dict_l1_length, dict_l2_length)
```


把这个函数放在（上一章定义的）`train_translator.py`文件中，并重命名为文件`train_chatbot.py`，然后我们可以开始训练聊天机器人。

几次迭代后，读者可以终止程序，并得到类似以下的输出：

```
[sentences_to_indexes] Did not find 0 words
[sentences_to_indexes] Did not find 0 words
global step 100 learning rate 1.0 step-time 7.708967611789704 perplexity
444.90090078460474
eval: perplexity 57.442316329639176
global step 200 learning rate 0.990234375 step-time 7.700247814655302
perplexity 48.8545568311572
eval: perplexity 42.190180314697045
global step 300 learning rate 0.98046875 step-time 7.69800933599472
perplexity 41.620538109894945
eval: perplexity 31.291903031786116
...
...
...
global step 2400 learning rate 0.79833984375 step-time 7.686293318271639
perplexity 3.7086356605442767
eval: perplexity 2.8348589631663046
global step 2500 learning rate 0.79052734375 step-time 7.689657487869262
perplexity 3.211876894960698
eval: perplexity 2.973809378544393
global step 2600 learning rate 0.78271484375 step-time 7.690396382808681
perplexity 2.878854805600354
eval: perplexity 2.563583924617356
```

如果改变参数，读者可以在不同的混淆度下终止程序。要得到最终结果，我们可以设置RNN的大小为256，层数为2，批大小为128个样本，学习率为1.0。

现在，聊天机器人可以接受测试了。尽管我们可以使用上一章`test_translator.py`中的相同代码进行测试，但是我们还是希望可以完成一个更加精细的方案，也就是让聊天机器人通过API作为服务发布。

### 聊天机器人API

首先，我们需要一个web框架，提供API。在这个项目中，我们选择Botttle，一种非常易用的轻量化简易框架。

> 要安装这个程序包，在命令行中运行`pip install bottle`。要获取更多信息和代码，可以查看项目网页：https://bottlepy.org。

现在，创建函数，解析用户提供的任意一个句子参数。以下所有代码应该放在`test_chatbot_aas.py`文件中。首先引入一些库和函数，便于借助词典清洗，分隔，准备句子：

```python
import pickle
import sys
import numpy as np
import tensorflow as tf
import data_utils
from corpora_tools import clean_sentence, sentences_to_indexes, prepare_sentences
from train_chatbot import get_seq2seq_model, path_l1_dict, path_l2_dict
model_dir = "/home/abc/chat/chatbot_model"
def prepare_sentence(sentence, dict_l1, max_length):
    sents = [sentence.split(" ")]
    clean_sen_l1 = [clean_sentence(s) for s in sents]
    idx_sentences_l1 = sentences_to_indexes(clean_sen_l1, dict_l1)
    data_set = prepare_sentences(idx_sentences_l1, [[]], max_length, max_length)
    sentences = (clean_sen_l1, [[]])
return sentences, data_set
```


函数`prepare_sentence`的功能如下：

- 分隔输入的句子
- 清洗句子（转成小写，清除停顿符号）
- 把分隔结果转换成字典ID列
- 添加标记和补全，以保证默认的长度

接着，我们需要一个函数把预测的序列ID转换成实际的词语序列。这个任务由函数`decode`完成，可以根据输入的句子和softmax函数预测最可能的输出。最后，代码返回不带补全和标记的句子（更全面的函数介绍在上一章）：

```python
def decode(data_set):
    with tf.Session() as sess:
        model = get_seq2seq_model(sess, True, dict_lengths, max_sentence_lengths,
                                  model_dir)
        model.batch_size = 1
        bucket = 0
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            {bucket: [(data_set[0][0], [])]}, bucket)
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket, True)
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        if data_utils.EOS_ID in outputs:
            outputs = outputs[1:outputs.index(data_utils.EOS_ID)]
            tf.reset_default_graph()
     return " ".join([tf.compat.as_str(inv_dict_l2[output]) for output in outputs])
```
最终，主函数，即脚本中的函数：

```python
if __name__ == "__main__":
    dict_l1 = pickle.load(open(path_l1_dict, "rb"))
    dict_l1_length = len(dict_l1)
    dict_l2 = pickle.load(open(path_l2_dict, "rb"))
    dict_l2_length = len(dict_l2)
    inv_dict_l2 = {v: k for k, v in dict_l2.items()}
    max_lengths = 10
    dict_lengths = (dict_l1_length, dict_l2_length)
    max_sentence_lengths = (max_lengths, max_lengths)
    from bottle import route, run, request
    @route('/api')
    def api():
        in_sentence = request.query.sentence
        _, data_set = prepare_sentence(in_sentence, dict_l1, max_lengths)
        resp = [{"in": in_sentence, "out": decode(data_set)}]
        return dict(data=resp)
    run(host='127.0.0.1', port=8080, reloader=True, debug=True)
```

首先，它加载词典，并准备逆序词典。然后，使用Bottle API创建一个（基于/api的URL） HTTP GET端点。当端点接收到HTTP GET请求时，路径装饰器可以设置和丰富了函数。在这种情况下，`api()` 开始运行，它首先把句子读取为HTTP参数，然后调用`prepare_sentence`函数，最后运行解码程序。返回的结果是一个包含用户输入句子和机器人回答的词典。

最后，网络服务器打开，本地端口为port 8080。使用Bottle构建聊天机器人服务就是这么简单！

现在运行服务，检查输出。在命令行中运行：

```shell
$> python3 –u test_chatbot_aas.py
```

然后，开始使用一些通用问题询问聊天机器人，我们可以使用CURL，这一简单的命令行完成这个操作。同时所有的浏览器都是可用的。只需注意URL应该经过编码，例如，空格符应该使用编码格式`%20`替代。

CURL提供简单的方法编码URL请求，可以让所有操作变得更简单。下面是两个例子：

```shell
$> curl -X GET -G http://127.0.0.1:8080/api --data-urlencode "sentence=how are you?"
{"data": [{"out": "i ' m here with you .", "in": "where are you?"}]}
$> curl -X GET -G http://127.0.0.1:8080/api --data-urlencode "sentence=are you here?"
{"data": [{"out": "yes .", "in": "are you here?"}]}
$> curl -X GET -G http://127.0.0.1:8080/api --data-urlencode "sentence=are you a chatbot?"
{"data": [{"out": "you ' for the stuff to be right .", "in": "are you a chatbot?"}]}
$> curl -X GET -G http://127.0.0.1:8080/api --data-urlencode "sentence=what is your name ?"
{"data": [{"out": "we don ' t know .", "in": "what is your name ?"}]}
$> curl -X GET -G http://127.0.0.1:8080/api --data-urlencode "sentence=how are you?"
{"data": [{"out": "that ' s okay .", "in": "how are you?"}]}
```


> 如果浏览器没有反应，可以试试编码URL，例如：
> ```shell
> $> curl -X GET
> http://127.0.0.1:8080/api?sentence=how%20are%20you?
> {"data": [{"out": "that ' s okay .", "in": "how are you?"}]}.
> ```

回复也很有趣；务必注意，我们是使用电影数据训练的聊天机器人，所以回复的风格也是这类的。

要关闭网络服务器，使用`Ctrl + C`。

#### 课后作业

下面是课后作业：

* 你可以使用JS创建一个简单网页来请求聊天机器人吗？
* 因特网上有许多其他的训练集；试着看看模型之间答案的差异。哪一个数据集最适合客服机器人？
* 你可以修改模型，作为服务训练吗？也就是说，使用HTTP GET/POST传递句子？

### 小结

在本章汇总，我们已经实现了一个聊天机器人，可以通过HTTP端点和GET API回答问题。这是我们使用RNN所完成的有一个很棒的例子。在下一章中，我们会讨论另一个话题：如何使用TensorFlow创建推荐系统。