## 第6章 构建和训练机器翻译系统

本项目的目标是训练一个**人工智能**（**  artificial intelligence**, AI）模型，以用于在两种语言之间进行翻译。具体的，将德语句子自动地翻译成英语。同时，本章所构建的模型可以适用于任何两种语言的互译。

本章包括以下四个部分：

* 机器翻译系统架构
* 语料库预处理
* 机器翻译模型训练
* 测试及翻译

上述每个部分都是本章的关键点，按照上述顺序讲解有助于帮助读者在脑中形成清晰的画面。

###机器翻译系统架构

一个机器翻译系统可以接收一种语言的任意字符串作为输入，并将其翻译成另一种语言中具有同样意思的语句。例如，谷歌翻译（其他IT公司也开发了自己的机器翻译系统）。在这些网页上，用户可以在100多种语言间进行选择并进行翻译。这些网页的使用也非常简单：在左边的输入框中输入你想翻译的句子（例如，你好世界），然后选择你输入句子的语言（例如，英语），最后选择你想要翻译成的语言即可。

下面是一个把“Hello word” 翻译成德语的例子：

![](figures\141_1.jpg)

是不是很简单呢？乍一看，读者可能认为这是一个简单的字典替换。单词可以被分块后在特定的英法词典上查找，用每个词的翻译来代替。可是，事实并非如此。在这个例子中，英语句子有两个单词，而法语句子有三个单词。更一般地，一些短语动词（例如，turn up、turn off、turn on、turndown等）、撒克逊属格、语法性别、时态、条件句等等，它们并不总是能够直接被翻译，正确的翻译往往需要考虑句子的上下文。

这也解释了为什么在做机器翻译时，往往需要人工智能工具。
具体地，对于很多**自然语言处理**（**natural language processing**， NLP）任务，往往需要用到**递归神经网络**（**recurrent neural networks**， RNNs）。上一章已经介绍了RNNs，此模型的最主要特征是适用于序列数据，即，输入一个序列可以输出一个序列。本章的目的是构建一个正确的模型，给定一个句子作为输入序列，将其对应的翻译作为输出。读者需谨记，`no free lunch`定理：这个过程并不容易，许多解决方式会有相同的结果。本章会给读者展示一种简单有效的解决方式。

首先，本章从语料库开始：语料库可能是最难找的，因为它需要包含许多句子，并保证从一种语言到另一种语言的翻译正确。
幸运的是，一个NLP方面著名的Python包，NLTK，包含了Comtrans语料库。Comtrans是**combination approach to machine translation**的缩写，包含了德语、法语和英语语料的对齐。

本项目用这些语料的原因如下：

 	1. 方便下载，且方便在Python中引入。
	2. 不需要为了从硬盘或互联网读入语料库而写函数。NLTK已将此步封装成了函数。
	3. 它所需空间小，可以在许多笔记本电脑上使用。
	4. 可以从互联网上免费下载。

> 读者如果需要查询Comtrans项目的更多信息，可以登录http://www.fask.uni-mainz.de/user/rapp/comtrans/ 。

更具体的，本章会构建一个机器翻译系统，以将德语翻译成英语。本章在Comtrans语料库的所有语言中随机的挑选了两种语言（读者也可自行选择语言种类）。本项目所使用的模型可以适用于任何两种语言的互译。

读者可以输入以下一些命令来查看语料库的格式：

```python
from nltk.corpus import comtrans
print(comtrans.aligned_sents('alignment-de-en.txt')[0])
```
输出如下：
<AlignedSent: 'Wiederaufnahme der S...' -> 'Resumption of the se...'>

调用函数`aligned_sents`可以看到一对对的句子。文件名中包含要翻译的原始语言名和目标语言名。此项目接下来的内容是将德语（*de*）翻译成英语（*en*）。函数返回的对象是`nltk.translate.api.AlignedSent`类的一个实例。说明文档中显示，第一种语言可以通过类的属性`words`获得，而第二种语言则通过属性`mots`获得。所以，为了把德语句子和其英语翻译分别抽出，需要执行以下代码：
```python
print(comtrans.aligned_sents()[0].words)
print(comtrans.aligned_sents()[0].mots)
```
代码输出如下：
['Wiederaufnahme', 'der', 'Sitzungsperiode']
['Resumption', 'of', 'the', 'session']

读者可以看到，句子已经被标记好了，并且看起来像是序列数据。实际上，这些句子可以被当成RNN的输入和输出，而这个RNN模型可以为将德语句子翻译成英语。

此外，如果读者想要理解语言的动态性，Comtrans可以提供翻译时单词的对齐方式：
```python	
print(comtrans.aligned_sents()[0].alignment)
```
代码输出为：
	0-0 1-1 1-2 2-3
德语中的第一个单词对应到英语里的第一个单词（*Wiederaufnahme*对应*Resumption*），第二个单词对应到第二和三个单词（*der*对应*of* 和*the*），第三个单词对应到第四个单词（*Sitzungsperiode*对应*session*）

###对语料库进行预处理
第一步是对语料库进行检索。在上一节已经为读者介绍如何操作，接下来要把这些操作封装成一个正式的函数。为了方便其他脚本调用，将其封装到`corpora_tools.py`中。

  1.为方便后续使用，先引入一些包：

```python
import pickle
import re
from collections import Counter
from nltk.corpus import comtrans
```
  2. 构建函数以方便在语料库中检索：
```python
def retrieve_corpora(translated_sentences_l1_l2='alignment-de-en.txt'): 
	print("Retrieving corpora: {}".format(translated_sentences_l1_l2))
	als = comtrans.aligned_sents(translated_sentences_l1_l2)
	sentences_l1 = [sent.words for sent in als]
	sentences_l2 = [sent.mots for sent in als]
	return sentences_l1, sentences_l2
```
此函数的参数是一个文件名：即，包含NLTK COMTRANS语料库对齐语句的文件。函数返回两个句子列表（实际是个包含一系列标记的句子列表），其中一个是源语言（本例中是德语），另一个是需要翻译的目标语言（本例子中是英语）。
  3. 在Python RePL上对此函数进行测试：
```python
sen_l1, sen_l2 = retrieve_corpora()
print("# A sentence in the two languages DE & EN")
print("DE:", sen_l1[0])
print("EN:", sen_l2[0])
print("# Corpora length (i.e. number of sentences)")
print(len(sen_l1))
assert len(sen_l1) == len(sen_l2)
```
  4. 输出如下：
```
Retrieving corpora: alignment-de-en.txt
# A sentence in the two languages DE & EN
DE: ['Wiederaufnahme', 'der', 'Sitzungsperiode']
EN: ['Resumption', 'of', 'the', 'session']
# Corpora length (i.e. number of sentences)
33334
```
  本例也打印了每个语料库中的句子数目，并断言源语言与目标语言的句子书相等。

    5. 接下来，需要清空标记。具体地，需要标记标点符号，并小写所有标记。为了实现这一步，本章会在`corpora_tools.py`中创建一个新的函数。用`regex`模块来对标记进行分割。

```python
def clean_sentence(sentence):
	regex_splitter = re.compile("([!?.,:;$\"')( ])")
	clean_words = [re.split(regex_splitter, word.lower()) for word in sentence]
	return [w for words in clean_words for w in words if words if w]
```
  6. 在REPL中测试这个函数：
```python
clean_sen_l1 = [clean_sentence(s) for s in sen_l1]
clean_sen_l2 = [clean_sentence(s) for s in sen_l2]
print("# Same sentence as before, but chunked and cleaned")
print("DE:", clean_sen_l1[0])
print("EN:", clean_sen_l2[0])
```
此代码的输出与之前相同的句子，但是句子被分块、被清理了标记：
```
DE: ['wiederaufnahme', 'der', 'sitzungsperiode']
EN: ['resumption', 'of', 'the', 'session']
```
接下来，需要过滤掉特别长的句子，这些句子不好处理。因为本章的目标是保证模型可以在本地机器上运行，所以应该将句子长度限制为*N*个标记。所以，设置*N*＝20，以便可以在24小时之内完成模型训练。如果读者电脑比较给力，可以将*N*设置的大一些。为了使函数足够通用，默认句子长度设置为0，即，空标记集。

  1. 函数的逻辑很简单：如果句子的标记长度或其翻译的标记长度大于*N*，则这个句子和其对应的翻译会被删除：
```python
def filter_sentence_length(sentences_l1, sentences_l2, min_len=0, max_len=20):
	filtered_sentences_l1 = []
	filtered_sentences_l2 = []
	for i in range(len(sentences_l1)):
		if min_len <= len(sentences_l1[i]) <= max_len and min_len <= len(sentences_l2[i]) <= max_len:
			filtered_sentences_l1.append(sentences_l1[i])
			filtered_sentences_l2.append(sentences_l2[i])
	return filtered_sentences_l1, filtered_sentences_l2
```
  同样的，在REPL中看一下经过过滤后还剩下多少句子。初始有33000多个句子：
```python
filt_clean_sen_l1, filt_clean_sen_l2 =filter_sentence_length(clean_sen_l1, clean_sen_l2)
print("# Filtered Corpora length (i.e. number of sentences)")
print(len(filt_clean_sen_l1))
assert len(filt_clean_sen_l1) == len(filt_clean_sen_l2)
```
  输出如下：
```
# Filtered Corpora length (i.e. number of sentences)
14788
```
过滤后还剩下将近15000个句子，基本上是原语料库的一半。

接下来，为了人工智能模型使用，需要将文本转化成数字。为了实现这一目标，本章为每种语言都创建了一个词典。这个词典需要包含大部分的单词，因为本章会丢弃一些低频词汇。在tf-idf（一个文档中的词条词频乘以它的逆文档频率，即，这个词条在多少个文档中出现过）中这也是常见的做法。丢弃罕见的词汇可以加速计算，并使得结果更具有可扩展性和通用性。本章还需要在每个词典中设置四个特殊的标记：
  1. 用一种符号来表示填充（稍后会进行解释）
  2. 用一种符号来分隔开两个句子
  3. 用一种符号来标识句子结尾
  4. 用一种符号来标记没见过的单词（例如很罕见的单词）

本章创建了一个新的脚本`data_utils.py`，并将下面代码写入此脚本中：
```python
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
OP_DICT_IDS = [PAD_ID, GO_ID, EOS_ID, UNK_ID]
```
在`corpora_tools.py`文件中添加下面的函数：
```python
import data_utils
def create_indexed_dictionary(sentences, dict_size=10000, storage_path=None):
	count_words = Counter()
	dict_words = {}
	opt_dict_size = len(data_utils.OP_DICT_IDS)
	for sen in sentences:
		for word in sen:
			count_words[word] += 1
	dict_words[data_utils._PAD] = data_utils.PAD_ID
	dict_words[data_utils._GO] = data_utils.GO_ID
	dict_words[data_utils._EOS] = data_utils.EOS_ID
	dict_words[data_utils._UNK] = data_utils.UNK_ID
	  
	for idx, item in enumerate(count_words.most_common(dict_size)):
		dict_words[item[0]] = idx + opt_dict_size
	if storage_path:
		pickle.dump(dict_words, open(storage_path, "wb"))
	return dict_words
```
此函数参数为用于构建词典的句子列表、词典大小，以及词典的存储路径。词典是在训练模型时构建的，测试阶段只加载模型，且需要保证相关联的标记或标识与训练时一致。如果独立的标记数超过所设置的词典大小，则需要保留出现次数较多的标记。最后，词典包含的是原始标记到其ID的映射关系。

构建好词典后，需要将标记替换为其ID，函数如下：

```python	
def sentences_to_indexes(sentences, indexed_dictionary):
	indexed_sentences = []
	not_found_counter = 0
	for sent in sentences:
		idx_sent = []
		for word in sent:
			try:
				idx_sent.append(indexed_dictionary[word])
			except KeyError:
				idx_sent.append(data_utils.UNK_ID)
				not_found_counter += 1
		indexed_sentences.append(idx_sent)
	print('[sentences_to_indexes] Did not find {} words'.format(not_found_counter))
	return indexed_sentences
```
 这一步很简单，用ID替换标记。如果标记不在词典里，就用unknown标记的ID替换。读者可在REPL中查看，经过上述步骤后，句子变成了什么样：
```python
dict_l1 = create_indexed_dictionary(filt_clean_sen_l1, dict_size=15000, storage_path="/tmp/l1_dict.p")
dict_l2 = create_indexed_dictionary(filt_clean_sen_l2, dict_size=10000, storage_path="/tmp/l2_dict.p")
idx_sentences_l1 = sentences_to_indexes(filt_clean_sen_l1, dict_l1)
idx_sentences_l2 = sentences_to_indexes(filt_clean_sen_l2, dict_l2)
print("# Same sentences as before, with their dictionary ID")
print("DE:", list(zip(filt_clean_sen_l1[0], idx_sentences_l1[0])))
```
这段代码将句子的标记和其对应的ID一起打印出来。在RNN中需要用到的只是二元组中的第二个元素，即，标记的ID:

```
# Same sentences as before, with their dictionary ID
DE: [('wiederaufnahme', 1616), ('der', 7), ('sitzungsperiode', 618)]
EN: [('resumption', 1779), ('of', 8), ('the', 5), ('session', 549)]
```
读者注意，出现频繁的标记，例如英语中的*the*、*of* 和德语中*der*，ID排序会很靠后。即，ID是按照其用法的广泛程度来排序的 （读者可以查看`create_indexed_dictionary`函数的函数体）。

另外，句子的长度做限制的同时需要创建一个函数来记录句子的最大长度。如果读者的机器特别给力，则不需要对句子长度做限制，只需要记录输入给RNN的句子的最大长度即可。
```python
def extract_max_length(corpora):
	return max([len(sentence) for sentence in corpora])
```
对句子执行以下的代码：
```python
max_length_l1 = extract_max_length(idx_sentences_l1)
max_length_l2 = extract_max_length(idx_sentences_l2)
print("# Max sentence sizes:")
print("DE:", max_length_l1)
print("EN:", max_length_l2)
```
输出如下：
```
# Max sentence sizes:
DE: 20
EN: 20
```
最后的预处理步骤是填充。由于输入给RNN的句子需要有相同的长度，故需要对较短的句子进行填充。同样的，为了让RNN知道句子的开头和结尾，本章需要插入开头和结尾的标记。

总的来说，步骤如下：

* 填充输入序列，使得所有序列包含20个标记。

* 填充输出序列，使得所有序列包含20个标记。

* 在每个输出序列的开头插入`_GO`标记，在结尾插入一个`_EOS`，来定位翻译的开始和结束。

代码如下（保存到`corpora_tools.py`中）：
```python
def prepare_sentences(sentences_l1, sentences_l2, len_l1, len_l2):
	assert len(sentences_l1) == len(sentences_l2)
	data_set = []
	for i in range(len(sentences_l1)):
		padding_l1 = len_l1 - len(sentences_l1[i])         
		pad_sentence_l1 = ([data_utils.PAD_ID]*padding_l1) + sentences_l1[i]
		padding_l2 = len_l2 - len(sentences_l2[i])
		pad_sentence_l2 = [data_utils.GO_ID] + sentences_l2[i] + [data_utils.EOS_ID] + ([data_utils.PAD_ID] * padding_l2)
		data_set.append([pad_sentence_l1, pad_sentence_l2])
	return data_set
```
用数据集中第一个句子进行测试：
```python
data_set = prepare_sentences(idx_sentences_l1, idx_sentences_l2, max_length_l1, max_length_l2)
print("# Prepared minibatch with paddings and extra stuff")
print("DE:", data_set[0][0])
print("EN:", data_set[0][1])
print("# The sentence pass from X to Y tokens")
print("DE:", len(idx_sentences_l1[0]), "->", len(data_set[0][0]))
print("EN:", len(idx_sentences_l2[0]), "->", len(data_set[0][1]))
```
输出如下：
```
# Prepared minibatch with paddings and extra stuff
DE: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1616, 7, 618]
EN: [1, 1779, 8, 5, 549, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# The sentence pass from X to Y tokens
DE: 3 -> 20
EN: 4 -> 22
```
读者可以看到，输入和输出都用0填充成了固定的长度（0对应于词典中的`_PAD`），在输出句子的开头和结尾之前标记了1和2。文献已有证明，在输入句子的开头和输出句子的结尾进行填充，效果比较好。进行填充步骤后，输入句子长度为20个标记，输出句子长度为22个标记。

###训练机器翻译模型
目前为止已经讲述了如何对语料库进行预处理，接下来要讲述如何运用模型。模型已经训练好了，读者可直接从https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/seq2seq_model.py 进行下载。
> 上述代码片段以Apache 2.0方式授权。十分感谢将此模型开源的作者。We really thank the authors for having open sourced such a great model. Copyright 2015 The TensorFlow Authors. All Rights Reserved. Licensed under the Apache License, Version 2.0 (the License); 未经授权不得使用。读者可以从http://www.apache.org/licenses/LICENSE-2.0 获取授权副本。
Unless required by applicable law or agreed to in writing, software.
Distributed under the License is distributed on an AS IS BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
express or implied. See the License for the specific language governing permissions and limitations under the License.

本节会为读者讲解如何使用模型。首先需要创建一个名为`train_translator.py`的新文件，然后引入一些包，并设置一些常量。本章会把词典、模型、检查点保存在`/tmp/`目录中：

```python
import time
import math
import sys
import pickle
import glob import os
import tensorflow as tf
from seq2seq_model import Seq2SeqModel
from corpora_tools import *
  
path_l1_dict = "/tmp/l1_dict.p"
path_l2_dict = "/tmp/l2_dict.p"
model_dir = "/tmp/translate "
model_checkpoints = model_dir + "/translate.ckpt"
```
接下来，写一个函数，把之前小节中创建的工具都应用起来，给定一个布尔类型的标志，此函数会返回语料库。具体来说，如果参数是`False`，函数会从零开始建立词典（并保存）；否则，函数会直接从路径中读取词典：
```python
def build_dataset(use_stored_dictionary=False):
	sen_l1, sen_l2 = retrieve_corpora()
	clean_sen_l1 = [clean_sentence(s) for s in sen_l1]
	clean_sen_l2 = [clean_sentence(s) for s in sen_l2]
	filt_clean_sen_l1, filt_clean_sen_l2 = filter_sentence_length(clean_sen_l1, clean_sen_l2)
	if not use_stored_dictionary:
		dict_l1 = create_indexed_dictionary(filt_clean_sen_l1, dict_size=15000, storage_path=path_l1_dict)
		dict_l2 = create_indexed_dictionary(filt_clean_sen_l2, dict_size=10000, storage_path=path_l2_dict)
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
    return (filt_clean_sen_l1, filt_clean_sen_l2), data_set, (max_length_l1, max_length_l2),(dict_l1_length, dict_l2_length)
```
上述函数返回的结果包括：经过清洗的句子、数据集、句子的最大长度，以及词典的大小。

同样的，本章还需要有一个函数来清理模型。每次重新训练的时候，需要清空存放模型的目录。用以下简单的代码可以实现这一功能：
```python
def cleanup_checkpoints(model_dir, model_checkpoints):
	for f in glob.glob(model_checkpoints + "*"):
		os.remove(f)
	try:
		os.mkdir(model_dir)
	except FileExistsError:
		pass
```
最后为读者展示如何以可重用的方式创建模型：
```python
def get_seq2seq_model(session, forward_only, dict_lengths, max_sentence_lengths, model_dir):
	model = Seq2SeqModel(
			source_vocab_size=dict_lengths[0], 
			target_vocab_size=dict_lengths[1], 
			buckets=[max_sentence_lengths],
			size=256, 
			num_layers=2, 
			max_gradient_norm=5.0, 
			batch_size=64, 
			learning_rate=0.5, 
			learning_rate_decay_factor=0.99, 
			forward_only=forward_only, 
			dtype=tf.float16)
	ckpt = tf.train.get_checkpoint_state(model_dir)
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):   
		print("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
		model.saver.restore(session, ckpt.model_checkpoint_path)     
	else:         
		print("Created model with fresh parameters.")         
		session.run(tf.global_variables_initializer())     
	return model
```
此函数调用模型构建函数，并传入以下参数：
* 源语言的词典大小（本例中的德语）
* 目标语言的词典大小（本例中的英语）
* buckets（由于已将所有的序列都填充成为了固定的长度，故本例中buckets中只有一组数值）
* LSTM内部神经元个数
* LSTM堆叠层数
* 梯度的最大模（用于梯度裁剪）
* mini-batch大小（即，在训练中每次迭代需要用多少观测数据）
* 学习率
* 学习率衰减因子
* 模型的方向
* 数据的类型（本例中用float16类型，即，一个float16类型字符占用2个字节）

为了使模型训练的又快又好，本章直接在代码中对上述参数的值进行了设置，读者可以修改这些值来观察模型的性能变化。
上述函数中最后一个if/else的作用是：如果模型已经存在，则直接从从检查点检索模型。实际上，在测试中的解码阶段，此函数也会做检索或构建模型的操作。

最后，训练机器翻译模型的函数如下：
```python
def train():
	with tf.Session() as sess:
		model = get_seq2seq_model(sess, False, dict_lengths, max_sentence_lengths, model_dir)
		# This is the training loop.
		step_time, loss = 0.0, 0.0
		current_step = 0
		bucket = 0
		steps_per_checkpoint = 100
		max_steps = 20000
		while current_step < max_steps:
			start_time = time.time()
			encoder_inputs, decoder_inputs, target_weights = model.get_batch([data_set], bucket)
			_, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket, False)
			step_time += (time.time() - start_time)/steps_per_checkpoint
			loss += step_loss / steps_per_checkpoint
			current_step += 1
			if current_step % steps_per_checkpoint == 0:
				perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
				print ("global step {} learning rate {} step-time {} perplexity {}".format( model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
				sess.run(model.learning_rate_decay_op)                 
				model.saver.save(sess, model_checkpoints, global_step=model.global_step)
				step_time, loss = 0.0, 0.0
				encoder_inputs, decoder_inputs, target_weights = model.get_batch([data_set], bucket)
				_, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket, True)
				eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300
else float("inf")
				print(" eval: perplexity {}".format(eval_ppx))           
				sys.stdout.flush()
```
此函数从构建模型开始。此外，为迭代过程中的每个检查点设置一些常量，并设置最大迭代次数。具体地，上述代码做到每100步保存一次模型， 并设置最大迭代次数为20000次。如果程序耗时太久，读者可以自行中断此程序：每个接差点都包含了训练好的模型，解码时可以运用最新的模型版本。

接下来为读者讲解每个循环中的内容。在每个循环中，模型会获取小批量的数据（之前设置的是64）。`get_batch`方法返回的对象是输入数据（即，源序列数据）、输出数据（目标序列数据）、以及模型的权重。 	`step`方法会对模型进行一次迭代。此方法会返回当前小批量数据上的模型的损失情况。模型训练就是这么简单！

平均困惑度（越低越好）可以用来衡量模型每100次迭代后表现如何，而存储检查点可以保存模型。困惑度可以用于衡量预测数据的不确定程度：越能确定下一个标记是什么，输出语句的困惑度就越低。 此外，重置计数器后，在小批量测试集数据（在本例中，直接对数据集进行随机小批量采样）上用同样的标准来度量，并打印其性能。然后，进行下一次的迭代。

作为改进，本例用学习率衰减因子，每100次迭代后会对学习率进行衰减。本例中的衰减因子是0.99。 这使得训练过程能更快收敛，结果也更加稳定。

接下来是将所有的函数联系起来。本例脚本中创建的`main`，使得此脚本可以直接在命令行中进行调用，也可以被其他脚本所引入:
```python
if __name__ == "__main__":
	_, data_set, max_sentence_lengths, dict_lengths = build_dataset(False)
	cleanup_checkpoints(model_dir, model_checkpoints)
	train()
```
在控制台中，读者可以使用非常简单的命令来训练机器翻译系统：
```shell
$> python train_translator.py
```
在没有NVIDIA GPU的普通笔记本上，如果想要困惑度低于10，模型训练会需要一天的时间（多余12小时），输出如下：
```shell
Retrieving corpora: alignment-de-en.txt
[sentences_to_indexes] Did not find 1097 words
[sentences_to_indexes] Did not find 0 words
Created model with fresh parameters. 
global step 100 learning rate 0.5 step-time 4.3573073434829713 perplexity
526.6638556683066
eval: perplexity 159.2240770935855
[...]
global step 10500 learning rate 0.180419921875 step-time
4.35106209993362414 perplexity 2.0458043055629487
eval: perplexity 1.8646006006241982 
[...]
```
###测试和翻译
翻译的代码在`test_translator.py`中。
首先引入一些包，并设置已训练好的模型的路径：
```python
import pickle
import sys
import numpy as np
import tensorflow as tf
import data_utils
from train_translator import (get_seq2seq_model, path_l1_dict, path_l2_dict, build_dataset)
model_dir = "/tmp/translate"
```
接下来，构建一个函数来解码RNN生成的输出序列。读者请注意，序列是多维的，每一维代表着一个单词，它的值代表这个单词的概率。故我们需要挑选概率最大的那个单词。在反向词典的帮助下，可以很轻易的找出实际单词是什么。最后，去掉标志位（字符串中填充、开始及结束的标志）后将结果输出。

本例对训练集中的前五个句子进行解码，这五个句子都是直接从原始语料库中抽出，读者可在句子中插入新的字符串，或者用其他的语料库进行测试：

```python
def decode():
	with tf.Session() as sess:
		model = get_seq2seq_model(sess, True, dict_lengths, max_sentence_lengths, model_dir)
		model.batch_size = 1
		bucket = 0
		for idx in range(len(data_set))[:5]:             
			print("-------------------")
			print("Source sentence: ", sentences[0][idx])
			print("Source tokens: ", data_set[idx][0])             
			print("Ideal tokens out: ", data_set[idx][1])             
			print("Ideal sentence out: ", sentences[1][idx])            
			encoder_inputs, decoder_inputs, target_weights =
model.get_batch(bucket: [(data_set[idx][0], [])]}, bucket)
			_, _, output_logits = model.step(sess, encoder_inputs,
decoder_inputs,target_weights, bucket, True)
			outputs = [int(np.argmax(logit, axis=1)) for logit in
output_logits]
			if data_utils.EOS_ID in outputs:
				outputs = outputs[1:outputs.index(data_utils.EOS_ID)]
			print("Model output: ", "".join([tf.compat.as_str(inv_dict_l2[output]) for output in outputs]))
			sys.stdout.flush()
```
这里仍需要一个`main`，以便在命令行中调用：
```python
if __name__ == "__main__":     
	dict_l2 = pickle.load(open(path_l2_dict, "rb"))     
	inv_dict_l2 = {v: k for k, v in dict_l2.items()}
	build_dataset(True)
	sentences, data_set, max_sentence_lengths, dict_lengths =build_dataset(False)
	try:         
		print("Reading from", model_dir)
		print("Dictionary lengths", dict_lengths)
		print("Bucket size", max_sentence_lengths)
	except NameError:         
		print("One or more variables not in scope. Translation not possible")         
		exit(-1)     
	decode()
```
运行上述代码后，结果如下：
```
Reading model parameters from /tmp/translate/translate.ckpt-10500
-------------------
Source sentence: ['wiederaufnahme', 'der', 'sitzungsperiode']
Source tokens: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1616, 7,618]
Ideal tokens out: [1, 1779, 8, 5, 549, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Ideal sentence out: ['resumption', 'of', 'the', 'session']
Model output: resumption of the session
-------------------
Source sentence: ['ich', 'bitte', 'sie', ',', 'sich', 'zu', 'einer', 'schweigeminute', 'zu', 'erheben', '.']
Source tokens: [0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 266, 22, 5, 29, 14, 78, 3931, 14, 2414, 4]
Ideal tokens out: [1, 651, 932, 6, 159, 6, 19, 11, 1440, 35, 51, 2639, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0]
Ideal sentence out: ['please', 'rise', ',', 'then', ',', 'for', 'this', 'minute', "'", 's', 'silence', '.'] 
Model output: i ask you to move , on an approach an approach .
-------------------
Source sentence: ['(', 'das', 'parlament', 'erhebt', 'sich', 'zu',  einer', 'schweigeminute', '.', ')']
Source tokens: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 52, 11, 58, 3267, 29, 14, 78, 3931, 4, 51]
Ideal tokens out: [1, 54, 5, 267, 3541, 14, 2095, 12, 1440, 35, 51, 2639, 53, 2, 0, 0, 0, 0, 0, 0, 0, 0]
Ideal sentence out: ['(', 'the', 'house', 'rose', 'and', 'observed', 'a', 'minute', "'", 's', 'silence', ')']
Model output: ( the house ( observed and observed a speaker )
-------------------
Source sentence: ['frau', 'präsidentin', ',', 'zur', 'geschäftsordnung', '.']
Source tokens: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 151, 5, 49, 488, 4]
Ideal tokens out: [1, 212, 44, 6, 22, 12, 91, 8, 218, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Ideal sentence out: ['madam', 'president', ',', 'on', 'a', 'point', 'of', 'order', '.'] 
Model output: madam president , on a point of order .
-------------------
Source sentence: ['wenn', 'das', 'haus', 'damit', 'einverstanden', 'ist', ',', 'werde', 'ich', 'dem', 'vorschlag', 'von', 'herrn', 'evans', 'folgen', '.']
Source tokens: [0, 0, 0, 0, 85, 11, 603, 113, 831, 9, 5, 243, 13, 39, 141, 18, 116, 1939, 417, 4]
Ideal tokens out: [1, 87, 5, 267, 2096, 6, 16, 213, 47, 29, 27, 1941, 25, 1441, 4, 2, 0, 0, 0, 0, 0, 0]
Ideal sentence out: ['if', 'the', 'house', 'agrees', ',', 'i', 'shall', 'do', 'as', 'mr', 'evans', 'has', 'suggested', '.']
Model output: if the house gave this proposal , i would like to hear mr byrne .
```
上述结果可显示出，输出中虽然有些许有问题的标记，但结果大部分是正确的。为了缓解这个问题，则需要更复杂的RNN，更长或更多样化的语料库。

###课后作业
本章模型是在同一个数据集上进行的测试和训练，这在数据科学中不是理想做法，然而在一个工作项目中必须要进行训练和测试。读者可以尝试寻找更长的语料库，并把它分成两部分：一份用于训练，一份用于测试：
* 尝试改变模型的设置：这些改变会如何影响模型性能和训练时间？
* 分析`seq2seq_model.py`的代码，请读者思考，如何把损失情况画到TensorBoard中？
* NLTK同样包含法语语料库，请读者思考，如何将德语同时翻译成英语和法语？

###小结
本章为读者介绍了如何利用RNN构建一个机器翻译系统。具体地，本章讲解了如何组织语料库，如何训练模型以及如何进行测试。下一章会为读者讲解RNN的其他应用：聊天机器人。
