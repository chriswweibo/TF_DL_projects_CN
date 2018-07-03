# 第3章 为图像生成标注

标注生成是深度学习领域中最重要的应用之一，近年来得到了广泛的关注。图像标注模型涉及视觉信息和自然语言处理的结合。  

在本章中，我们将了解：  

* 标注生成领域的最新进展
* 标注生成是如何工作的
* 标注生成模型的实现

## 什么是标注生成？  

标注生成是用自然语言来描述图像。在以前的研究中，标注生成模型使用物体检测模型与生成文本的模板。随着深度学习的发展，这些模型已经被卷积神经网络和递归神经网络的结合所取代。  

一个例子如下：  

<这里应该有一张图>

有几个数据集可以帮助我们训练图像标注模型。  

## 探索图像标注数据集  

有许多个数据集可用于标注图像任务。数据集通常是由向几个人显示一幅图像并要求他们每个人写一个关于该图像的句子得到的。通过该方法，同一图像可以得到多个标注。多个标注选项有助于更好地泛化。这个问题的难点在于模型性能的排序，对于每一代模型，最好由人类评估标注质量。对于这项任务来说，自动评估是较为困难的。后面让我们研究一下Flickr 8数据集。  

## 下载数据集  

Flickr 8是从Flickr收集的，不允许用于商业用途。读者可以从https://forms.illinois.edu/sec/1713398 下载Flickr 8数据集。标注可以从以下网址下载 http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html 。我们需要分别下载文本和图像。可通过填写页面上显示的表格获得访问权限：
<这里应该有一张图>
电子邮件将与下载链接一起发送。下载并解压缩后，文件应该如下所示：
```
Flickr8k_text 
CrowdFlowerAnnotations.txt 
Flickr_8k.devImages.txt 
ExpertAnnotations.txt 
Flickr_8k.testImages.txt 
Flickr8k.lemma.token.txt 
Flickr_8k.trainImages.txt 
Flickr8k.token.txt readme.txt 
```
下面是数据集中的一些示例：  

<这里应该有一张图>

上图对应的标注是：  

```
A man in street racer armor is examining the tire of another racer's motor bike 
The two racers drove the white bike down the road 
Two motorists are riding along on their vehicle that is oddly designed and colored 
Two people are in a small race car driving by a green hill 
Two people in racing uniforms in a street car
```

下面是示例2：  

<这里应该有一张图>

上图对应的标注是：  

```
A man in a black hoodie and jeans skateboards down a railing 
A man skateboards down a steep railing next to some steps 
A person is sliding down a brick rail on a snowboard 
A person walks down the brick railing near a set of steps 
A snowboarder rides down a handrail without snow
```

如上所示，一个图像对应多个标注。这也表示了图像标注任务的难度。  

## 将单词转换为词嵌入  

为了生成标注，英语单词必须转换为词嵌入。嵌入是文字或图像的矢量或数字表示。如果将单词转换为向量形式，就可以使用这些向量执行算术运算，这是很有用的。  

这种嵌入可以通过两种方法学习，如下图所示：  

<这里应该有一张图>

CBOW方法通过给定目标周围单词来预测目标单词以学习词嵌入。Skip-gram方法使用目标单词对目标单词的周围词进行预测，这与CBOW方法相反。根据上下文，可以对目标词进行训练，如下图所示：

<这里应该有一张图>

 一旦训练结束，词嵌入可以可视化：

<这里应该有一张图>  

这种类型的嵌入可以用来执行词的向量运算。本章中词嵌入的概念将非常关键。  

## 图像标注方法

标注图像有几种方法。以前的方法是根据图像中存在的对象和属性构造句子。之后利用递归神经网络(RNN)生成句子。目前最精确的方法是基于注意力机制的标注方法。我们在本节中将详细探讨这些技术和结果。 

### 条件随机场

研究者们首先尝试了一种利用条件随机场(conditional random field, CRF)构造句子的方法，该方法利用图像中检测到的对象和属性来构造句子。这一过程所涉及的步骤如下：

<这里应该有一张图>

工作流示例(来源：http://www.tamaraberg.com/papers/generation_cvpr11.pdf)。

CRF造出流畅句子的能力有限，生成的句子质量不高，如以下截图所示：

<这里应该有一张图>

尽管对象和属性正确，但是这里显示的句子太结构化了。

Kulkarni等人在论文http://www.tamaraberg.com/Papers/Generationcvpr11.pdf中提出了一种从图像中找出对象和属性并利用其生成具有条件随机场(CRF)的文本的方法。

### 基于卷积神经网络的递归神经网络

将递归神经网络与卷积神经网络特征相结合来生成句子，这使模型的端到端训练成为可能。以下是该模型的体系结构：

<这里应该有一张图>

LSTM模型(来源：https://arxiv.org/pdf/1411.4555.pdf)。

使用了多层LSTM来产生所需的结果。下图显示了该模型的一部分结果：

<这里应该有一张图>

这些结果优于CRF。这说明了LSTM在生成句子方面的强大能力。

参考文献：Vinyals等人，在论文https://arxiv.org/pdf/1411.4555.pdf中提议通过将CNN和RNN堆叠起来对图像标注进行端到端的学习。

### 标注排名

标注排名是一种有趣的从一组标注中选择标注的方法。首先，根据图像的特征对图像进行排序，并选择相应的标注，如下图所示：

<这里应该有一张图>

资料来源：http://papers.nips.cc/paper/4470-im2text-describing-images-using-1-million-captioned-photographs.pdf

可以使用一组不同的属性对以上图像重新进行排序。通过获得更多的图像可以大幅度提高质量，如下图所示：

<这里应该有一张图>

资料来源：http://papers.nips.cc/paper/4470-im2text-describing-images-using-1-million-captioned-photographs.pdf

随着数据集中图像数量的增加，结果会变好。
欲了解有关标注排名的更多信息，请参阅 http://papers.nips.cc/paper/4470-im2text-describing-images-using-1-million-captionedphotographs.pdf

### 密集标注

密集标注是一个图像上的多个标注的问题。以下是该问题的架构：

<这里应该有一张图>

资料来源：https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Johnson_DenseCap_Fully_Convolutional_CVPR_2016_paper.pdf
这种架构得到了较优的效果。
欲了解更多信息，请参阅：johnson等人的文章https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Johnson_DenseCap_Fully_Convolutional_CVPR_2016_paper.pdf ，此文中提出了一种密集标注的方法。

### RNN标注

视觉特征可以与序列学习一起使用来形成输出。

<这里应该有一张图>

这是一种用于标注生成的体系结构。
详情请参阅：Donahue等人的文章https://arxiv.org/pdf/1411.4389.pdf ，提出了用于图像标注的长期循环卷积结构(LRCN)。

### 多模态标注

图像和文本都可以映射到相同的嵌入空间以生成标注。

<这里应该有一张图>

需要解码器来生成标注。

### 基于注意力机制的标注
详情请参阅：Xu等人在论文https://arxiv.org/pdf/1502.03044 中提出了一种基于注意力机制的图像标注方法。
基于注意力的标注方法最近比较流行，得益于更好的准确性：
<这里应该有一张图>
这种方法按照标注的顺序训练注意力模型，以产生更好的结果：
<这里应该有一张图>
下面是一个使用了注意力标注机制LSTM的图表：
<这里应该有一张图>
这里展示了几个示例，其中以时间序列的方式出色的展现了对象的可视化：
<这里应该有一张图>
结果真的很好！

## 实现标注生成模型
首先，让我们读取数据集并按照我们需要的方式进行转换。导入`os`库并声明数据集所在的目录，如下代码所示：   
```python
import os 
annotation_dir = 'Flickr8k_text'
```
接下来，定义一个函数来打开文件并将文件中的行作为列表返回： 

```python
def read_file(file_name):
    with open(os.path.join(annotation_dir, file_name), 'rb') as file_handle:
        file_lines = file_handle.read().splitlines()
    return file_lines
```

读取标注以及训练和测试数据集的图像路径：  

```python
train_image_paths = read_file('Flickr_8k.trainImages.txt') 
test_image_paths = read_file('Flickr_8k.testImages.txt') 
captions = read_file('Flickr8k.token.txt')

print(len(train_image_paths)) 
print(len(test_image_paths)) 
print(len(captions))
```

输出结果应该如下：

```
6000
1000
40460  
```

接下来需要生成图像到标注的映射。这将有助于训练时方便的查找标注。此外，标注数据集中的单词将有助于创建词汇表：   

```python
image_caption_map = {} 
unique_words = set() 
max_words = 0 
for caption in captions:    
    image_name = caption.split('#')[0]    
    image_caption = caption.split('#')[1].split('\t')[1]    
    if image_name not in image_caption_map.keys():        
        image_caption_map[image_name] = [image_caption]    
    else:        
        image_caption_map[image_name].append(image_caption)    
    caption_words = image_caption.split()
    max_words = max(max_words, len(caption_words))    
    [unique_words.add(caption_word) for caption_word in caption_words]
```

现在需要建立两个映射，一个是从词到索引，另一个是从索引到词：  

```python
unique_words = list(unique_words) 
word_to_index_map = {} 
index_to_word_map = {} 
for index, unique_word in enumerate(unique_words):    
    word_to_index_map[unique_word] = index    
    index_to_word_map[index] = unique_word 
print(max_words)
```

标注中出现的最大单词数为38个，这将有助于定义结构。接下来，导入库：

```python
from data_preparation import train_image_paths, test_image_paths 
from keras.applications.vgg16 import VGG16 
from keras.preprocessing import image 
from keras.applications.vgg16 import preprocess_input 
import numpy as np 
from keras.models import Model 
import pickle 
import os
```

现在开始创建`ImageModel`类，以加载VGG模型及其权重：  

```python
class ImageModel:
    def __init__(self):
        vgg_model = VGG16(weights='imagenet', include_top=True)
        self.model = Model(input=vgg_model.input,
                           output=vgg_model.get_layer('fc2').output)
```

权重将被下载并存储。第一次使用此代码可能需要一些时间（用于下载权重）。接下来再创建一个模型，以便使用第二个全连接层的输出。以下是从路径读取图像并进行预处理的方法：   

```python
@staticmethod
def load_preprocess_image(image_path):
    image_array = image.load_img(image_path, target_size=(224, 224))
    image_array = image.img_to_array(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_array
```

接下来定义一个加载图像并进行预测的方法。预测的第二个全连接层的尺寸可以被重塑为4096：  

```python
def extract_feature_from_imagfe_path(self, image_path):
    image_array = self.load_preprocess_image(image_path)
    eatures = self.model.predict(image_array)
    return features.reshape((4096, 1))
```

浏览一个图像路径列表并创建一个特征列表：  

```python
def extract_feature_from_image_paths(self, work_dir, image_names):
    features = []
    for image_name in image_names:
        image_path = os.path.join(work_dir, image_name)
        feature = self.extract_feature_from_image_path(image_path)
        features.append(feature)
    return features
```

接下来，将提取的特征存储为一个`pickle`文件：  

```python
def extract_features_and_save(self, work_dir, image_names, file_name):
    features = self.extract_feature_from_image_paths(work_dir, image_names)
    with open(file_name, 'wb') as p:
        pickle.dump(features, p)
```

接下来，初始化类并提取训练集和测试集的图像特征：

```python
I = ImageModel() 
I.extract_features_and_save(b'Flicker8k_Dataset',train_image_paths, 'train_image_features.p') 
I.extract_features_and_save(b'Flicker8k_Dataset',test_image_paths, 'test_image_features.p')
```

导入构建模型所需的层：  

```python
from data_preparation import get_vocab
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten
from keras.preprocessing import image, sequence
```

获得所需的词汇表：  

```python
image_caption_map, max_words, unique_words, \
word_to_index_map, index_to_word_map = get_vocab() 
vocabulary_size = len(unique_words)
```

对于最终生成标注的模型：

```python
image_model = Sequential() 
image_model.add(Dense(128, input_dim=4096, activation='relu')) 
image_model.add(RepeatVector(max_words))
```

对于语言创建一个模型：  

```python
lang_model = Sequential() 
lang_model.add(Embedding(vocabulary_size, 256, input_length=max_words)) 
lang_model.add(LSTM(256, return_sequences=True)) 
lang_model.add(TimeDistributed(Dense(128)))
```

将两个模型合并为最终模型：

```python
model = Sequential() 
model.add(Merge([image_model, lang_model], mode='concat')) 
model.add(LSTM(1000, return_sequences=False)) 
model.add(Dense(vocabulary_size)) 
model.add(Activation('softmax')) 
model.compile(loss='categorical_crossentropy', 
              optimizer='rmsprop', 
              metrics=['accuracy']) 
batch_size = 32 
epochs = 10 
total_samples = 9 
model.fit_generator(data_generator(batch_size=batch_size), 
                    steps_per_epoch=total_samples / batch_size,
                    epochs=epochs, verbose=2)
```

这个模型可以被训练用于产生图像标注。 

## 总结  

在本章中，我们学习了图像标注技术。首先，我们了解了词向量的嵌入空间。然后，对几种图像标注处理方法进行了研究。接着实现了图像标注模型。  

在下一章中，我们将研究生成对抗网络(GAN)的概念。GAN有趣且有用，可以产生各种用途的图像。 