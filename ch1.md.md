

## 第一章  用卷积网络识别交通标志



作为本书的第一个项目，我们首先尝试用一个简单的模型解决交通标志识别问题，在这个问题上深度学习的表现非常好。简而言之，给定一副交通标志的彩色图像，我们的模型可以识别出它是什么信号。下面，我们将研究以下几个方面：

* 数据集是如何构成的
* 应该使用哪种深度网络
* 如何对数据集中的图像进行预处理
* 如何训练和预测并关注性能

##数据集

由于我们将要尝试用图像预测交通标志，因此我们需要使用为此而建立的数据集。幸运的是，德国 Neuroinformatik研究所的研究者建立了一个包含约40,000张不同图像的数据集，指向43种交通标志。我们将要用到的数据集是德国交通标志识别基准（German Traffic Sign Recognition Benchmark， GTSRB）竞赛的一个子集，竞赛试图对为实现同一目标而建立的多个模型的性能打分。尽管数据集比较古老——来自2011年，但是对于我们的项目来说，它是一个看起来很好的数据集。
数据集下载地址：t http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip.

在运行代码之前，请先下载文件并解压到代码所在的目录下。解压后，你将得到一个名为GTSRB的文件夹，其中包含数据集。本书的作者感谢为这一开源数据集做出贡献的人。

另外，可以参考 http://cs231n.github.io/convolutional-networks，以便了解更多关于CNN的知识。

限速20千米/小时:

<img src="figures\7_1.jpg" />

直行或右转：

<img src="figures\7_2.jpg" />

弯道：

<img src="figures\8_1.jpg" />

正如你所见，标志的亮度并不统一（有些很暗而有些很亮），尺寸不同，视角不同，背景不同，并且可能包含其他交通标志。

数据集的组织方式如下：所有标签相同的图像在同一文件夹中。例如，在 GTSRB/Final_Training/Images/00040/这一文件夹下，所有的图像的标签都是40，而 GTSRB/Final_Training/Images/00005/中的图像的标签为5。注意，所有图像都是PPM格式，这是一种无损压缩格式，拥有很多开源的编码/解码器。

The CNN network

##CNN网络

在我们的项目中，我们使用一个具有如下结构的非常简单的网络：
<img src="figures\8_2.jpg" />
在这一结构中，我们仍然有以下选择：

* 二维卷积层中滤波器的个数和核大小

* 池化层中的核的大小

* 全连接层中的单元数

* 批大小，优化算法，学习步骤（目标，衰减率），每个层的激活函数，和迭代的数量




##图像预处理

模型的第一步操作是读入图像并进行标准化。事实上，我们不能在图像尺寸不统一的情况下进行后续工作。因此，第一步，我们将加载图像并且将其变形为指定的尺寸（32x32)。除此之外，我们需要对标签进行热编码，得到一个43维的矩阵，矩阵中的每一维只有一个元素有效；与此同时，我们把图像的颜色空间从RGB转为灰度图。观察图像可以发现，我们所需要的信息不是在标志的颜色中，而是在形状和设计中。

下面，让我们打开一个jupyter notebook，并写入一些代码。首先，我们设置一些全局变量，包括类的数量（43）和变性后的图像的尺寸：


```pyrthon
N_CLASSES = 43
RESIZED_IMAGE = (32, 32)
```

下一步，我们写一个函数，用来将读取给定目录下的所有图像，把它们转化为给定形状，转为灰度图，对标签做one-hot encoder。为了完成这些，我们需要使用一个名为数据集的元组：


```python
import matplotlib.pyplot as plt
import glob
from skimage.color 
import rgb2lab from skimage.transform 
import resize from collections 
import namedtuple
import numpy as np np.random.seed(101) %matplotlib inline
Dataset = namedtuple('Dataset', ['X', 'y']) 
def to_tf_format(imgs):
    return np.stack([img[:, :, np.newaxis] for img in imgs],
axis=0).astype(np.float32) 
def read_dataset_ppm(rootpath, n_labels, resize_to): 
    images = [] labels = [] for c in range(n_labels):    
        full_path = rootpath + '/' + format(c, '05d') + '/'    
        for img_name in glob.glob(full_path + "*.ppm"):      
            img = plt.imread(img_name).astype(np.float32)
            img = rgb2lab(img / 255.0)[:,:,0]      
            if resize_to:        
                img = resize(img, resize_to, mode='reflect')             
                label = np.zeros((n_labels, ), dtype=np.float32)
                label[c] = 1.0                
             images.append(img.astype(np.float32))
            labels.append(label)
return Dataset(X = to_tf_format(images).astype(np.float32),
               y = np.matrix(labels).astype(np.float32))
dataset = read_dataset_ppm('GTSRB/Final_Training/Images', N_CLASSES,RESIZED_IMAGE) 
print(dataset.X.shape) print(dataset.y.shape)
```

skimage模块使得图像的读取、转化、变形操作非常容易。在我们的计划中，我们决定对原始的颜色空间（RGB）进行转化，只保留亮度分量。这里另一个好的变换是YUV，只有Y通道会作为灰度图保存。

运行代码的结果如下：

```python
(39209, 32, 32, 1)
(39209, 43)
```

关于输出格式：待观测的矩阵X维度为4。第一维代表索引位置（近40000），其他三维表示图像信息（32*32*1的灰度图）.这是用tensorflow处理图像的默认形状（详见代码中`_tf_format`函数）。

对于标签矩阵，行是待观测目标的索引，列是标签的独热编码。

为了更好地理解物体矩阵，我们打印第一个样本的特征向量和标签：

```
plt.imshow(dataset.X[0, :, :, :].reshape(RESIZED_IMAGE)) #sample
print(dataset.y[0, :]) #label
```
<img src="figures\11_1.jpg" />
```
 [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.

0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
```


下面我们打印最后一个样本：

```python
plt.imshow(dataset.X[-1, :, :, :].reshape(RESIZED_IMAGE)) #sample
print(dataset.y[-1, :]) #label
```
<img src="F:\TF\figures\12_1.jpg" />
```
[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.

0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
```

可以发现，图像的特征向量维度为32*32.标签只在第一个位置包含1维

这是我们建立模型需要的两部分信息。请格外注意形状，因为用深度学习处理图像的过程中它们很关键。与经典机器学习矩阵相比，这里X的维度为4。

预处理的最后一步是训练集/测试集的分割。我们希望在数据集的一个子集上训练模型，并在其补集，即测试集上测试性能。为实现这一目标，我们需要用`sklearn`提供的功能。

```
from sklearn.model_selection import train_test_split 
idx_train, idx_test = train_test_split(range(dataset.X.shape[0]),  
                                       test_size=0.25, random_state=101)
X_train = dataset.X[idx_train, :, :, :]
X_test = dataset.X[idx_test, :, :, :]
y_train = dataset.y[idx_train, :] y_test = dataset.y[idx_test, :]
print(X_train.shape) print(y_train.shape) 
print(X_test.shape) print(y_test.shape)
```

在这个例子中，我们用数据集中75%的样本训练，用余下25%的样本测试。事实上，这是当前代码的输出：

```python
(29406, 32, 32, 1)
(29406, 43)
(9803, 32, 32, 1)
(9803, 43)
```

##训练模型并进行预测

首先要具备的是一个队训练数据生成分批处理的函数，事实上，对于每个训练的迭代，我们需要插入训练集中样本的一个分批处理。这里，我们需要建立一个函数，它可以获取样本，标签，分批处理并返回分批处理生成器。

进一步地，为了引入训练数据的变化性，我们在函数中加入新的选项，对于不同的生成器是否混合数据的可能性。每个分批处理有不同的数据会使模型学习输入-输出的连接并且不存储序列。

```python
def minibatcher(X, y, batch_size, shuffle): 
assert X.shape[0] == y.shape[0] n_samples = X.shape[0] 
if shuffle:  
    idx = np.random.permutation(n_samples) 
else:   
    idx = list(range(n_samples)) 
for k in range(int(np.ceil(n_samples/batch_size))):    
    from_idx = k*batch_size    
    to_idx = (k+1)*batch_size    
    yield X[idx[from_idx:to_idx], :, :, :], y[idx[from_idx:to_idx], :]
```

为了测试这一函数，下面我们打印分批处理为10000时分批处理的形状

```python
for mb in minibatcher(X_train, y_train, 10000, True): 
    print(mb[0].shape, mb[1].shape)
```
打印结果如下：

```python
(10000, 32, 32, 1) (10000, 43)
(10000, 32, 32, 1) (10000, 43)
(9406, 32, 32, 1) (9406, 43)
```
不出所料，训练集中的29406个样本被分成了两个10000，最后一个9406.当然，标签矩阵中元素的数量也是这些。

下面终于到了建立模型的时候。首先，我们需要确定网络的模块。我们可以从建立全连接层开始，加入可变数量的单元，不加入激活层。我们采用Xavier对参数进行初始化，偏移量设为0.输出是输入经过权重，相加，偏移后的组合。需要注意的是权重的维度是动态定义的，可以在网络的任何地方使用。

```python                                        
import tensorflow as tf def fc_no_activation_layer(in_tensors, n_units): 
    w = tf.get_variable('fc_W', 
                        [in_tensors.get_shape()[1], n_units],
                        tf.float32,
                        tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('fc_B',
                        [n_units, ], 
                        tf.float32,
                        tf.constant_initializer(0.0)) 
    return tf.matmul(in_tensors, w) + b
```

下面我们来创建带激活的全连接层，特别地，这里我们的激活函数采用relu。如你所见，我们可以用下面的方法来实现这一功能：

```python
def fc_layer(in_tensors, n_units): 
            return tf.nn.leaky_relu(fc_no_activation_layer(in_tensors, n_units))
```
最后，我们创建一个卷积层，参数包含输入数据，核尺寸，滤波器（或神经元）个数。我们将采用与全连接层相同的激活函数。在此，输出层需要通过leaky relu激活：

 ```python
def conv_layer(in_tensors, kernel_size, n_units): 
    w = tf.get_variable('conv_W',
					[kernel_size, kernel_size, in_tensors.get_shape()[3], n_units],           
                      tf.float32,
                      tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('conv_B',
                       [n_units, ],
                       tf.float32,
                       tf.constant_initializer(0.0))
    return tf.nn.leaky_relu(tf.nn.conv2d(in_tensors, w, [1, 1, 1, 1], 'SAME') + b)
 ```

现在需要建立池化层。这里，窗口的尺寸和步长都是平方级的。

```python
def maxpool_layer(in_tensors, sampling): 
    return tf.nn.max_pool(in_tensors, [1, sampling, sampling, 1], [1, sampling, sampling, 1], 'SAME')
```

最后要做的是定义dropout，用来标准化网络。dropout的创建相当简单，只需要记住，5它在训练时会用到，预测时不会使用。因此，我们需要一个额外的操作去定义是否进行dropout

```python
def dropout(in_tensors, keep_proba, is_training): 
    return tf.cond(is_training, lambda: tf.nn.dropout(in_tensors, keep_proba), lambda: in_tensors)
```
最后，需要按照前边的定义把各功能结合并创建模型。我们建立的模型包含以下层：

1. 二维卷积，5*5，32个滤波器
2. 二维卷积，5*5，64个滤波器
3. 展平
4. 全连接层，1024个单元
5. 40%的dropout
6. 全连接层（无激活函数）
7. 多元逻辑回归

下面是代码：

```python
def model(in_tensors, is_training): 
    # First layer: 5x5 2d-conv, 32 filters, 2x maxpool, 20% drouput with tf.variable_scope('l1'):   
    l1 = maxpool_layer(conv_layer(in_tensors, 5, 32), 2)
    l1_out = dropout(l1, 0.8, is_training) 
    # Second layer: 5x5 2d-conv, 64 filters, 2x maxpool, 20% drouput with tf.variable_scope('l2'):
    l2 = maxpool_layer(conv_layer(l1_out, 5, 64), 2)
    l2_out = dropout(l2, 0.8, is_training) with tf.variable_scope('flatten'):    
        l2_out_flat = tf.layers.flatten(l2_out) 
        # Fully collected layer, 1024 neurons, 40% dropout with tf.variable_scope('l3'):
        l3 = fc_layer(l2_out_flat, 1024)    
        l3_out = dropout(l3, 0.6, is_training)
# Output
with tf.variable_scope('out'):   
    out_tensors = fc_no_activation_layer(l3_out, N_CLASSES) return out_tensors
```

现在，我们需要写函数来训练模型并测试性能。请注意，下面所有的代码都属于训练模型的函数，为了便于解释被拆成小块。

函数的参数包括训练集、测试集及对应的标签，学习率，迭代次数，批大小。在这一切当中，首先，需要定义tensorflow定位符：每个分批处理，一个分批处理的标签，是否进行训练（主要用于dropout层）
```
**from sklearn.metrics import classification_report, confusion_matrix def train_model(X_train, y_train, X_test, y_test, learning_rate, max_epochs, batch_size): in_X_tensors_batch = tf.placeholder(tf.float32, shape = (None,**
**RESIZED_IMAGE[0], RESIZED_IMAGE[1], 1))**
**in_y_tensors_batch = tf.placeholder(tf.float32, shape = (None, N_CLASSES)) is_training = tf.placeholder(tf.bool)**
```
下面，我们定义输出，度量分数，优化器。这里，我们采用AdamOptimizer和交叉熵及多元逻辑回归作为损失函数：
```python
**logits = model(in_X_tensors_batch, is_training)**
**out_y_pred = tf.nn.softmax(logits)**
**loss_score = tf.nn.softmax_cross_entropy_with_logits(logits=logits,**
**labels=in_y_tensors_batch) loss = tf.reduce_mean(loss_score) optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)**
**And finally, here's the code for training the model with 分批处理es:**
**with tf.Session() as session:    session.run(tf.global_variables_initializer())    for epoch in range(max_epochs):     print("Epoch=", epoch)      tf_score = []      for mb in 分批处理er(X_train, y_train, batch_size, shuffle = True):        tf_output = session.run([optimizer, loss],**
   **feed_dict = {in_X_tensors_batch : mb[0],                                             in_y_tensors_batch : b[1],**
                **is_training : True})**
       **tf_score.append(tf_output[1])**
     **print(" train_loss_score=", np.mean(tf_score))**
```
在训练之后，需要在测试集上测试模型。这里，我们用整个测试集测试，而不是分批处理。由于我们不想使用dropout，对应的选项需要设为false
```python
   **print("TEST SET PERFORMANCE")**
   y_test_pred, test_loss = session.run([out_y_pred, loss],                                          feed_dict = {in_X_tensors_batch :
   **X_test,**
   **in_y_tensors_batch : y_test, is_training : False})**```
```
作为最终操作，我们打印分类结果并画出混淆矩阵来观察误分类的情况：
```python
   print(" test_loss_score=", test_loss)
   y_test_pred_classified = np.argmax(y_test_pred, axis=1).astype(np.int32)    y_test_true_classified = np.argmax(y_test, axis=1).astype(np.int32)
   print(classification_report(y_test_true_classified,
   **y_test_pred_classified))**
   cm = confusion_matrix(y_test_true_classified, y_test_pred_classified)
   plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
   **plt.colorbar()    plt.tight_layout()    plt.show()**
```

`log2`的版本，强调错误的分类`
```
    plt.imshow(np.log2(cm + 1), interpolation='nearest',
    cmap=plt.get_cmap("tab20"))
   plt.colorbar()    plt.tight_layout()    plt.show() tf.reset_default_graph()
```

最后，我们运行带有一些参数的函数。这里，我们运行模型，学习率是0.001，批大小 256， 迭代10次：

```train_model(X_train, y_train, X_test, y_test, 0.001, 10, 256)```

下面是输出：


```
Epoch= 0
train_loss_score= 3.4909246
Epoch= 1
train_loss_score= 0.5096467
Epoch= 2
train_loss_score= 0.26641673
Epoch= 3
train_loss_score= 0.1706828
Epoch= 4
train_loss_score= 0.12737551
Epoch= 5
train_loss_score= 0.09745725
Epoch= 6
train_loss_score= 0.07730477
Epoch= 7
train_loss_score= 0.06734192
Epoch= 8
train_loss_score= 0.06815668
Epoch= 9
train_loss_score= 0.060291935
TEST SET PERFORMANCE test_loss_score= 0.04581982
```

下面是每个类结果：

```
    精度  召回率 f1-score   样本数          
0   1.00     0.96     0.98       67
1	0.99     0.99      0.99      539
2	0.99     1.00     0.99       558
3	0.99     0.98     0.98       364
4	0.99     0.99     0.99       487
5	0.98     0.98     0.98       479
6	1.00    0.99     1.00       105
7	1.00     0.98     0.99       364
8	0.99     0.99     0.99       340
9	0.99     0.99     0.99       384
10	0.99     1.00     1.00       513
11	0.99     0.98     0.99       334
12	0.99     1.00     1.00       545
13	1.00     1.00     1.00       537
14	1.00     1.00     1.00       213
15	0.98     0.99     0.98       164
16	1.00     0.99     0.99       98
17	0.99     0.99     0.99       281
18	1.00     0.98     0.99       286
19	1.00     1.00     1.00       56
20	0.99     0.97     0.98       78
21	0.97     1.00     0.98       95
22	1.00     1.00     1.00       97
23	1.00     0.97     0.98       123
24	1.00     0.96     0.98       77
25	0.99     1.00     0.99      401
26	0.98     0.96     0.97       135
27	0.94     0.98     0.96       60
28	1.00     0.97     0.98       123
29	1.00     0.97     0.99       69
30	0.88     0.99    0.93       115
31	1.00     1.00     1.00       178
32	0.98     0.96     0.97       55
33	0.99     1.00     1.00       177
34	0.99     0.99     0.99       103
35	1.00      1.00     1.00       277
36	0.99     1.00     0.99       78
37	0.98     1.00     0.99       63
38	1.00     1.00     1.00       540
39	1.00     1.00     1.00       60
40	1.00     0.98     0.99       85
41	1.00     1.00     1.00       47         
42  0.98     1.00     0.99       53 
avg/total 0.99 0.99   0.99     9803
```
如你所见，我们在测试集上达到了99%的准确率，此外，召回率和f1值也为99%。由于测试集与最后一个迭代相似，模型稳定，没有过拟合或欠拟合。

下图是混淆矩阵：

<img src="figures\19_1.jpg" />

下面是log2版本进度的截屏：

<img src="figures\20_1.jpg" />

##后续问题

> 尝试添加或去掉卷积层或全连接层。这些改变会导致性能怎样变化？

> 这个简单的项目表明了dropout的必要性。改变dropout的比例，观察输出的过拟合和欠拟合情况。

> 现在，拍一张你所在城市的交通标志图，在现实生活中测试一下训练好的模型！



##总结

在这一章，我们看到了怎样用卷积神经网络识别交通标志。下一章，我们将用CNN完成更加复杂的任务。


