#第4章 为条件图像生成建立GAN

Yann LeCun，Facebook人工智能的领导者，最近提出“生成对抗网络是机器学习中十年来最有趣的想法“， 这一观点也由于引起了学界对这一深度学习方法的浓厚兴趣而得以证实。如果读者阅读了最近有关深度学习的论文（          同时也要注意LinkedIn或中级职位的主要趋势），就会发现GANs已经产生了非常多的变体。

读者可以通过Hindu Puravinash,创建的、支持更新的参考文献列表来了解GANs的世界的发展，详见https://github.com/hindupuravinash/the-gan-zoo/blob/master/gans.tsv；也可以通过学习Zheng Liu准备的GANs的时间线https://github.com/dongb5/GAN-Timeline来达到这一目的，它可以帮助读者把所有相关内容放在时间的角度。

GANs有能力去激发想象力，因为它们能展示人工智能的创造力，而不仅仅是它的计算能力。在本章中，我们将： 

* 揭开GANs的神秘面纱，为读者提供所有必要的概念来理解GANs是什么，GANs现在能做什么，以及GANs期望做什么。 

* 演示如何基于初始的示例图像来生成图像（即所谓的非监督的GANs）

* 解释如何使利用GANs为读者生成读者所需要的类型的结果图。

* 将GaNs与所期望的图像生成的条件相结合。 

* 建立一个基本的、完整的项目，可以处理不同的手写字符和图标数据集。 

* 为读者提供如何在云（特别是亚马逊云）上训练读者的GANs的基本指令              

GANs的成功很大程度上取决于读者使用的特定神经网络结构，以及它们所面对的问题和读者提供给它们的数据。 本章中我们选择的数据集可以提供满意的结果。我们希望读者能享受并被GAN的创造力所激励！


## GANs简介

我们将从最近的一些历史开始，因为GANs是读者能发现的整个人工智能和深度学习领域最新的想法之一。

一切起源于2014年，Ian Goodfellow和他的同事（Yoshua Bengio也在贡献者名单中）在蒙特利尔大学的Departement d'informatique et de recherche发表了一篇关于生成对抗网络（GANs）的论文，提出了能够基于一系列初始样例生成新数据的框架：、

GOODFELLOW, Ian, et al.Generative Adversarial Nets.In: Advances in Neural Information Processing Systems.2014.p.2672-2680: https://arxiv.org/abs/1406.2661.

考虑到以前使用马尔可夫链的尝试远不能令人信服，这种网络所产生的初始图像是惊人的。 在下图中，读者剋看到

论文中提出的一些示例，分别来自于MNIST，多伦多人脸数据集（Toronto Face Dataset，TFD），一个非公开数据集和cifar10数据集。

<img src="PATH\figures\72_1.jpg" />

图1：GANs的第一篇论文中采用不同数据集生成新图像的样本 a) MNIST b) TFD c) and d) CIFAR-10

来源：GOODFELLOW, Ian, et al.Generative Adversarial Nets.In: Advances in Neural Information Processing Systems.2014.p.2672-2680

这篇论文被认为是颇具创新性的，因为它把一个非常深的神经网络和博弈理论结合在一个真正智能的体系结构中，它不需要比通常的反向传播需要更多的训练。 GANs是生成类的模型，模型可以生成数据，因为它们刻画了模型分布图（例如，它们学习它）。因此，当它们生成某种东西时，就好像它们是从那个分布中取样一样。 

## 关键在于对抗方式

理解GANs为何能成为如此成功的生成模型在于对抗。事实上，GAN是的结构由两个不同的网络组成，它们基于各误差的汇集进行优化，这个过程就是对抗过程。

读者可以从一个真实数据集着手，我们称之为R，它包含读者的不同种类的图像或数据（GANs不局限于图像，尽管这是它们主要的应用）。之后读者可以建立生成网络G，用来生成与真实数据尽可能相似的伪造数据；读者还需要建立一个判别器D，用来比较G生成的数据和真实数据，指出数据是否真实。
Goodfellow用艺术伪造这一比喻来描述生成器，而判别器是侦探（或是艺术评论家），必须揭露他们的罪行。伪造者和侦探之间存在一个挑战，因为伪造者需要更有技巧以便不被侦探发现，而侦探在寻找伪造者方面也需要提升。伪造者和侦探之间的一切无休无止，知道伪造的产品与原始的完全相似。事实上，当GAN是过拟合时，它们会重新输出原始数据。这似乎可以解释为竞争市场，而它也确实是，因为GANs的想法就起源于竞争博弈论。

在GANs中，生成器产生图像，直到判别器无法分辨它们的真假。生成器的一个明显的解决方案是简单地复制一些训练图像或采用判别器无法判断的看起来成功的生成图像。我们的项目中将应用的是单面标签平滑技术的解决方案。具体描述见SALIMANS, Tim, etal.Improved techniques for training gans.In: Advances in Neural Information Processing Systems.2016.p.2234-2242: https://arxiv.org/abs/1606.03498

下面我们讨论GANs如何运作。首先，生成器G没有线索，完全随机地生成数据（事实上它甚至不考虑原始数据），因此它会被判别器D惩罚——此时分辨真实数据和伪造数据是很容易的。from D. G承担全部责任，开始尝试不同的东西以获得更好的D反馈。这一过程也是随机完成的，因为G能看见的是随机的输入，Z，它无法接触真实的数据。在很多次尝试和失败后，在判别器的指导下，生成器最终会指出如何做并开始生成可靠的输出。最后，经过足够的时间，生成器会生成器将完全复制所有原始数据，即使它并没有见到过其中任何一个示例。

<img src="PATH\figures\74_1.jpg" />

​							图2：一个简单的GAN架构如何工作的例子

##寒武纪爆发

正如上文提到的，关于GANs的新论文每个月都在产生（读者可以在Hindu Puravinash上查找文献列表，在本章的开头我们已经提到过）。

不管怎样，除了 Goodfellow和他的同事们最初的描述实施方法的论文之外，值得注意的最著名的的实施方案是**深度卷积对抗网络**（deep convolutional generative adversarial networks，*DCGANs*）和**条件生成对抗网络**（conditional GANs， *CGANs*）。

* DCGANs是基于CNN结构的GANs(RADFORD, Alec; METZ, Luke;CHINTALA, Soumith.Unsupervised representation learning with deep convolutionalgenerative adversarial networks.arXiv preprint arXiv:1511.06434, 2015: https://arxiv.org/abs/1511.06434).


* CGANs是在DAGANs的基础上，在输入标签上增加了一些条件，从而使得读者得到的结果中包含想得到的特征(MIRZA, Mehdi;OSINDERO, Simon.Conditional generative adversarial nets.arXiv preprint arXiv:1411.1784, 2014: https://arxiv.org/abs/1411.1784)。我们的项目会编写一个CGANs的类并在不同的数据集上训练它，来证明其功能。

 此外也有一些有趣的例子（我们的项目中不包括这些），给出了创造图像或提升的问题的实际解决方案：

* **循环·对抗生成网络**（CycleGAN），将一幅图像转化为另一幅（经典的例子是将马变为斑马: ZHU, Jun-Yan, et al.Unpaired image-	to-image translation using cycle-consistent adversarial networks.arXiv preprint arXiv:1703.10593, 2017: https://arxiv.org/abs/1703.10593)
* **堆栈对抗生成网络**（StackGAN）可以根据描述图像的文本生成图像 (ZHANG, Han, et al.Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks.arXiv preprint arXiv:1612.03242, 2016: https://arxiv\.org/abs/1612.03242)
	 **发现对抗生成网络**（DiscoveryGAN ，*DiscoGAN*) 传递不同图像之间的风格元素，	从而可以将一个时尚品，比如背包的纹理和装饰传递到一双鞋上(KIM, Taeksoo, et al.Learning to discover cross-domain relations with generative adversarial networks.arXiv preprint arXiv:1703.05192, 2017: https://arxiv.org/abs/1703.05192)
* SRGAN把低质量的图像转化为高分辨率图像(LEDIG, Christian, et al.Photo-realistic single image super-resolution using a generative adversarial network.arXiv preprint arXiv:1609.04802, 2016: https://arxiv.org/abs/1609.04802)



##DCGANs

DCGANs是GANs结构的第一个提升。DCGANs成功完成训练阶段，给出足够的轮和栗子，它们倾向于得到质量令人满意的输出。这使得它们很快成为GANs的基准并且有助于产生令人惊喜的成果，例如根据已知神奇宝贝产生新的：https://www.youtube.com/watch?v=rs3aI7bACGc或创造一些名人的脸，它们事实上并不存在但看起来像是真的（并不神秘），正如NVIDIA所做的：https://youtu.be/XOxxPcy5Gr4，用一个名为progressing growing:的新的训练方法，详见http://research.nvidia.com/sites/default/files/publications/karras2017gan- paper.pdf。它们的根源在于使用与深度学习监督网络中的图像分类中使用的相同的卷积，并且采用了一些巧妙的技巧： 

* 在两个网络中都使用批归一化
	 没有隐藏的连接层	
* 没有池化，只在卷积层设置步长
* 采用ReLU作为激活函数



##条件对抗生成网络

在条件对抗生成网络中，增加一个特征向量可以控制输出并更好地引导生成器认识到应该做什么。这样一个特征向量可以编码为图像应该导出的类（图像是女人还是男人，如果我们想创建虚构的演员的面孔）或者是我们希望从图像中得到的一系列特定的特征（对于虚构的演员，可以是发型，眼睛或肤色）。这里的技巧是将信息合并到要学习的图像中并交给Z输入，这里的输入不再完全是随机的。判别器的评价不知需要从原始图像中判断出伪造图像，还需要找到伪造图像对应的标签（或特征）：

<img src="PATH\figures\77_1.jpg" />

​					图3：将Z输入与Y输入结合（有标签的特征向量）允许生成受约束的图像。



## 项目

在我们开始处导入正确的库。除了Tensoflow之外，我们还需要使用numpy和math进行计算，scipy和matplotlib用来处理图像和图表，warnings、random和distuils来支持特定操作：

```	python
import numpy as np
import tensorflow as tf
import math
import warnings
import matplotlib.pyplot as plt
from scipy.misc import imresize
from random import shuffle
from distutils.version import LooseVersion
```


## 数据集

我们的第一步是提供数据。我们依赖已经完成预处理的数据集，但读者可以在自己的GAN中使用不同种类的图像。我们的想法是保持一个数据集类，它的任务是为我们以后构建的GANS类提供标准化和重构图像的批次。

在初始化时，我们需要同时处理图像和标签（如果存在）。图像首先需要变形（如果它们的形状和示例的类中定义的不同），然后搅乱。比起有序，例如按数据集中初始的类的顺序，搅乱能帮助GANs更好地学习，对于任何基于随机梯度下降（ stochastic gradient descent，sgd）机器学习方法这一点都是成立的：BOTTOU, Léon.Stochastic gradient descent tricks.In: Neural networks: Tricks of the trade.Springer, Berlin, Heidelberg, 2012.p.421-436: https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf）。标签采用one-hot encoder编码，为每一个类创建一个二进制变量，将其设置为1（其他类设为0）以保证标签可以转化为向量。例如，如果我们的类是 `{dog:0, cat:1}`, 我们需要两个 one-hot encoded 向量来表示它们:` {dog:[1, 0], cat:[0, 1]}`.

用这种方法，我们可以轻松地把向量加入我们的图像，作为另一个通道，并在其中加入一些会被我们的GAN重现的视觉特征。另外，我们可以安排向量的顺序，组成更复杂的有特殊特征的类。比方说，我们可以为我们希望生成的类指定编码，也可以指定它的一些特征：

```python
class Dataset(object):     
	def __init__(self, data, labels=None, width=28, height=28,                            max_value=255, channels=3):
        # Record image specs         
        self.IMAGE_WIDTH = width         
        self.IMAGE_HEIGHT = height         
        self.IMAGE_MAX_VALUE = float(max_value)
        self.CHANNELS = channels
        self.shape = len(data), self.IMAGE_WIDTH, 
        					  self.IMAGE_HEIGHT,self.CHANNELS       
        if self.CHANNELS == 3:             
        	self.image_mode = 'RGB'             
        	self.cmap = None         
        elif self.CHANNELS == 1:             
        	self.image_mode = 'L'             
        	self.cmap = 'gray'
        # Resize if images are of different size         
        if data.shape[1] != self.IMAGE_HEIGHT or \                   
        					data.shape[2] != self.IMAGE_WIDTH:             
        	data = self.image_resize(data,                    
        							self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
        # Store away shuffled data         
        index = list(range(len(data)))
        shuffle(index)         
        self.data = data[index]
        if len(labels) > 0:
            # Store away shuffled labels             
            self.labels = labels[index]             
            # Enumerate unique classes             
            self.classes = np.unique(labels)
            # Create a one hot encoding for each class
            # based on position in self.classes
            one_hot = dict()             
            no_classes = len(self.classes)             
            for j, i in enumerate(self.classes):                 
            	one_hot[i] = np.zeros(no_classes)
                one_hot[i][j] = 1.0             
            self.one_hot = one_hot         
         else:
            # Just keep label variables as placeholders
            self.labels = None             
            self.classes = None             
            self.one_hot = None
            
    def image_resize(self, dataset, newHeight, newWidth):
        """Resizing an image if necessary"""         
        channels = dataset.shape[3]         
        images_resized = np.zeros([0, newHeight,                          
        						newWidth, channels], dtype=np.uint8)         
        for image in range(dataset.shape[0]):
            if channels == 1:
                temp = imresize(dataset[image][:, :, 0],[newHeight, newWidth], 'nearest')
                temp = np.expand_dims(temp, axis=2)             
            else:                 
            	temp = imresize(dataset[image],
                               [newHeight, newWidth], 'nearest')             
                images_resized = np.append(images_resized,                          												np.expand_dims(temp, axis=0), axis=0)
        return images_resized
```

`get_batches`方法释放了数据集的一个子集并进行标准化，每个像素值除以它们的最大值（256），再减去0.5。结果图中的浮点的值域为` [-0.5, +0.5]`：
```python
def get_batches(self, batch_size):
    """Pulling batches of images and their labels"""
    current_index = 0
    # Checking there are still batches to deliver
    while current_index < self.shape[0]:
        if current_index + batch_size > self.shape[0]:
            batch_size = self.shape[0] - current_index
            data_batch = self.data[current_index:current_index \
            + batch_size]
        if len(self.labels) > 0:
            y_batch = np.array([self.one_hot[k] for k in \
            self.labels[current_index:current_index +\
            batch_size]])
        else:
            y_batch = np.array([])
        current_index += batch_size
        yield (data_batch /self.IMAGE_MAX_VALUE) - 0.5, y_batch
```


##CGAN class

基于CGAN模型的类包含运行CGAN所需要的所有函数。DCGANs被证明可以生成类似于照片质量的输出的性能。我们已经介绍过CGAN，为了提醒读者，参考文献如下：

> RADFORD, Alec; METZ, Luke; CHINTALA, Soumith.Unsupervisedrepresentation learning with deep convolutional Generative AdversarialNetworks.arXiv preprint arXiv:1511.06434, 2015 at https://arxiv.org/abs/1511.06434.

使用标签并将它们与图像集成（这是诀窍）。将导致更好的图像和决定生成图像的特征的可能性。 
conditional GANs 的参考文献：

>  MIRZA, Mehdi; OSINDERO, Simon.Conditional Generative Adversarial Nets.arXiv preprint 	arXiv:1411.1784, 2014, https://arxiv.org/abs/411.1784.

我们的CGAN希望输入数据集类对象，轮数，图像批大小，用于生成的输入的图像维数（`z_dim`）和GAN的名字（便于保存）。它会采用不同的值初始化，达到`alpha`或平滑。后面我们会讨论这两种参数对GAN网络的影响。

下面的示例设置了所有内部参数并在系统上检查性能，如果没有检测到GPU则给出警告：

```python
class CGan(object):
	def __init__(self, dataset, epochs=1, batch_size=32,z_dim=96, generator_name='generator',
				alpha=0.2, smooth=0.1,learning_rate=0.001, beta1=0.35):
        # As a first step, checking if the
        # system is performing for GANs
        self.check_system()
        # Setting up key parameters
        self.generator_name = generator_name
        self.dataset = dataset
        self.cmap = self.dataset.cmap
        self.image_mode = self.dataset.image_mode
        self.epochs = epochs
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.alpha = alpha
        self.smooth = smooth
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.g_vars = list()
        self.trained = False
     def check_system(self):
        """
        Checking system suitability for the project
        """
        # Checking TensorFlow version >=1.2
        version = tf.__version__
        print('TensorFlow Version: %s' % version)
        
        assert LooseVersion(version) >= LooseVersion('1.2'),\
        ('You are using %s, please use TensorFlow version 1.2 \
                            or newer.' % version)
        # Checking for a GPU
        if not tf.test.gpu_device_name():
            warnings.warn('No GPU found installed on the system.\
                          It is advised to train your GAN using\
                          a GPU or on AWS')
        else:
            print('Default GPU Device: %s' % tf.test.gpu_device_name())
```

`instantiate_inputs`函数未输入创建Tensorflow占位符，是一个随机实数。它也提供标签（创建与原始图像形状相同但通道数量等于类数的图像），对于训练过程的学习率：

```python
def instantiate_inputs(self, image_width, image_height,image_channels, z_dim, classes):
    """
    Instantiating inputs and parameters placeholders:
    real input, z input for generation,
    real input labels, learning rate
    """
    inputs_real = tf.placeholder(tf.float32,
                                (None, image_width, image_height,
    image_channels), name='input_real')
    inputs_z = tf.placeholder(tf.float32,
                              (None, z_dim + classes), name='input_z')
    labels = tf.placeholder(tf.float32,
                            (None, image_width, image_height,
                            classes), name='labels')
    learning_rate = tf.placeholder(tf.float32, None)
    return inputs_real, inputs_z, labels, learning_rate
```



下面，我们的工作转向网络结构，定义一些基本的函数例如`leaky_ReLU_activation`函数（我们将会在判别器和生成器中使用它，与深度卷积GANs的原始论文中描述的相反）：

```
def leaky_ReLU_activation(self, x, alpha=0.2):
	return tf.maximum(alpha * x, x)
def dropout(self, x, keep_prob=0.9):
	return tf.nn.dropout(x, keep_prob)
```

我们的下一个函数展示了一个判别器的层。它用Xavier初始化创建一个新的层，对结果执行批归一化，设置` leaky_ReLU_activation`，最后应用dropout做正则化：

```python
def d_conv(self, x, filters, kernel_size, strides,
    padding='same', alpha=0.2, keep_prob=0.5,
    train=True):
    """
    Discriminant layer architecture
    Creating a convolution, applying 批归一化,
    leaky rely activation and dropout
    """
    x = tf.layers.conv2d(x, filters, kernel_size,
    strides, padding, kernel_initializer=\
    tf.contrib.layers.xavier_initializer())
    x = tf.layers.batch_normalization(x, training=train)
    x = self.leaky_ReLU_activation(x, alpha)
    x = self.dropout(x, keep_prob)
    return x
```

Xavier初始化保证卷积的初始权重不会太小也不会太大。以便从最初的轮开始，就允许信号通过网络更好地转化。

> Xavier初始化提供了一个高斯分布，均值为0，方差为1除以一层中的神经元数量。这是因为这种初始化脱离了深度学习的预训练技术，以前用于设置可以在即使存在多个层也能进行反向传播初始的权重。读者可以在这篇文章中了解更多关于Glorot和Bengio的初始化变量的内容：http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavierinitialization。 

批归一化在这篇论文中被描述：

> IOFFE, Sergey; SZEGEDY, Christian.批归一化: Accelerating deep network training by reducing internal covariate shift.In: International Conference on Machine Learning.2015.p.448-456.

正如作者所指出的，批归一化算法做标准化需要处理变量偏移问题（http://sifaka.cs.uiuc.edu/jiang4/domain_adaptation/survey/node8.html），换言之，改变输入的分布会引起之前学习到的权重不再使用。事实上，作为第一个输入层初始习到的分布，它们会被传输到下面所有层，并且由于输入突然改变而随之变化（例如，一开始读者已经输入了狗的照片和更多的猫的照片，现在反过来了），除非读者把学习率设定得很低，否则很可能会让人望而生畏。

批归一化解决了改变输入的分布引起的问题，因为它对于每个batch用均值和方差进标准化（用batch统计数据），正如论文IOFFE, Sergey; SZEGEDY,Christian.批归一化: Accelerating deep network training byreducing internal covariate shift.In: International Conference on Machine Learning.2015.p.448-456所阐述的（它可以在网上找到，网址是https://arxiv.org/abs/1502.03167）。

`g_reshaping `和 `g_conv_transpose`是`generato`r中的两个函数。它们对输入重新定形，无论是平铺层还是卷积层。实际上，它们与卷积所做的工作相反，可以将卷积得到的特征恢复为原始特征：

```python
def g_reshaping(self, x, shape, alpha=0.2,
                keep_prob=0.5, train=True):
    """
    Generator layer architecture
    Reshaping layer, applying 批归一化,
    leaky rely activation and dropout
    """
    x = tf.reshape(x, shape)
    x = tf.layers.batch_normalization(x, training=train)
    x = self.leaky_ReLU_activation(x, alpha)
    x = self.dropout(x, keep_prob)
    return x
def g_conv_transpose(self, x, filters, kernel_size,
                    strides, padding='same', alpha=0.2,
                    keep_prob=0.5, train=True):
    """
    Generator layer architecture
    Transposing convolution to a new size,
    applying 批归一化,
    leaky rely activation and dropout
    """
    x = tf.layers.conv2d_transpose(x, filters, kernel_size,strides, padding)
    x = tf.layers.batch_normalization(x, training=train)
    x = self.leaky_ReLU_activation(x, alpha)
    x = self.dropout(x, keep_prob)
    return x
```

判别器体系结构以图像为输入，通过各种卷积变换，直到结果展平，变成对数和概率（通过`sigmoid`函数）。实际上，一切都与有序卷积相同： 

```python
def discriminator(self, images, labels, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # Input layer is 28x28x3 --> concatenating input
        x = tf.concat([images, labels], 3)
        
        # d_conv --> expected size is 14x14x32
        x = self.d_conv(x, filters=32, kernel_size=5,
                strides=2, padding='same',
                alpha=0.2, keep_prob=0.5)
        
        # d_conv --> expected size is 7x7x64
        x = self.d_conv(x, filters=64, kernel_size=5,
                strides=2, padding='same',
                alpha=0.2, keep_prob=0.5)
        
        # d_conv --> expected size is 7x7x128
        x = self.d_conv(x, filters=128, kernel_size=5,
                strides=1, padding='same',
                alpha=0.2, keep_prob=0.5)
        
        # Flattening to a layer --> expected size is 4096
        x = tf.reshape(x, (-1, 7 * 7 * 128))
        
        # Calculating logits and sigmoids
        logits = tf.layers.dense(x, 1)
        sigmoids = tf.sigmoid(logits)
        
        return sigmoids, logits	
```

至于生成器，它的结构和判别器相反。从输入向量z开始，首先创建`dense`层，之后进行一系列变换以重现判别器中卷积的逆过程，直到生成与输入形状相同的张量为止，并通过`tanh`函数做进一步的变换：

```python
def generator(self, z, out_channel_dim, is_train=True):
    with tf.variable_scope('generator',
                            reuse=(not is_train)):
        
        # First fully connected layer
        x = tf.layers.dense(z, 7 * 7 * 512)
        
        # Reshape it to start the convolutional stack
        x = self.g_reshaping(x, shape=(-1, 7, 7, 512),
        		alpha=0.2, keep_prob=0.5,
        		train=is_train)
        
        # g_conv_transpose --> 7x7x128 now
        x = self.g_conv_transpose(x, filters=256,
        		kernel_size=5,
        		strides=2, padding='same',
        		alpha=0.2, keep_prob=0.5,
        		train=is_train)
        
        # g_conv_transpose --> 14x14x64 now
        x = self.g_conv_transpose(x, filters=128,
                kernel_size=5, strides=2,
                padding='same', alpha=0.2,
                keep_prob=0.5,
                train=is_train)
        
        # Calculating logits and Output layer --> 28x28x5 now
        logits = tf.layers.conv2d_transpose(x,
                filters=out_channel_dim,
                kernel_size=5,
                strides=1,
                padding='same')
        output = tf.tanh(logits)
        return output
```

这一结构与介绍CGANs的论文中画出的结构非常相似，论文中画出了如何通过大小为100的输入向量重构64*64**3的图像：

<img src="PATH\figures\87_1.jpg" />

​								图4：DVGANs生成器结构

​								来源: arXiv, 1511.06434,2015

在定义结构之后，损失函数是接下来需要定义的重要元素。它采用两个输出，来自生成器的输出，将要在管道中被输入到判别器输出对数中；以及来自真实图像的输出，在管道中将要被输入到判别器中。对这二者而言，下一步需要计算损失。这里，平滑的参数很有用，因为它可以使真实图像转化为某些东西的概率不为1，允许一个更好的、更有可能性的结果被GANs学习（在完全惩罚的情况下，对于伪造图像，得到可以对抗真实图像的机会可能会变得更难）。

最终的判别器损失，是根据伪造图像和真实图像计算的损失的和。在真实图像上的损失是通过比较估计的logit与平滑的概率（我们的案例中是0.9）来计算的，这是为了防止过拟合，以及判别器可以通过存储图像来简单地学习并判断真实图像。生成器损失是由对伪图像的判别器估计的对数来计算的，其概率为1。 用这种方法，生成器努力产生伪造图像，它们能够被判别器判定为真（拥有较高的概率）。因此，在一个循环中，损失简单地从判别器对伪造图像的估计向生成器转化：

```python
def loss(self, input_real, input_z, labels, out_channel_dim):
    # Generating output
    g_output = self.generator(input_z, out_channel_dim)
    # Classifying real input
        d_output_real, d_logits_real = self.discriminator(input_real,labels, reuse=False)
    # Classifying generated output
    d_output_fake, d_logits_fake = self.discriminator(g_output, labels,reuse=True)
    # Calculating loss of real input classification
    real_input_labels = tf.ones_like(d_output_real) * (1 - self.smooth)
    # smoothed ones
    d_loss_real = tf.reduce_mean(
    			tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
    			labels=real_input_labels))
    # Calculating loss of generated output classification
    fake_input_labels = tf.zeros_like(d_output_fake) 
    # just zeros
    d_loss_fake = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                        labels=fake_input_labels))
    # Summing the real input and generated output classification losses
    d_loss = d_loss_real + d_loss_fake # Total loss for discriminator
    # Calculating loss for generator: all generated images should have been
    # classified as true by the discriminator
    target_fake_input_labels = tf.ones_like(d_output_fake) 
    # all ones
    g_loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                        labels=target_fake_input_labels))
    return d_loss, g_loss
```

既然GANs上的工作是可视化的，这里有一些对当前生成器产生的样例和特定的图像集合可视化的函数：

```python
def rescale_images(self, image_array):
    """
    Scaling images in the range 0-255
    """
    new_array = image_array.copy().astype(float)
    min_value = new_array.min()
    range_value = new_array.max() - min_value
    new_array = ((new_array - min_value) /range_value) * 255
    return new_array.astype(np.uint8)
    
def images_grid(self, images, n_cols):
    """
    Arranging images in a grid suitable for plotting
    """
    # Getting sizes of images and defining the grid shape
    n_images, height, width, depth = images.shape
    n_rows = n_images //n_cols
    
    projected_images = n_rows * n_cols
    # Scaling images to range 0-255
    images = self.rescale_images(images)
    # Fixing if projected images are less
    if projected_images < n_images:
        images = images[:projected_images]
        # Placing images in a square arrangement
        square_grid = images.reshape(n_rows, n_cols,
        height, width, depth)
        square_grid = square_grid.swapaxes(1, 2)
        # Returning a image of the grid
    if depth >= 3:
        return square_grid.reshape(height * n_rows, width * n_cols, depth)
    else:
        return square_grid.reshape(height * n_rows, width * n_cols)
    
def plotting_images_grid(self, n_images, samples):
    """
    Representing the images in a grid
    """
    n_cols = math.floor(math.sqrt(n_images))
    images_grid = self.images_grid(samples, n_cols)
    plt.imshow(images_grid, cmap=self.cmap)
    plt.show()
    
def show_generator_output(self, sess, n_images, input_z,
                         labels, out_channel_dim,
                         image_mode):
    """
    Representing a sample of the
    actual generator capabilities
    """
    # Generating z input for examples
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, \
    z_dim - labels.shape[1]])
    example_z = np.concatenate((example_z, labels), axis=1)
    # Running the generator
    sample = sess.run(
                  self.generator(input_z, out_channel_dim, False),
                  feed_dict={input_z: example_z})
    # Plotting the sample
    self.plotting_images_grid(n_images, sample)
    
def show_original_images(self, n_images):
    """
    Representing a sample of original images
    """
    # Sampling from available images
    index = np.random.randint(self.dataset.shape[0],
                              size=(n_images))
    sample = self.dataset.data[index]
    # Plotting the sample
    self.plotting_images_grid(n_images, sample)
```

使用Adam优化器，从最初的判别器开始判别器和生成器的损失都被减小（确定生成器针对真实图像的生成有多好），然后基于生成器产生的伪造图像对判别器器的影响的评估，将反馈传播到生成器： 

```python
def optimization(self):
    """
    GAN optimization procedure
    """
    # Initialize the input and parameters placeholders
    cases, image_width, image_height,\
    out_channel_dim = self.dataset.shape
                        input_real, input_z, labels, learn_rate = \
                                self.instantiate_inputs(image_width,
                                            image_height,
                                            out_channel_dim,
                                            self.z_dim,
                                            len(self.dataset.classes))
    # Define the network and compute the loss
    d_loss, g_loss = self.loss(input_real, input_z,
                               labels, out_channel_dim)
    # Enumerate the trainable_variables, split into G and D parts
    d_vars = [v for v in tf.trainable_variables() \
        if v.name.startswith('discriminator')]
    g_vars = [v for v in tf.trainable_variables() \
        if v.name.startswith('generator')]
    self.g_vars = g_vars
    
    # Optimize firt the discriminator, then the generatvor
    with tf.control_dependencies(\
                    tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(
                self.learning_rate,
                self.beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(
                self.learning_rate,
                self.beta1).minimize(g_loss, var_list=g_vars)
    return input_real, input_z, labels, learn_rate,_loss, g_loss, d_train_opt, g_train_opt
```

最后，我们完成了训练阶段。在训练中，有两部分需要注意：

* 如何在两步中完成优化、
    1. 运行判别器优化
    2. 运行生成器优化
* 如何对随机输入和真实图像进行预处理，对它们和标签进行混合，创建包含与标签相关的独热编码的类信息更多图像层

用这种方法，可以使输入和输出数据中，类被包含在图像里，调节生成器以便考虑到这些信息，同时，如果无法生成与正确标签相对应的真实图像，对生成器进行惩罚。比方说我们的生成器产生够的图像，但是却给了它猫的标签。在这种情况下，生成器会受到判别器的惩罚，因为判别器会注意到生成器产生的毛和真实的猫不同，因为它们具有不同的标签：

```python
def train(self, save_every_n=1000):
    losses = []
    step = 0
    epoch_count = self.epochs
    batch_size = self.batch_size
    z_dim = self.z_dim
    learning_rate = self.learning_rate
    get_batches = self.dataset.get_batches
    classes = len(self.dataset.classes)
    data_image_mode = self.dataset.image_mode
    
    cases, image_width, image_height,\
    out_channel_dim = self.dataset.shape
    input_real, input_z, labels, learn_rate, d_loss,\
    g_loss, d_train_opt, g_train_opt = self.optimization()
    
    # Allowing saving the trained GAN
    saver = tf.train.Saver(var_list=self.g_vars)
    
    # Preparing mask for plotting progression
    rows, cols = min(5, classes), 5
    target = np.array([self.dataset.one_hot[i] \
    	for j in range(cols) for i in range(rows)])
                        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            for batch_images, batch_labels \
                in get_batches(batch_size):
                # Counting the steps
                step += 1
                # Defining Z
                batch_z = np.random.uniform(-1, 1, size=\
                             (len(batch_images), z_dim))
                batch_z = np.concatenate((batch_z,\
                            batch_labels), axis=1)
                # Reshaping labels for generator
                batch_labels = batch_labels.reshape(batch_size, 1, 1,
                            classes)
                batch_labels = batch_labels * np.ones((batch_size,
                            image_width, image_height, classes))
                # Sampling random noise for G
                 batch_images = batch_images * 2
                # Running optimizers
               _ = sess.run(d_train_opt, feed_dict={input_real:
                            batch_images, input_z: batch_z, labels:
                            batch_labels, learn_rate: learning_rate})
                _ = sess.run(g_train_opt, feed_dict={input_z: batch_z,
                                input_real: batch_images,
                                labels:
                                batch_labels, learn_rate: learning_rate})
            # Cyclic reporting on fitting and generator output
                if step % (save_every_n//10) == 0:
                    train_loss_d = sess.run(d_loss,
                                        {input_z: batch_z,
                                        input_real: batch_images, labels: batch_labels})
                                        train_loss_g = g_loss.eval({input_z: batch_z, labels:
                                        batch_labels})
                    print("Epoch %i/%i step %i..." % (epoch_i + 1,
                                        epoch_count, step),
                                        "Discriminator Loss: %0.3f..." %
                                        train_loss_d,
                                        "Generator Loss: %0.3f" % train_loss_g)
                if step % save_every_n == 0:
                    rows = min(5, classes)
                    cols = 5
                    target = np.array([self.dataset.one_hot[i] for j in range(cols) for i in range(rows)])
                    self.show_generator_output(sess, rows * cols, input_z,
                    target, out_channel_dim, data_image_mode)
                    saver.save(sess,
                    '/'+self.generator_name+'/generator.ckpt')
                    # At the end of each epoch, get the losses and print them out
            try:
                train_loss_d = sess.run(d_loss, {input_z: batch_z,
                input_real: batch_images, labels: batch_labels})
                train_loss_g = g_loss.eval({input_z: batch_z, labels:
                batch_labels})
                print("Epoch %i/%i step %i..." % (epoch_i + 1, epoch_count,
                step),
                "Discriminator Loss: %0.3f..." % train_loss_d,
                "Generator Loss: %0.3f" % train_loss_g)
            except:
                train_loss_d, train_loss_g = -1, -1
                # Saving losses to be reported after training
            losses.append([train_loss_d, train_loss_g])

                # Final generator output
        self.show_generator_output(sess, rows * cols, input_z, target,
        out_channel_dim, data_image_mode)
        saver.save(sess, '/' + self.generator_name + '/generator.ckpt')

    return np.array(losses)
```

在训练过程中，网络不断地被保存到磁盘。当需要生成新图像时，读者不需要重新训练，只需要加载网络并指定读者希望GAN产生的图像的标签：

```python
def generate_new(self, target_class=-1, rows=5, cols=5, plot=True):
    """
    Generating a new sample
    """
    # Fixing minimum rows and cols values
    rows, cols = max(1, rows), max(1, cols)
    n_images = rows * cols
    # Checking if we already have a TensorFlow graph
    if not self.trained:
        # Operate a complete restore of the TensorFlow graph
        tf.reset_default_graph()
        self._session = tf.Session()
        self._classes = len(self.dataset.classes)
        self._input_z = tf.placeholder(tf.float32, (None, self.z_dim +
        self._classes), name='input_z')
        out_channel_dim = self.dataset.shape[3]
        # Restoring the generator graph
        self._generator = self.generator(self._input_z,
        out_channel_dim)
        g_vars = [v for v in tf.trainable_variables() if
        v.name.startswith('generator')]
        saver = tf.train.Saver(var_list=g_vars)
        print('Restoring generator graph')
        saver.restore(self._session,
        tf.train.latest_checkpoint(self.generator_name))
        # Setting trained flag as True
        self.trained = True
        
    # Continuing the session
    sess = self._session
    # Building an array of examples examples
    target = np.zeros((n_images, self._classes))
    for j in range(cols):
        for i in range(rows):
        if target_class == -1:
            target[j * cols + i, j] = 1.0
        else:
            target[j * cols + i] =
            self.dataset.one_hot[target_class].tolist()
    # Generating the random input
    z_dim = self._input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1,
    size=[n_images, z_dim - target.shape[1]])
    example_z = np.concatenate((example_z, target), axis=1)
    # Generating the images
    sample = sess.run(
                self._generator,
                feed_dict={self._input_z: example_z})
    # Plotting
    if plot:
        if rows * cols==1:
            if sample.shape[3] <= 1:
                images_grid = sample[0,:,:,0]
            else:
                images_grid = sample[0]
            else:
                images_grid = self.images_grid(sample, cols)
        plt.imshow(images_grid, cmap=self.cmap)
        plt.show()
    # Returning the sample for later usage
    # (and not closing the session)
    return sample
```

这一类由fit方法完成，可以接受学习率参数和beta1（Adam优化器的参数，基于初始的平均值调节学习率参数），并在训练完成后绘制来自判别器和生成器的损失结果：

```python
def fit(self, learning_rate=0.0002, beta1=0.35):
    """
    Fit procedure, starting training and result storage
    """
    # Setting training parameters
    self.learning_rate = learning_rate
    self.beta1 = beta1
    # Training generator and discriminator
    with tf.Graph().as_default():
        train_loss = self.train()
    # Plotting training fitting
    plt.plot(train_loss[:, 0], label='Discriminator')
    plt.plot(train_loss[:, 1], label='Generator')
    plt.title("Training fitting")
    plt.legend()
```

##将CGAN应用于一些实例 

既然已经完成了CGAN的类，下面我们通过一些例子为读者提供如何使用这一项目的新鲜的想法。首先，我们需要为下载必要的数据和训练我们的GAN做好准备。我们导入程序库开始：

```python
import numpy as np 
import urllib.request
import tarfile 
import os
import zipfile
import gzip 
import os from glob
import glob from tqdm 
import tqdm
```

然后，我们载入数据集和之前准备好的CGAN：

```python
from cGAN import Dataset, CGAN
```

类`TqdmUpTo`是一个`tqdm`包装，可以显示下载进度。这个类直接来自于项目的主页https://github.com/tqdm/tqdm:

```python
class TqdmUpTo(tqdm):
    """
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Inspired by https://github.com/pypa/twine/pull/242
    https://github.com/pypa/twine/commit/42e55e06
    """
    def update_to(self, b=1, bsize=1, tsize=None):
    """
    Total size (in tqdm units).
    If [default: None] remains unchanged.
    """
    if tsize is not None:
        self.total = tsize
    # will also set self.n = b * bsize
    self.update(b * bsize - self.n
```

最后，如果我们使用`jupyter notebook`（强烈建议尝试），读者必须启用图像的内联绘制

```python
%matplotlib inline
```

 我们现在为开始第一个例子做好了准备。 

##MNIST

MNIST手写数字数据集由Yann LeCun在NYU的ourant研究所， Corinna Cortes（谷歌实验室） Christopher J.C.Burges （微软研究院）时提供。它被认为是学习真实世界图像数据的标准，只需要进行少量的预处理和格式化。数据集由手写数字组成，提供了60000个训练样本，10000个测试样本。它实际上是更大规模的NIST数据集的一个子集。所有数字的尺寸已经被标准化，并处于固定尺寸的图像中央：

http://yann.lecun.com/exdb/mnist/

<img src="PATH\figures\97_1.jpg" />

​				图5：:MNIST原始数据集的示例，帮助理解由CGAN复制的图像的质量。

第一步，我们从网上加载数据集并储存到本地：

```python
labels_filename = 'train-labels-idx1-ubyte.gz'
images_filename = 'train-images-idx3-ubyte.gz'
url = "http://yann.lecun.com/exdb/mnist/"
with TqdmUpTo() as t: # all optional kwargs
    urllib.request.urlretrieve(url+images_filename,
                                'MNIST_'+images_filename,
                                reporthook=t.update_to, data=None)
with TqdmUpTo() as t: # all optional kwargs
    urllib.request.urlretrieve(url+labels_filename,
                                'MNIST_'+labels_filename,
                                reporthook=t.update_to, data=None)
```

为了学习这一手写数字集合，我们将一批设为32张图像，学习率为0.0002，`beta1`值为0.35，`z_dim`为96，训练15轮：

```python
labels_path = '/MNIST_train-labels-idx1-ubyte.gz'
images_path = '/MNIST_train-images-idx3-ubyte.gz'
with gzip.open(labels_path, 'rb') as lbpath:
    labels = np.frombuffer(lbpath.read(),
                            dtype=np.uint8, offset=8)
with gzip.open(images_path, 'rb') as imgpath:
    images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                            offset=16).reshape(len(labels), 28, 28, 1)
batch_size = 32
z_dim = 96
epochs = 16

dataset = Dataset(images, labels, channels=1)
gan = CGAN(dataset, epochs, batch_size, z_dim, generator_name='mnist')

gan.show_original_images(25)
gan.fit(learning_rate = 0.0002, beta1 = 0.35)
```

下面的图像展示了GAN在第二轮和最后一轮生成的数字的示例：

<img src="PATH\figures\98_1.jpg" />

​									图6：经过若干轮训练后GANs的结果

经过16轮后，数字表现出较好的形状并且可以使用。下面，我们提取按行排列的所有类的示例。

评估GaN的性能通常仍然是由人类法官对一些结果进行视觉检查的问题，试图从整体上或通过精确地揭示细节来判断图像是否可能是假的(像判别器一样)。GANs缺乏客观的功能来评价和比较它们，尽管有一些计算技术可以用作度量，如对数似然，正如THEIS, Lucas; OORD, Aäron van den; BETHGE, Matthias.A
note on the evaluation of generative models.arXiv preprint arXiv:1511.01844,
2015: https://arxiv.org/abs/1511.01844.中所描述的。

我们将保持评估的简单性和经验性，因此我们将使用一个由经过训练的GaN生成的图像样本来评估网络的性能，我们还将尝试检查生成器和鉴别器的训练损失，以便发现任何特定的趋势：

<img src="PATH\figures\99_1.jpg" />

​	图7：对MNIST进行培训后的最终结果的样本显示，这对一个GaN网络来说是一项可完成的任务。

观察下图所示的训练拟合图，我们注意到训练结束时生成器是如何达到最低误差的。判别器，在上一个峰值之后，正在努力回到它以前的性能值，指出了一个可能的生成器的突破。我们可以预期，更多的训练周期可以改善这个GaN网络的性能，但是随着输出质量的提高，可能需要花费成倍的时间。一般而言，一个很好的GaN收敛指标是判别器和生成器都有下降的趋势，这可以通过将线性回归线拟合到两个损失向量来推断：

<img src="PATH\figures\100_1.jpg" />

​										图8:16轮训练的拟合情况

训练一个惊人的GaN网络可能需要很长的时间和大量的计算资源。通过阅读这篇纽约时代上的最新文章，https://www.nytimes.com/interactive/2018/01/02/technology/ai-generated-photos.html，读者可以从NVIDIA公司找到一张图表，显示在从名人照片中学习一种进步的GaN的时间上所取得的进展。虽然可能需要几天的时间才能得到一个好的结果，但对于一个令人惊讶的结果，读者至少需要两周的时间。同样地，即使用我们的例子，读者投入的训练时间越多，效果就越好。

##Zalando MNIST

流行的MNIST是Zalano的文章中的图像数据集，由包含60000样本的训练集和10000样本的测试集组成。在MNIST数据集中，每个样本都是28*28的灰度图，与一个十分类的标签相关联。来自Zalando Research(https://github.com/zalandoresearch/fashion-mnist/graphs/contributors)的原作者的意图是作为原始mnist数据集的替代，以便更好地测试机器学习算法，因为在实际任务中学习更具有挑战性，而且更能代表深度学习(https://twitter.com/fchollet/status/852594987527045120).。

https://github.com/zalandoresearch/fashion-mnist

<img src="PATH\figures\101_1.jpg" />

​									图9：原始Zalando 数据集的样例

我们分别下载数据和它们的标签：

```python
url = "http://fashion-mnist.s3-website.eu-central-\        
			1.amazonaws.com/train-images-idx3-ubyte.gz"
filename = "train-images-idx3-ubyte.gz" with TqdmUpTo() as t: # all optional kwargs     urllib.request.urlretrieve(url, filename,reporthook=t.update_to, data=None)

url = "http://fashion-mnist.s3-website.eu-central-\        
			1.amazonaws.com/train-labels-idx1-ubyte.gz"
filename = "train-labels-idx1-ubyte.gz" _ = urllib.request.urlretrieve(url, filename)

```

为了学习这一手写数字集合，我们将一批设为32张图像，学习率为0.0002，`beta1`值为0.35，`z_dim`为96，训练10轮：

```python
labels_path = '/train-labels-idx1-ubyte.gz'
images_path = '/train-images-idx3-ubyte.gz'
label_names = ['t_shirt_top', 'trouser', 'pullover',
'dress', 'coat', 'sandal', 'shirt',
'sneaker', 'bag', 'ankle_boots']
with gzip.open(labels_path, 'rb') as lbpath:
    labels = np.frombuffer(lbpath.read(),
    dtype=np.uint8,
    offset=8)
with gzip.open(images_path, 'rb') as imgpath:
    images = np.frombuffer(imgpath.read(), dtype=np.uint8,
    offset=16).reshape(len(labels), 28, 28, 1)
batch_size = 32
z_dim = 96
epochs = 64

dataset = Dataset(images, labels, channels=1)
gan = CGAN(dataset, epochs, batch_size, z_dim, generator_name='zalando')

gan.show_original_images(25)
gan.fit(learning_rate = 0.0002, beta1 = 0.35
```

训练需要很长时间来完成所有轮，但是质量显示迅速稳定，尽管一些问题需要更多的轮来解决（例如T恤中的洞）：

<img src="PATH\figures\103_1.jpg" />

​									图10： CGAN训练的各轮的变化

这是64轮后的结果：

<img src="PATH\figures\103_2.jpg" />

​						图11： Zalando数据集经过64轮训练后达到的结果概览

结果非常令人满意，尤其是衣服和男鞋。然而，女鞋似乎更难被学习，因为与其他图像相比，它们更小，并且具有更多细节。


##EMNIST

EMNIST数据集是从NIST特殊数据库派生的一组手写字符数字，转换为直接匹配MNIST数据集的28x28像素图像格式和数据集结构。我们将使用EMNIST Balance，这是一组字符，每个类有相同数量的样本，它由131600个字符组成，分布在47个平衡类中。读者找到关于这个数据集的所有参考资料，在：

>  Cohen, G., Afshar, S., Tapson, J., & van Schaik, A.(2017).EMNIST: an extension of MNIST to handwritten letters.Retrieved from http://arxiv.org.abs1702.05373.

读者也可以通过浏览EMNIST数据集的官方主页https://www.nist.gov/itl/iad/image- group/emnist- dataset来探索关于它的全部信息。下面是在EMNIST Balance中可以找到的字符类型的提取：

<img src="PATH\figures\104_1.jpg" />

​									图11：原始EMNIST数据集的示例

```python
url = "http://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
filename = "gzip.zip"
with TqdmUpTo() as t: 
    # all optional kwargs     
    urllib.request.urlretrieve(url, filename,reporthook=t.update_to, data=None)
```

在从NIST网站下载后，我们解压下载的包：

```
zip_ref = zipfile.ZipFile(filename, 'r')
zip_ref.extractall('.') zip_ref.close()
```

在检查解压成功后，我们删除无用的ZIP文件：

```python
if os.path.isfile(filename):     
	os.remove(filename)
```

为了学习这一手写数字集合，我们将一批设为32张图像，学习率为0.0002，`beta1`值为0.35，`z_dim`为96，训练10轮：

```python
labels_path = '/gzip/emnist-balanced-train-labels-idx1-ubyte.gz'
images_path = '/gzip/emnist-balanced-train-images-idx3-ubyte.gz'
label_names = []
with gzip.open(labels_path, 'rb') as lbpath:
    labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
    offset=8)
with gzip.open(images_path, 'rb') as imgpath:
    images = np.frombuffer(imgpath.read(), dtype=np.uint8,
    offset=16).reshape(len(labels), 28, 28, 1)
batch_size = 32
z_dim = 96
epochs = 32
dataset = Dataset(images, labels, channels=1)
gan = CGAN(dataset, epochs, batch_size, z_dim,
generator_name='emnist')
gan.show_original_images(25)
gan.fit(learning_rate = 0.0002, beta1 = 0.35)
```

下面是完成32轮训练后生成的手写字母的示例：

<img src="PATH\figures\106_1.jpg" />

​						图12：在EMNIST数据集上对CGAN进行培训的结果概览

对于MNIST，GAN可以在合理的时间内学会以准确和可信的方式复制手写的字母。



##重用训练好的CGAN

训练CGAN后，读者可能会发现在其他应用程序中使用生成的图像很有用。`generate_new`函数可以用于提取单个图像或图像集合（为了检验特定图像类生成的结果的质量）。它在以前训练过的CGAN类上运行，所以读者所要做的就是将其进行`pickle`，以便首先保存它，然后在需要时再次恢复它。
当训练完成时，读者可以用pickle保存读者的CGAN类，正如下面的命令所示：

```python
import pickle 
pickle.dump(gan, open('mnist.pkl', 'wb'))
```

在这一案例中，我们保存从MNIST数据集训练的CGAN。

当读者重启Python会话后，内存中所有变量被清除，读者可以再次导入所有的类，并恢复pickled的CGAN：

```python
from CGan import Dataset, CGan
import pickle 
gan = pickle.load(open('mnist.pkl', 'rb'))
```

完成后，读者可以设置读者需要CGAN生成的目标类（在这个例子中，我们要求打印数字8），读者可以生成一个例子，一个5*5的网格的粒子或更大规模的10 *10的网格：

```python
nclass = 8
_ = gan.generate_new(target_class=nclass,
	rows=1, cols=1, plot=True)
_ = gan.generate_new(target_class=nclass,
	rows=5, cols=5, plot=True)
images = gan.generate_new(target_class=nclass,
	rows=10, cols=10, plot=True)
print(images.shape)
```

如果读者只想获得所有类的概览，只需要把参数 `target_class` 设为-1       

在设置了要表示的目标类之后，`generate_new `被调用3次，最后一次返回的值被存在变量images中，其尺寸为 (100, 28, 28, 1) ，包含一个生成图像的numpy矩阵，它可以为我们的目标所重用。每次读者调用这一方法，都会绘制出一个网格的结果，正如下图所示：

<img src="PATH\figures\107_1.jpg" />

图13：所绘制的网格是所生成图像的组合，即图像本身。从左到右，图中要求一个1x1，5x5，10x10格的结果。实际图像由该方法返回并可重用。

如果读者不需要 `generate_new`来绘制结果，读者可以简单地将绘制参数设为False：

```python
images = gan.generate_new(target_class=nclass, rows=10, cols=10, plot=False).
```



##使用亚马逊网络服务

如前所述，我们热烈建议读者使用GPU来训练本章中给出的示例。仅仅使用CPU在合理的时间内获得结果确实是不可能的，而且使用GPU可能会变成等待计算机完成训练的所需要的相当长的时间。一个需要付费的解决方案是，使用**亚马逊弹性计算云**（AmazonElasticComputeCloud），即AmazonEC 2(https://aws.amazon.com/it/ec2/)，它是**亚马逊网络服务**（Amazon Web Services，*AWS*)的一部分。在EC2上，读者可以启动虚拟服务器，从读者的计算机通过网络连接来控制它。读者可以在EC2上申请强大的GPU服务器，并且使读者的Tensorflow项目生涯更加轻松。

AmazonEC 2并不是唯一的云服务。我们向读者推荐这个服务，因为它是我们用来测试本书中的代码的服务。事实上，还有很多选择，比如谷歌云计算 (cloud.google.com)，Microsoft Azure (azure.microsoft.com) 和其他等等。
在EC2上运行本章的代码需要有一个AWS账户。如果没有，第一步是注册aws.amazon.com，完成所有必要的表格，并开始一个免费的基本支持计划。
在读者注册AWS之后，读者可以登陆并访问EC2页面(https://aws.amazon.com/ec2)。在这里读者将：从 EU (Ireland), Asia Pacific (Tokyo), US East (N.Virginia) 和 US West (Oregon)选择一个既离读者近又便宜并且允许我们需要的GPU实例类型的地区。

2 https://console.aws.amazon.com/ec2/v2/home? #Limits更新读者的服务限制报告。读者需要访问一个p3.2xlarge的实例。因此，如果读者的实际限制为零，则应该使用请求限制增长表单将其至少取为1（这可能需要24小时，但在它完成之前读者无法访问该类型的实例）。

3 获得一些AWS信用积分(例如提供信用卡)。

设置区域并增加足够的信用和请求限制后，读者可以启动`p3.2xlarger`服务器（一个为深度学习应用提供的GPU计算服务器），操作系统已经准备好，包含读者需要的所有软件（感谢AMI，亚马逊准备的图像）：

1. 开始`EC2 Management Console`，点击 `Launch Instance`按钮

2. 点击`AWS Marketplace`，搜索 Deep Learning AMI with Source Code v2.0 (ami-bcce6ac4) AMI。这一AMI好好所有预先安装的软件：CUDA, cuDNN (https://developer.nvidia.com/cudnn), Tensorflow。

3. 选择GPU计算`p3.2xlarge`实例。这一实例拥有强大的NVIDIA Tesla V100 GPU。

4. 通过添加`Custom TCP Rule`来认证一个安全的集群（读者可以调用jupyter），这是一个TCP协议，在8888端口上，可以在任何地方访问。这允许读者在机器上运行jupyter并在任何连网的计算机上看到界面。

5. 创建身份验证密钥对。例如，您可以将其命名为`DECHPLICIN_JUPYTER.pem`。将它保存在您的计算机上一个您可以轻松访问的目录中。

6. .启动实例。记住读者需要从此刻开始付费，直到读者在AWS菜单中停止它——读者仍然会承担一些费用，但费用较小，读者将可以保存实例与读者的所有数据-或简单地终止它，不再为它支付任何费用。

   ​

   .在一切启动之后，您可以使用ssh从计算机访问服务器。

   * 注意机器的IP。例如 xx.xx.xxx.xxx
   * 从一个shell指向.pem文件所在的目录。例如ssh -i deeplearning_jupyter.pem ubuntu@ xx.xx.xxx.xxx
   * 当您访问了服务器计算机后，通过键入以下命令来配置它的jupyter服务器：

   ```
   jupyter notebook --generate-config
   sed -ie "s/#c.NotebookApp.ip = 'localhost'/#c.NotebookApp.ip =
   '*'/g" ~/.jupyter/jupyter_notebook_config.py

   ```

   * 通过复制代码来操作服务器（例如，通过git克隆代码存储库）并存储读者可能需要的所有库。例如，读者可以为特定项目存储下面的包：

   ```
   sudo pip3 install tqdm 
   sudo pip3 install conda
   ```

   * 用下面的命令启动jupyter服务：

   ```
   jupyter notebook --ip=0.0.0.0 --no-browser
   ```

   * 此时，服务器将运行，读者的ssh shell将提示读者从Jupyter获得日志。在这些日志中，记录令牌（它是一个由数字和字母组成的序列）
   * 打开读者的浏览器并在地址栏输入：

   ```
   http://xx.xx.xxx.xxx:8888/
   ```

   当需要时，输入令牌，读者可以在本地机器上使用Jupyter Notebook，但实际上它在服务器上运行。此时，读者拥有一个强大的GPU服务器，可以运行读者在GANs上的所有实验、

   ​

##致谢

在结束本章时，我们要感谢Udca和Mat Leonard提供的DCGAN教程，该教程是由MIT授权的（https://github.com/udacity/deep-learning/blob/master/LICENSE），它为本项目提供了一个良好的起点和基准。

##总结

在本章中，我们详细讨论了生成对抗网络的主题，它们是如何工作的，以及它们如何被训练和用于不同的目的。作为一个项目，我们创建了一个有条件的GAN，它可以根据读者的输入生成不同类型的图像，并且我们学习了如何处理一些示例数据集并对它们进行培训，以便有一个能够根据需要创建新图像的pickable类。