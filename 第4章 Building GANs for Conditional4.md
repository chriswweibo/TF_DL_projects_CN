#第4章 为条件图像生成建立GAN

Yann LeCun，Facebook人工智能的领导者，最近提出“生成对抗网络是机器学习中十年来最有趣的想法“， 这一观点也由于引起了学界对这一深度学习方法的浓厚兴趣而得以证实。如果你阅读了最近有关深度学习的论文（          同时也要注意LinkedIn或中级职位的主要趋势），就会发现GANs已经产生了非常多的变体。

你可以通过Hindu Puravinash,创建的、支持更新的参考文献列表来了解GANs的世界的发展，详见https:/ / github. com/ hindupuravinash/ the- gan- zoo/ blob/ master/ gans. tsv；也可以通过学习Zheng Liu准备的GANs的时间线https:/ /github. com/ dongb5/ GAN- Timeline来达到这一目的，它可以帮助你把所有相关内容放在时间的角度。

GANS有能力去激发想象力，因为它们能展示人工智能的创造力，而不仅仅是它的计算能力。在本章中，我们将： 

​	揭开GANS的神秘面纱，为你提供所有必要的概念来理解GANs是什么，GANs现在能做什么，以及GANs期望做什么。 

​	演示如何基于初始的示例图像来生成图像（即所谓的非监督的GANs）

​	解释如何使利用GANs为你生成你所需要的类型的结果图。

将GaNs与所期望的图像生成的条件相结合。 

​	建立一个基本的、完整的项目，可以处理不同的手写字符和图标数据集。 

​	为你提供如何在云（特别是亚马逊AWS）上训练你的GANs的基本指令              

​	GANs的成功很大程度上取决于你使用的特定神经网络结构，以及它们所面对的问题和你提供给它们的数据。 本章中我们选择的数据集可以提供满意的结果。我们希望你能享受并被GAN的创造力所激励！


## GANs简介

我们将从最近的一些历史开始，因为GANs是你能发现的整个人工智能和深度学习领域最新的想法之一。

一切起源于2014年，Ian Goodfellow和他的同事（Yoshua Bengio也在贡献者名单中）在蒙特利尔大学的Departement d'informatique et de recherche发表了一篇关于生成对抗网络（GANs）的论文，提出了能够基于一系列初始样例生成新数据的框架：、

​	GOODFELLOW, Ian, et al. Generative Adversarial Nets. In: Advances in
	Neural Information Processing Systems. 2014. p. 2672-2680: https:/ / arxiv.
	org/ abs/ 1406. 2661.

考虑到以前使用马尔可夫链的尝试远不能令人信服，这种网络所产生的初始图像是惊人的。 在下图中，你剋看到

论文中提出的一些示例，分别来自于MNIST，多伦多人脸数据集（Toronto Face Dataset，TFD），一个非公开数据集和cifar10数据集。

图1：GANs的第一篇论文中采用不同数据集生成新图像的样本 a) MNIST b) TFD c) and d) CIFAR-10

来源：GOODFELLOW, Ian, et al. Generative Adversarial Nets. In: Advances in Neural Information Processing Systems. 2014. p. 2672-2680

这篇论文被认为是颇具创新性的，因为它把一个非常深的神经网络和博弈理论结合在一个真正智能的体系结构中，它不需要比通常的反向传播需要更多的训练。 GANs是生成类的模型，模型可以生成数据，因为它们刻画了模型分布图（例如，它们学习它）。因此，当它们生成某种东西时，就好像它们是从那个分布中取样一样。 

## 关键在于对抗方式

理解GANs为何能成为如此成功的生成模型在于对抗。事实上，GAN是的结构由两个不同的网络组成，它们基于各误差的汇集进行优化，这个过程就是对抗过程。

你可以从一个真实数据集着手，我们称之为R，它包含你的不同种类的图像或数据（GANs不局限于图像，尽管这是它们主要的应用）。之后你可以建立生成网络G，用来生成与真实数据尽可能相似的伪造数据；你还需要建立一个判别器D，用来比较G生成的数据和真实数据，指出数据是否真实。
Goodfellow用艺术伪造这一比喻来描述生成器，而判别器是侦探（或是艺术评论家），必须揭露他们的罪行。伪造者和侦探之间存在一个挑战，因为伪造者需要更有技巧以便不被侦探发现，而侦探在寻找伪造者方面也需要提升。伪造者和侦探之间的一切无休无止，知道伪造的产品与原始的完全相似。事实上，当GAN是过拟合时，它们会重新输出原始数据。这似乎可以解释为竞争市场，而它也确实是，因为GANs的想法就起源于竞争博弈论。
	在GAN是中，生成器产生图像，直到判别器无法分辨它们的真假。生成器的一个明显的解决方案是简单地复		

​	制一些训练图像或采用判别器无法判断的看起来成功的生成图像。我们的项目中将应用的是单面标签平滑技

​	术的解决方案。具体描述见SALIMANS, Tim, etal. Improved techniques for training gans. In: Advances in 		

​	Neural Information Processing Systems. 2016. p. 2234-2242: https:/ / arxiv. org/abs/ 1606. 03498.

下面我们讨论GANs如何运作。首先，生成器G没有线索，完全随机地生成数据（事实上它甚至不考虑原始数据），因此它会被判别器D惩罚——此时分辨真实数据和伪造数据是很容易的。from D.  G承担全部责任，开始尝试不同的东西以获得更好的D反馈。这一过程也是随机完成的，因为G能看见的是随机的输入，Z，它无法接触真实的数据。在很多次尝试和失败后，在判别器的指导下，生成器最终会指出如何做并开始生成可靠的输出。最后，经过足够的时间，生成器会生成器将完全复制所有原始数据，即使它并没有见到过其中任何一个示例。



##寒武纪爆发

正如上文提到的，关于GANs的新论文每个月都在产生（你可以在Hindu Puravinash上查找文献列表，在本章的开头我们已经提到过）。

不管怎样，除了 Goodfellow和他的同事们最初的描述实施方法的论文之外，值得注意的最著名的的实施方案是深度卷积对抗网络（DCGANs）和条件生成对抗网络（CGANs）。
	DCGANs是基于CNN结构的GANs(RADFORD, Alec; METZ, Luke;
	CHINTALA, Soumith. Unsupervised representation learning with deep convolutional
	generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015: https:/ /
	arxiv. org/ abs/ 1511. 06434).

​	CGANs是在DAGANs的基础上，在输入标签上增加了一些条件，从而使得你得到的结果中包含想得到的特

​	征(MIRZA, Mehdi;OSINDERO, Simon. Conditional generative adversarial nets. arXiv preprint
	arXiv:1411.1784, 2014: https:/ / arxiv. org/ abs/ 1411. 1784)。我们的项目会编写一个CGANs的类并在不同

​	的数据集上训练它，来证明其功能。

 此外也有一些有趣的例子（我们的项目中不包括这些），给出了创造图像或提升的问题的实际解决方案：

​	CycleGAN，将一幅图像转化为另一幅（经典的例子是将马变为斑马: ZHU, Jun-Yan, et al. Unpaired image-	to-image translation using cycle-consistent adversarial networks. arXiv preprint arXiv:1703.10593, 2017: https:/ /arxiv. org/ abs/ 1703. 10593)

​	StackGAN可以根据描述图像的文本生成图像 (ZHANG, Han, et al. Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks. arXiv preprint arXiv:1612.03242, 2016: https:// arxiv	.org	/ abs/1612	.0324	2)

​	DiscoveryGAN (DiscoGAN) 传递不同图像之间的风格元素，	从而可以将一个时尚品，比如背包的纹理和装饰传递到一双鞋上(KIM, Taeksoo, et al. Learning to discover cross-domain relations with generative adversarial networks. arXiv preprint arXiv:1703.05192, 2017: https:// arxiv	.org	/abs	/1703	.0519	2)

​	SRGAN把低质量的图像转化为高分辨率图像(LEDIG, Christian, et al. Photo-realistic single image super-resolution using a generative adversarial network. arXiv preprint arXiv:1609.04802, 2016: https:// arxiv	.org	/ abs/1609	.0480	2)



##DCGANs

DCGANs是GANs结构的第一个提升。DCGANs成功完成训练阶段，给出足够的轮和栗子，它们倾向于得到质量令人满意的输出。这使得它们很快成为GANs的基准并且有助于产生令人惊喜的成果，例如根据已知神奇宝贝产生新的：https:/ / www. youtube. com/ watch? v= rs3aI7bACGc或创造一些名人的脸，它们事实上并不存在但看起来像是真的（并不神秘），正如NVIDIA所做的：https:/ / youtu. be/ XOxxPcy5Gr4，用一个名为progressing growing:的新的训练方法，详见http:/ / research. nvidia. com/ sites/ default/ files/
publications/ karras2017gan- paper. pdf。它们的根源在于使用与深度学习监督网络中的图像分类中使用的相同的卷积，并且采用了一些巧妙的技巧： 

​	在两个网络中都使用BatchNormalization

​	没有隐藏的连接层	

​	没有池化，只在卷积层设置步长

​	采用ReLU作为激活函数



##Conditional GANs

在conditional GANs (CGANs),中，增加一个特征向量可以控制输出并更好地引导生成器认识到应该做什么。这样一个特征向量可以编码为图像应该导出的类（图像是女人还是男人，如果我们想创建虚构的演员的面孔）或者是我们希望从图像中得到的一系列特定的特征（对于虚构的演员，可以是发型，眼睛或肤色）。这里的技巧是将信息合并到要学习的图像中并交给Z输入，这里的输入不再完全是随机的。判别器的评价不知需要从原始图像中判断出伪造图像，还需要找到伪造图像对应的标签（或特征）：

图3：将Z输入与Y输入结合（有标签的特征向量）允许生成受约束的图像。



## 项目

在我们开始处导入正确的库。除了Tensoflow之外，我们还需要使用numpy和math进行计算，scipy和matplotlib用来处理图像和图表，warnings、random和distuils来支持特定操作：

​	import numpy as np
	import tensorflow as tf
	import math
	import warnings
	import matplotlib.pyplot as plt
	from scipy.misc import imresize
	from random import shuffle
	from distutils.version import LooseVersion



## 数据集

我们的第一步是提供数据。我们依赖已经完成预处理的数据集，但读者可以在自己的GAN中使用不同种类的图像。我们的想法是保持一个数据集类，它的任务是为我们以后构建的GANS类提供标准化和重构图像的批次。

在初始化时，我们需要同时处理图像和标签（如果存在）。图像首先需要变形（如果它们的形状和示例的类中定义的不同），然后搅乱。比起有序，例如按数据集中初始的类的顺序，搅乱能帮助GANs更好地学习，对于任何基于随机梯度下降（ stochastic gradient descent，sgd）机器学习方法这一点都是成立的：BOTTOU, Léon. Stochastic gradient descent tricks.In: Neural networks: Tricks of the trade. Springer, Berlin, Heidelberg, 2012. p. 421-436: https:/ /www. microsoft. com/ en- us/ research/ wp- content/ uploads/ 2012/ 01/ tricks- 2012. pdf.）。标签采用one-hot encoder编码，为每一个类创建一个二进制变量，将其设置为1（其他类设为0）以保证标签可以转化为向量。例如，如果我们的类是 {dog:0, cat:1}, w我们需要两个 one-hot encoded 向量来表示它们: {dog:[1, 0], cat:[0, 1]}.

用这种方法，我们可以轻松地把向量加入我们的图像，作为另一个通道，并在其中加入一些会被我们的GAN重现的视觉特征。另外，我们可以安排向量的顺序，组成更复杂的有特殊特征的类。比方说，我们可以为我们希望生成的类指定编码，也可以指定它的一些特征：


class Dataset(object):     

def __init__(self, data, labels=None, width=28, height=28,                                     max_value=255, channels=3):
        # Record image specs         self.IMAGE_WIDTH = width         self.IMAGE_HEIGHT = height         self.IMAGE_MAX_VALUE = float(max_value)
        self.CHANNELS = channels
        self.shape = len(data), self.IMAGE_WIDTH,
                                self.IMAGE_HEIGHT, self.CHANNELS         if self.CHANNELS == 3:             self.image_mode = 'RGB'             self.cmap = None         elif self.CHANNELS == 1:             self.image_mode = 'L'             self.cmap = 'gray'
        # Resize if images are of different size         if data.shape[1] != self.IMAGE_HEIGHT or \                             data.shape[2] != self.IMAGE_WIDTH:             data = self.image_resize(data,                    self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
        # Store away shuffled data         index = list(range(len(data)))
        shuffle(index)         self.data = data[index]
        if len(labels) > 0:
            # Store away shuffled labels             self.labels = labels[index]             # Enumerate unique classes             self.classes = np.unique(labels)
            # Create a one hot encoding for each class
            # based on position in self.classes
            one_hot = dict()             no_classes = len(self.classes)             for j, i in enumerate(self.classes):                 one_hot[i] = np.zeros(no_classes)
                one_hot[i][j] = 1.0             self.one_hot = one_hot         else:
            # Just keep label variables as placeholders
            self.labels = None             self.classes = None             self.one_hot = None
    def image_resize(self, dataset, newHeight, newWidth):
        """Resizing an image if necessary"""         channels = dataset.shape[3]         images_resized = np.zeros([0, newHeight,                          newWidth, channels], dtype=np.uint8)         for image in range(dataset.shape[0]):
            if channels == 1:
                temp = imresize(dataset[image][:, :, 0],
                               [newHeight, newWidth], 'nearest')
                temp = np.expand_dims(temp, axis=2)             else:                 temp = imresize(dataset[image],
                               [newHeight, newWidth], 'nearest')             images_resized = np.append(images_resized,                             np.expand_dims(temp, axis=0), axis=0)
        return images_resized

get_batches方法释放了数据集的一个子集并进行标准化，每个像素值除以它们的最大值（256），再减去0.5。结果图中的浮点的值域为 [-0.5, +0.5]：

def get_batches(self, batch_size):
    """Pulling batches of images and their labels"""
    current_index = 0     # Checking there are still batches to deliver     while current_index < self.shape[0]:         if current_index + batch_size > self.shape[0]:             batch_size = self.shape[0] - current_index         data_batch = self.data[current_index:current_index \                                + batch_size]         if len(self.labels) > 0:             y_batch = np.array([self.one_hot[k] for k in \             self.labels[current_index:current_index +\
            batch_size]])         else:             y_batch = np.array([])         current_index += batch_size         yield (data_batch / self.IMAGE_MAX_VALUE) - 0.5, y_batch



##CGAN class

基于CGAN模型的类包含运行CGAN所需要的所有函数。DCGANs被证明可以生成类似于照片质量的输出的性能。我们已经介绍过CGAN，为了提醒你，参考文献如下：

​	RADFORD, Alec; METZ, Luke; CHINTALA, Soumith. Unsupervised
	representation learning with deep convolutional Generative Adversarial
	Networks. arXiv preprint arXiv:1511.06434, 2015 at https:/ / arxiv. org/
	abs/ 1511. 06434.

使用标签并将它们与图像集成（这是诀窍）。将导致更好的图像和决定生成图像的特征的可能性。 
conditional GANs 的参考文献：
	MIRZA, Mehdi; OSINDERO, Simon. Conditional Generative Adversarial Nets. arXiv preprint 	arXiv:1411.1784, 2014, https:/ /	arxiv	.	org	/	abs	/	1411	.
1784.

我们的CGAN希望输入数据集类对象，轮数，图像batchsize，用于生成的输入的图像维数（z_dim）和GAN的名字（便于保存）。它会采用不同的值初始化，达到alpha或平滑。后面我们会讨论这两种参数对GAN网络的影响。

下面的示例设置了所有内部参数并在系统上检查性能，如果没有检测到GPU则给出警告：

class CGan(object):     

​	def __init__(self, dataset, epochs=1, batch_size=32,                  z_dim=96, generator_name='generator',
                 alpha=0.2, smooth=0.1,                  learning_rate=0.001, beta1=0.35):
        # As a first step, checking if the
        # system is performing for GANs         self.check_system()
        # Setting up key parameters         self.generator_name = generator_name
        self.dataset = dataset         self.cmap = self.dataset.cmap         self.image_mode = self.dataset.image_mode
        self.epochs = epochs         self.batch_size = batch_size
        self.z_dim = z_dim         self.alpha = alpha         self.smooth = smooth
        self.learning_rate = learning_rate
        self.beta1 = beta1         self.g_vars = list()         self.trained = False
    def check_system(self):
        """
        Checking system suitability for the project
        """
        # Checking TensorFlow version >=1.2
        version = tf.__version__
        print('TensorFlow Version: %s' % version)
        assert LooseVersion(version) >= LooseVersion('1.2'),\         ('You are using %s, please use TensorFlow version 1.2 \                                          or newer.' % version)
        # Checking for a GPU         if not tf.test.gpu_device_name():             warnings.warn('No GPU found installed on the system.\                            It is advised to train your GAN using\
                           a GPU or on AWS')         else:             print('Default GPU Device: %s' % tf.test.gpu_device_name())



instantiate_inputs函数未输入创建Tensorflow占位符，是一个随机实数。它也提供标签（创建与原始图像形状相同但通道数量等于类数的图像），对于训练过程的学习率：

​    def instantiate_inputs(self, image_width, image_height,                            

​	image_channels, z_dim, classes):         """
        Instantiating inputs and parameters placeholders:         

​	real input, z input for generation,         

​	real input labels, learning rate
        """
        inputs_real = tf.placeholder(tf.float32,
                       (None, image_width, image_height,                         image_channels), name='input_real')
        inputs_z = tf.placeholder(tf.float32,
                       (None, z_dim + classes), name='input_z')
        labels = tf.placeholder(tf.float32,
                        (None, image_width, image_height,                          classes), name='labels')         

​	learning_rate = tf.placeholder(tf.float32, None)         

​	return inputs_real, inputs_z, labels, learning_rate

下面，我们的工作转向网络结构，定义一些基本的函数例如leaky_ReLU_activation函数（我们将会在判别器和生成器中使用它，与深度卷积GANs的原始论文中描述的相反）：

 def leaky_ReLU_activation(self, x, alpha=0.2):      return tf.maximum(alpha * x, x)  def dropout(self, x, keep_prob=0.9):
     return tf.nn.dropout(x, keep_prob)

我们的下一个函数展示了一个判别器的层。它用Xavier初始化创建一个新的层，对结果执行batch normalization，设置a leaky_ReLU_activation，最后应用dropout做正则化：

​    def d_conv(self, x, filters, kernel_size, strides,                padding='same', alpha=0.2, keep_prob=0.5,                train=True):
        """
        Discriminant layer architecture
        Creating a convolution, applying batch normalization,
        leaky rely activation and dropout
        """
        x = tf.layers.conv2d(x, filters, kernel_size,                           strides, padding, kernel_initializer=\                           tf.contrib.layers.xavier_initializer())         x = tf.layers.batch_normalization(x, training=train)
        x = self.leaky_ReLU_activation(x, alpha)
        x = self.dropout(x, keep_prob)         return x

Xavier初始化保证卷积的初始权重不会太小也不会太大。以便从最初的轮开始，就允许信号通过网络更好地转化。

Xavier初始化提供了一个高斯分布，均值为0，方差为1除以一层中的神经元数量。这是因为这种初始化脱离了深度学习的预训练技术，以前用于设置可以在即使存在多个层也能进行反向传播初始的权重。你可以在这篇文章中了解更多关于Glorot和Bengio的初始化变量的内容：http:/ / andyljones. tumblr. com/ post/ 110998971763/ an- explanation- of- xavierinitialization。 
Batch normalization在这篇论文中被描述：
IOFFE, Sergey; SZEGEDY, Christian. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In: International Conference on Machine Learning. 2015. p. 448-456.

正如作者所指出的，batchnormalization算法做标准化需要处理变量偏移问题（http:/ / sifaka. cs. uiuc. edu/ jiang4/ domain_ adaptation/ survey/ node8.html），换言之，改变输入的分布会引起之前学习到的权重不再使用。事实上，作为第一个输入层初始学习到的分布，它们会被传输到下面所有层，并且由于输入突然改变而随之变化（例如，一开始你已经输入了狗的照片和更多的猫的照片，现在反过来了），除非你把学习率设定得很低，否则很可能会让人望而生畏。

Batch normalization解决了改变输入的分布引起的问题，因为它对于每个batch用均值和方差进标准化（用batch统计数据），正如论文IOFFE, Sergey; SZEGEDY,Christian. Batch normalization: Accelerating deep network training byreducing internal covariate shift. In: International Conference on Machine Learning. 2015. p. 448-456所阐述的（它可以在网上找到，网址是https:/ /arxiv. org/ abs/ 1502. 03167）。

g_reshaping 和 g_conv_transpose是generator中的两个函数。它们对输入重新定形，无论是平铺层还是卷积层。实际上，它们与卷积所做的工作相反，可以将卷积得到的特征恢复为原始特征：
g_reshaping and g_conv_transpose are two functions that are part of the generator.
    def g_reshaping(self, x, shape, alpha=0.2,                     keep_prob=0.5, train=True):
        """
        Generator layer architecture
        Reshaping layer, applying batch normalization,
        leaky rely activation and dropout
        """
        x = tf.reshape(x, shape)
        x = tf.layers.batch_normalization(x, training=train)
        x = self.leaky_ReLU_activation(x, alpha)
        x = self.dropout(x, keep_prob)         return x
    def g_conv_transpose(self, x, filters, kernel_size,                          strides, padding='same', alpha=0.2,                          keep_prob=0.5, train=True):
        """
        Generator layer architecture
        Transposing convolution to a new size,         applying batch normalization,         leaky rely activation and dropout
        """
        x = tf.layers.conv2d_transpose(x, filters, kernel_size,
                                       strides, padding)         x = tf.layers.batch_normalization(x, training=train)
        x = self.leaky_ReLU_activation(x, alpha)         x = self.dropout(x, keep_prob)         return x



判别器体系结构以图像为输入，通过各种卷积变换，直到结果展平，变成对数和概率（通过sigmoid函数）。实际上，一切都与有序卷积相同： 

 def discriminator(self, images, labels, reuse=False):      with tf.variable_scope('discriminator', reuse=reuse):          # Input layer is 28x28x3 --> concatenating input          x = tf.concat([images, labels], 3)
         # d_conv --> expected size is 14x14x32          x = self.d_conv(x, filters=32, kernel_size=5,                          strides=2, padding='same',                          alpha=0.2, keep_prob=0.5)
         # d_conv --> expected size is 7x7x64          x = self.d_conv(x, filters=64, kernel_size=5,                          strides=2, padding='same',                          alpha=0.2, keep_prob=0.5)
         # d_conv --> expected size is 7x7x128          x = self.d_conv(x, filters=128, kernel_size=5,                          strides=1, padding='same',                          alpha=0.2, keep_prob=0.5)
         # Flattening to a layer --> expected size is 4096          x = tf.reshape(x, (-1, 7 * 7 * 128))
         # Calculating logits and sigmoids          logits = tf.layers.dense(x, 1)          sigmoids = tf.sigmoid(logits)          return sigmoids, logits

至于生成器，它的结构和判别器相反。从输入向量z开始，首先创建dense层，之后进行一系列变换以重现判别器中卷积的逆过程，直到生成与输入形状相同的张量为止，并通过tanh函数做进一步的变换：

 def generator(self, z, out_channel_dim, is_train=True):
        with tf.variable_scope('generator',                                 reuse=(not is_train)):
            # First fully connected layer             x = tf.layers.dense(z, 7 * 7 * 512)
            # Reshape it to start the convolutional stack             x = self.g_reshaping(x, shape=(-1, 7, 7, 512),                                  alpha=0.2, keep_prob=0.5,                                  train=is_train)
            # g_conv_transpose --> 7x7x128 now             x = self.g_conv_transpose(x, filters=256,                                       kernel_size=5,                                       strides=2, padding='same',                                       alpha=0.2, keep_prob=0.5,                                       train=is_train)
            # g_conv_transpose --> 14x14x64 now             x = self.g_conv_transpose(x, filters=128,                                       kernel_size=5, strides=2,                                       padding='same', alpha=0.2,
                                      keep_prob=0.5,                                       train=is_train)
            # Calculating logits and Output layer --> 28x28x5 now
            logits = tf.layers.conv2d_transpose(x,
                                         filters=out_channel_dim,
                                         kernel_size=5,                                          strides=1,                                          padding='same')             output = tf.tanh(logits)             return output

这一结构与介绍CGANs的论文中画出的结构非常相似，论文中画出了如何通过大小为100的输入向量重构64*64**3的图像：

​	图4：DVGANs生成器结构

​	来源: arXiv, 1511.06434,2015

在定义结构之后，损失函数是接下来需要定义的重要元素。它采用两个输出，来自生成器的输出，将要在管道中被输入到判别器输出对数中；以及来自真实图像的输出，在管道中将要被输入到判别器中。对这二者而言，下一步需要计算损失。这里，平滑的参数很有用，因为它可以使真实图像转化为某些东西的概率不为1，允许一个更好的、更有可能性的结果被GANs学习（在完全惩罚的情况下，对于伪造图像，得到可以对抗真实图像的机会可能会变得更难）。

最终的判别器损失，是根据伪造图像和真实图像计算的损失的和。在真实图像上的损失是通过比较估计的logit与平滑的概率（我们的案例中是0.9）来计算的，这是为了防止过拟合，以及判别器可以通过存储图像来简单地学习并判断真实图像。生成器损失是由对伪图像的判别器估计的对数来计算的，其概率为1。 用这种方法，生成器努力产生伪造图像，它们能够被判别器判定为真（拥有较高的概率）。因此，在一个循环中，损失简单地从判别器对伪造图像的估计向生成器转化：
    def loss(self, input_real, input_z, labels, out_channel_dim):
        # Generating output
        g_output = self.generator(input_z, out_channel_dim)
        # Classifying real input
        d_output_real, d_logits_real = self.discriminator(input_real,
labels, reuse=False)
        # Classifying generated output
        d_output_fake, d_logits_fake = self.discriminator(g_output, labels,
reuse=True)
        # Calculating loss of real input classification
        real_input_labels = tf.ones_like(d_output_real) * (1 - self.smooth)

smoothed ones

```
  d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
```

labels=real_input_labels))
        # Calculating loss of generated output classification         fake_input_labels = tf.zeros_like(d_output_fake) # just zeros
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
labels=fake_input_labels))
        # Summing the real input and generated output classification losses         d_loss = d_loss_real + d_loss_fake # Total loss for discriminator         # Calculating loss for generator: all generated images should have
been
        # classified as true by the discriminator
        target_fake_input_labels = tf.ones_like(d_output_fake) # all ones
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=target_fake_input_labels))         return d_loss, g_loss

既然GANs上的工作是可视化的，这里有一些对当前生成器产生的样例和特定的图像集合可视化的函数：

​    def rescale_images(self, image_array):
        """
        Scaling images in the range 0-255
        """
        new_array = image_array.copy().astype(float)
        min_value = new_array.min()
        range_value = new_array.max() - min_value
        new_array = ((new_array - min_value) / range_value) * 255         return new_array.astype(np.uint8)
    def images_grid(self, images, n_cols):
        """
        Arranging images in a grid suitable for plotting
        """
        # Getting sizes of images and defining the grid shape         n_images, height, width, depth = images.shape
        n_rows = n_images // n_cols
        projected_images = n_rows * n_cols         # Scaling images to range 0-255         images = self.rescale_images(images)         # Fixing if projected images are less         if projected_images < n_images:
            images = images[:projected_images]         # Placing images in a square arrangement         square_grid = images.reshape(n_rows, n_cols,                                      height, width, depth)         square_grid = square_grid.swapaxes(1, 2)         # Returning a image of the grid         if depth >= 3:             return square_grid.reshape(height * n_rows,                                        width * n_cols, depth)         else:             return square_grid.reshape(height * n_rows,                                        width * n_cols)
    def plotting_images_grid(self, n_images, samples):
        """
        Representing the images in a grid
        """
        n_cols = math.floor(math.sqrt(n_images))         images_grid = self.images_grid(samples, n_cols)         plt.imshow(images_grid, cmap=self.cmap)         plt.show()
    def show_generator_output(self, sess, n_images, input_z,                               labels, out_channel_dim,                               image_mode):
        """
        Representing a sample of the         actual generator capabilities
        """
        # Generating z input for examples         z_dim = input_z.get_shape().as_list()[-1]         example_z = np.random.uniform(-1, 1, size=[n_images, \                                        z_dim - labels.shape[1]])         example_z = np.concatenate((example_z, labels), axis=1)
        # Running the generator         sample = sess.run(
            self.generator(input_z, out_channel_dim, False),
            feed_dict={input_z: example_z})
        # Plotting the sample
        self.plotting_images_grid(n_images, sample)
    def show_original_images(self, n_images):
        """
        Representing a sample of original images
        """
        # Sampling from available images
        index = np.random.randint(self.dataset.shape[0],                                   size=(n_images))         sample = self.dataset.data[index]
        # Plotting the sample         self.plotting_images_grid(n_images, sample)

使用Adam优化器，从最初的判别器开始判别器和生成器的损失都被减小（确定生成器针对真实图像的生成有多好），然后基于生成器产生的伪造图像对判别器器的影响的评估，将反馈传播到生成器： 

​    def optimization(self):
        """
        GAN optimization procedure
        """
        # Initialize the input and parameters placeholders
        cases, image_width, image_height,\         out_channel_dim = self.dataset.shape         input_real, input_z, labels, learn_rate = \                         self.instantiate_inputs(image_width,                                                image_height,                                             out_channel_dim,                                                  self.z_dim,                                   len(self.dataset.classes))
        # Define the network and compute the loss         d_loss, g_loss = self.loss(input_real, input_z,                                     labels, out_channel_dim)
        # Enumerate the trainable_variables, split into G and D parts
        d_vars = [v for v in tf.trainable_variables() \                     if v.name.startswith('discriminator')]         g_vars = [v for v in tf.trainable_variables() \                     if v.name.startswith('generator')]         self.g_vars = g_vars
        # Optimize firt the discriminator, then the generatvor
        with tf.control_dependencies(\                      tf.get_collection(tf.GraphKeys.UPDATE_OPS)):             d_train_opt = tf.train.AdamOptimizer(
                                             self.learning_rate,                    self.beta1).minimize(d_loss, var_list=d_vars)
            g_train_opt = tf.train.AdamOptimizer(
                                             self.learning_rate,                    self.beta1).minimize(g_loss, var_list=g_vars)
        return input_real, input_z, labels, learn_rate,                d_loss, g_loss, d_train_opt, g_train_opt

最后，我们完成了训练阶段。在训练中，有两部分需要注意：

At last, we have the complete training phase. In the training, there are two parts that require attention:
		如何在两步中完成优化、

   1. 运行判别器优化

   2. 运行生成器优化

      ​	如何对随机输入和真实图像进行预处理，对它们和标签进行混合，创建包含与标签相关的独热编码的类信息更多图像层

用这种方法，可以使输入和输出数据中，类被包含在图像里，调节生成器以便考虑到这些信息，同时，如果无法生成与正确标签相对应的真实图像，对生成器进行惩罚。比方说我们的生成器产生够的图像，但是却给了它猫的标签。在这种情况下，生成器会受到判别器的惩罚，因为判别器会注意到生成器产生的毛和真实的猫不同，因为它们具有不同的标签：

def train(self, save_every_n=1000):     losses = []     step = 0
  epoch_count = self.epochs     batch_size = self.batch_size
  z_dim = self.z_dim
  learning_rate = self.learning_rate     get_batches = self.dataset.get_batches     classes = len(self.dataset.classes)     data_image_mode = self.dataset.image_mode
  cases, image_width, image_height,\     out_channel_dim = self.dataset.shape     input_real, input_z, labels, learn_rate, d_loss,\     g_loss, d_train_opt, g_train_opt = self.optimization()

Allowing saving the trained GAN     saver = tf.train.Saver(var_list=self.g_vars)

Preparing mask for plotting progression

  rows, cols = min(5, classes), 5     target = np.array([self.dataset.one_hot[i] \              for j in range(cols) for i in range(rows)])     with tf.Session() as sess:        sess.run(tf.global_variables_initializer())        for epoch_i in range(epoch_count):            for batch_images, batch_labels \                      in get_batches(batch_size):
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
              # Sampling random noise for G                 batch_images = batch_images * 2                 # Running optimizers
              _ = sess.run(d_train_opt, feed_dict={input_real:
batch_images, input_z: batch_z,                                                          labels: batch_labels, learn_rate: learning_rate})
              _ = sess.run(g_train_opt, feed_dict={input_z: batch_z,
input_real: batch_images,                                                          labels: batch_labels, learn_rate: learning_rate})
              # Cyclic reporting on fitting and generator output                 if step % (save_every_n//10) == 0:                     train_loss_d = sess.run(d_loss,
                                              {input_z: batch_z,
input_real: batch_images, labels: batch_labels})                     train_loss_g = g_loss.eval({input_z: batch_z, labels: batch_labels})
                  print("Epoch %i/%i step %i..." % (epoch_i + 1,
epoch_count, step),
                            "Discriminator Loss: %0.3f..." %
train_loss_d,                               "Generator Loss: %0.3f" % train_loss_g)                 if step % save_every_n == 0:                     rows = min(5, classes)
                  cols = 5
                  target = np.array([self.dataset.one_hot[i] for j in
range(cols) for i in range(rows)])
                  self.show_generator_output(sess, rows * cols, input_z,
target, out_channel_dim, data_image_mode)
                  saver.save(sess, './'+self.generator_name+'/generator.ckpt')
          # At the end of each epoch, get the losses and print them out             try:                 train_loss_d = sess.run(d_loss, {input_z: batch_z,
input_real: batch_images, labels: batch_labels})                 train_loss_g = g_loss.eval({input_z: batch_z, labels: batch_labels})
              print("Epoch %i/%i step %i..." % (epoch_i + 1, epoch_count,
step),
                       "Discriminator Loss: %0.3f..." % train_loss_d,
                        "Generator Loss: %0.3f" % train_loss_g)             except:                 train_loss_d, train_loss_g = -1, -1
          # Saving losses to be reported after training             losses.append([train_loss_d, train_loss_g])
      # Final generator output
      self.show_generator_output(sess, rows * cols, input_z, target,
out_channel_dim, data_image_mode)
      saver.save(sess, './' + self.generator_name + '/generator.ckpt')     return np.array(losses)

在训练过程中，网络不断地被保存到磁盘。当需要生成新图像时，你不需要重新训练，只需要加载网络并指定你希望GAN产生的图像的标签：

def generate_new(self, target_class=-1, rows=5, cols=5, plot=True):
      """
      Generating a new sample
      """
      # Fixing minimum rows and cols values         rows, cols = max(1, rows), max(1, cols)         n_images = rows * cols
      # Checking if we already have a TensorFlow graph         if not self.trained:
          # Operate a complete restore of the TensorFlow graph
          tf.reset_default_graph()             self._session = tf.Session()             self._classes = len(self.dataset.classes)
          self._input_z = tf.placeholder(tf.float32, (None, self.z_dim +
self._classes), name='input_z')
          out_channel_dim = self.dataset.shape[3]
          # Restoring the generator graph             self._generator = self.generator(self._input_z,
out_channel_dim)
          g_vars = [v for v in tf.trainable_variables() if
v.name.startswith('generator')]
          saver = tf.train.Saver(var_list=g_vars)             print('Restoring generator graph')             saver.restore(self._session, tf.train.latest_checkpoint(self.generator_name))
          # Setting trained flag as True             self.trained = True
      # Continuing the session         sess = self._session
      # Building an array of examples examples         target = np.zeros((n_images, self._classes))         for j in range(cols):             for i in range(rows):                 if target_class == -1:                     target[j * cols + i, j] = 1.0                 else:                     target[j * cols + i] = self.dataset.one_hot[target_class].tolist()
      # Generating the random input
      z_dim = self._input_z.get_shape().as_list()[-1]         example_z = np.random.uniform(-1, 1,                     size=[n_images, z_dim - target.shape[1]])         example_z = np.concatenate((example_z, target), axis=1)
      # Generating the images         sample = sess.run(             self._generator,
          feed_dict={self._input_z: example_z})         # Plotting         if plot:             if rows * cols==1:                 if sample.shape[3] <= 1:                     images_grid = sample[0,:,:,0]                 else:
                  images_grid = sample[0]             else:                 images_grid = self.images_grid(sample, cols)             plt.imshow(images_grid, cmap=self.cmap)
          plt.show()
      # Returning the sample for later usage
      # (and not closing the session)
      return sample

这一类由fit方法完成，可以接受学习率参数和beta1（Adam优化器的参数，基于初始的平均值调节学习率参数），并在训练完成后绘制来自判别器和生成器的损失结果：


  def fit(self, learning_rate=0.0002, beta1=0.35):
      """
      Fit procedure, starting training and result storage
      """
      # Setting training parameters         self.learning_rate = learning_rate
      self.beta1 = beta1         # Training generator and discriminator         with tf.Graph().as_default():             train_loss = self.train()         # Plotting training fitting
      plt.plot(train_loss[:, 0], label='Discriminator')         plt.plot(train_loss[:, 1], label='Generator')
      plt.title("Training fitting")         plt.legend()

##将CGAN应用于一些实例 

既然已经完成了CGAN的类，下面我们通过一些例子为你提供如何使用这一项目的新鲜的想法。首先，我们需要为下载必要的数据和训练我们的GAN做好准备。我们导入程序库开始：

import numpy as np 

import urllib.request

 import tarfile 

import os

 import zipfile

 import gzip 

import os from glob

 import glob from tqdm 

import tqdm

然后，我们载入数据集和之前准备好的CGAN：

from cGAN import Dataset, CGAN

类TqdmUpTo是一个tqdm包装，可以显示下载进度。这个类直接来自于项目的主页https:/ /github. com/ tqdm/ tqdm:

class TqdmUpTo(tqdm):     """     Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
  Inspired by https://github.com/pypa/twine/pull/242     https://github.com/pypa/twine/commit/42e55e06     """
  def update_to(self, b=1, bsize=1, tsize=None):         """         Total size (in tqdm units).
      If [default: None] remains unchanged.
      """         if tsize is not None:             self.total = tsize
      # will also set self.n = b * bsize         self.update(b * bsize - self.n)

最后，如果我们使用jupyter notebook（强烈建议尝试），你必须启用图像的内联绘制

%matplotlib inline

 我们现在为开始第一个例子做好了准备。 

##MNIST

The MNIST database of handwritten digits was provided by Yann LeCun when he was at
ourant Institute, NYU, and by Corinna Cortes (Google Labs) and Christopher J.C. Burges (Microsoft Research). It is considered the standard for learning from real-world image data with minimal effort in preprocessing and formatting. The database consists of handwritten digits, offering a training set of 60,000 examples and a test set of 10,000. It is actually a subset of a larger set available from NIST. All the digits have been size-normalized and centered in a fixed-size image: http:// yann	.lecun	.com /exdb	/mnist	/

Figure 5: A sample of the original MNIST helps to understand the quality of the images to be reproduced by the CGAN.
As a first step, we upload the dataset from the Internet and store it locally:
labels_filename = 'train-labels-idx1-ubyte.gz' images_filename = 'train-images-idx3-ubyte.gz'
url = "http://yann.lecun.com/exdb/mnist/" with TqdmUpTo() as t: # all optional kwargs     urllib.request.urlretrieve(url+images_filename,
                               'MNIST_'+images_filename,                                reporthook=t.update_to, data=None)
with TqdmUpTo() as t: # all optional kwargs     urllib.request.urlretrieve(url+labels_filename,
                               'MNIST_'+labels_filename,                                reporthook=t.update_to, data=None)
In order to learn this set of handwritten numbers, we apply a batch of 32 images, a learning rate of 0.0002, a beta1 of 0.35, a z_dim of 96, and 15 epochs for training:
labels_path = './MNIST_train-labels-idx1-ubyte.gz' images_path = './MNIST_train-images-idx3-ubyte.gz'
with gzip.open(labels_path, 'rb') as lbpath:         labels = np.frombuffer(lbpath.read(),                                dtype=np.uint8, offset=8) with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
        offset=16).reshape(len(labels), 28, 28, 1)
batch_size = 32 z_dim = 96 epochs = 16
dataset = Dataset(images, labels, channels=1)
gan = CGAN(dataset, epochs, batch_size, z_dim, generator_name='mnist')
gan.show_original_images(25) gan.fit(learning_rate = 0.0002, beta1 = 0.35)
The following image represents a sample of the numbers generated by the GAN at the second epoch and at the last one:

After 16 epochs, the numbers appear to be well shaped and ready to be used. We then extract a sample of all the classes arranged by row.
Evaluating the performances of a GAN is still most often the matter of visual inspecting some of its results by a human judge, trying to figure out if the image could be a fake (like a discriminator) from its overall aspect or by precisely revealing details. GANs lack an objective function to help to evaluate and compare them, though there are some computational techniques that could be used as a metric such as the log-likelihood, as described by THEIS, Lucas; OORD, Aäron van den; BETHGE, Matthias. A note on the evaluation of generative models. arXiv preprint arXiv:1511.01844, 2015: https:/ /	arxiv	. org /	abs	/	1511	.	0184	4.
We will keep our evaluation simple and empirical and thus we will use a sample of images generated by the trained GAN in order to evaluate the performances of the network and we also try to inspect the training loss for both the generator and the discriminator in order to spot any particular trend:

Figure 7: A sample of the final results after training on MNIST reveals it is an accessible task for a GAN network
Observing the training fit chart, represented in the figure the following, we notice how the generator reached the lowest error when the training was complete. The discriminator, after a previous peak, is struggling to get back to its previous performance values, pointing out a possible generator's breakthrough. We can expect that even more training epochs could improve the performance of this GAN network, but as you progress in the quality the output, it may take exponentially more time. In general, a good indicator of convergence of a GAN is having a downward trend of both the discriminator and generator, which is something that could be inferred by fitting a linear regression line to both loss vectors:

Training an amazing GAN network may take a very long time and a lot of computational resources. By reading this recent article appeared in the
New York Times, https:/ /	www	.	nytimes	.	com	/	interactive	/	2018	/ 01 / 02 / technology/ ai -	generated	-	photos	.	htm	l, you can find a chart from NVIDIA showing the progress in time for the training of a progressive GAN learning from photos of celebrities. Whereas it can take a few days to get a decent result, for an astonishing one you need at least a fortnight. In the same way, even with our examples, the more training epochs you put in, the better the results.
Zalando MNIST
Fashion MNIST is a dataset of Zalando's article images, composed of a training set of 60,000 examples and a test set of 10,000 examples. As with MNIST, each example is a 28x28 grayscale image, associated with a label from 10 classes. It was intended by authors from
Zalando Research (https:// github	.com /zalandoresearch	/fashion	-mnist	/graphs	/ contributors) as a replacement for the original MNIST dataset in order to better benchmark machine learning algorithms since it is more challenging to learn and much more representative of deep learning in real-world tasks (https:// twitter	.com	/fchollet	/ status/85259498752704512	0). https://github.com/zalandoresearch/fashion-mnist

We download the images and their labels separately:
url = "http://fashion-mnist.s3-website.eu-central-\        1.amazonaws.com/train-images-idx3-ubyte.gz"
filename = "train-images-idx3-ubyte.gz" with TqdmUpTo() as t: # all optional kwargs     urllib.request.urlretrieve(url, filename,
                               reporthook=t.update_to, data=None)
url = "http://fashion-mnist.s3-website.eu-central-\        1.amazonaws.com/train-labels-idx1-ubyte.gz"
filename = "train-labels-idx1-ubyte.gz" _ = urllib.request.urlretrieve(url, filename)
In order to learn this set of images, we apply a batch of 32 images, a learning rate of 0.0002, a beta1 of 0.35, a z_dim of 96, and 10 epochs for training:
labels_path = './train-labels-idx1-ubyte.gz' images_path = './train-images-idx3-ubyte.gz' label_names = ['t_shirt_top', 'trouser', 'pullover',                'dress', 'coat', 'sandal', 'shirt',
               'sneaker', 'bag', 'ankle_boots']
with gzip.open(labels_path, 'rb') as lbpath:         labels = np.frombuffer(lbpath.read(),                                dtype=np.uint8,                                offset=8) with gzip.open(images_path, 'rb') as imgpath:         images = np.frombuffer(imgpath.read(), dtype=np.uint8,
        offset=16).reshape(len(labels), 28, 28, 1)
batch_size = 32 z_dim = 96 epochs = 64
dataset = Dataset(images, labels, channels=1)
gan = CGAN(dataset, epochs, batch_size, z_dim, generator_name='zalando')
gan.show_original_images(25) gan.fit(learning_rate = 0.0002, beta1 = 0.35)
The training takes a long time to go through all the epochs, but the quality appears to soon stabilize, though some problems take more epochs to disappear (for instance holes in shirts):

Figure 10: The evolution of the CGAN's training through epochs
Here is the result after 64 epochs:

Figure 11: An overview of the results achieved after 64 epochs on Zalando dataset
The result is fully satisfactory, especially for clothes and men's shoes. Women's shoes, however, seem more difficult to be learned because smaller and more detailed than the other images.
EMNIST
The EMNIST dataset is a set of handwritten character digits derived from the NIST Special Database and converted to a 28 x 28 pixel image format and dataset structure that directly matches the MNIST dataset. We will be using EMNIST Balanced, a set of characters with an equal number of samples per class, which consists of 131,600 characters spread over 47 balanced classes. You can find all the references to the dataset in:
Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http:/ /	arxiv	. org/ abs /	1702	.	0537	3.
You can also explore complete information about EMNIST by browsing the official page of the dataset: https:// www .nist	.gov /itl /iad	/image	-group	/emnist	-datase	t. Here is an extraction of the kind of characters that can be found in the EMNIST Balanced:

Figure 11: A sample of the original EMNIST dataset
url = "http://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
filename = "gzip.zip"
with TqdmUpTo() as t: # all optional kwargs     urllib.request.urlretrieve(url, filename,                                reporthook=t.update_to,                                data=None)
After downloading from the NIST website, we unzip the downloaded package:
zip_ref = zipfile.ZipFile(filename, 'r')
zip_ref.extractall('.') zip_ref.close()
We remove the unused ZIP file after checking that the unzipping was successful:
if os.path.isfile(filename):     os.remove(filename)
In order to learn this set of handwritten numbers, we apply a batch of 32 images, a learning rate of 0.0002, a beta1 of 0.35, a z_dim of 96, and 10 epochs for training:
labels_path = './gzip/emnist-balanced-train-labels-idx1-ubyte.gz' images_path = './gzip/emnist-balanced-train-images-idx3-ubyte.gz' label_names = []
with gzip.open(labels_path, 'rb') as lbpath:         labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
         offset=8) with gzip.open(images_path, 'rb') as imgpath:         images = np.frombuffer(imgpath.read(), dtype=np.uint8,                   offset=16).reshape(len(labels), 28, 28, 1)
batch_size = 32 z_dim = 96 epochs = 32
dataset = Dataset(images, labels, channels=1) gan = CGAN(dataset, epochs, batch_size, z_dim,            generator_name='emnist')
gan.show_original_images(25)
gan.fit(learning_rate = 0.0002, beta1 = 0.35)
Here is a sample of some handwritten letters when completing the training after 32 epochs:

As for MNIST, a GAN can learn in a reasonable time to replicate handwritten letters in an accurate and credible way.
Reusing the trained CGANs
After training a CGAN, you may find useful to use the produced images in other applications. The method generate_new can be used to extract single images as well as a set of images (in order to check the quality of results for a specific image class). It operates on a previously trained CGan class, so all you have to do is just to pickle it in order first to save it, then to restore it again when needed.
When the training is complete, you can save your CGan class using pickle, as shown by these commands:
import pickle pickle.dump(gan, open('mnist.pkl', 'wb'))
In this case, we have saved the CGAN trained on the MNIST dataset.
After you have restarted the Python session and memory is clean of any variable, you can just import again all the classes and restore the pickled CGan:
from CGan import Dataset, CGan
import pickle gan = pickle.load(open('mnist.pkl', 'rb'))
When done, you set the target class you would like to be generated by the CGan (in the example we ask for the number 8 to be printed) and you can ask for a single example, a grid 5 x 5 of examples or a larger 10 x 10 grid:
nclass = 8
_ = gan.generate_new(target_class=nclass,                      rows=1, cols=1, plot=True)
_ = gan.generate_new(target_class=nclass,                      rows=5, cols=5, plot=True) images = gan.generate_new(target_class=nclass,                      rows=10, cols=10, plot=True) print(images.shape)
If you just want to obtain an overview of all the classes, just set the parameter target_class to -1.
After having set out target class to be represented, the generate_new is called three times and the last one the returned values are stored into the images variable, which is sized (100, 28, 28, 1) and contains a Numpy array of the produced images that can be reused for our purposes. Each time you call the method, a grid of results is plotted as shown in the following figure:

Figure 13: The plotted grid is a composition of the produced images, that is an image itself. From left to right, the plot of a request for a 1 x 1, 5 x 5, 10 x 10 grid of results. The real images are returned by the method and can be reused.
If you don't need generate_new to plot the results, you simply set the plot parameter to False: images = gan.generate_new(target_class=nclass, rows=10, cols=10, plot=False).
Resorting to Amazon Web Service
As previously noticed, it is warmly suggested you use a GPU in order to train the examples proposed in this chapter. Managing to obtain results in a reasonable time using just a CPU is indeed impossible, and also using a GPU may turn into quite long hours waiting for the computer to complete the training. A solution, requiring the payment of a fee, could be to resort to Amazon Elastic Compute Cloud, also known as Amazon EC2 (https:// aws	. amazon.com /it /ec2 /) , part of the Amazon Web Services (AWS). On EC2 you can launch virtual servers that you can control from your computer using the Internet connection. You can require servers with powerful GPUs on EC2 and make your life with TensorFlow projects much easier.
Amazon EC2 is not the only cloud service around. We have suggested you this service because it is the one we used in order to test the code in this book. Actually, there are alternatives, such as Google Cloud Compute (cloud.google.com), Microsoft Azure (azure.microsoft.com) and many others.
Running the chapter’s code on EC2 requires having an account in AWS. If you don’t have one, the first step is to register at aws.amazon.com, complete all the necessary forms and start with a free Basic Support Plan.
After you are registered on AWS, you just sign in and visit the EC2 page (https:// aws	. amazon.com /ec 2). There you will:

1. Select a region which is both cheap and near to you which allows the kind of GPU instances we need, from EU (Ireland), Asia Pacific (Tokyo), US East (N. Virginia) and US West (Oregon).
   .	Upgrade your EC2 Service Limit report at: https:// console	.aws	.amazon	.com	/ ec2/v2 /home	?#Limit	s. You will need to access a p3.2xlarge instance. Therefore if your actual limit is zero, that should be taken at least to one, using the Request Limit Increase form (this may take up to 24 hours, but before it's complete, you won’t be able to access this kind of instance).
   .	Get some AWS credits (providing your credit card, for instance).
   After setting your region and having enough credit and request limit increase, you can start a p3.2xlarge server (a GPU compute server for deep learning applications) set up with an OS already containing all the software you need (thanks to an AMI, an image prepared by Amazon):
   .	Get to the EC2 Management Console, and click on the Launch Instance button.
   .	Click on AWS Marketplace, and search for Deep Learning AMI with Source Code v2.0 (ami-bcce6ac4) AMI. This AMI has everything pre-installed: CUDA, cuDNN (https:// developer	.nvidia	.com	/cudn	n), Tensorflow.
   .	Select the GPU compute p3.2xlarge instance. This instance has a powerful NVIDIA Tesla V100 GPU.
   .	Configure a security group (which you may call Jupyter) by adding Custom TCP Rule, with TCP protocol, on port 8888, accessible from anywhere. This will allow you to run a Jupyter server on the machine and see the interface from any computer connected to the Internet.
   .	Create an Authentication Key Pair. You can call it deeplearning_jupyter.pem for instance. Save it on your computer in a directory you can easily access.
   .	Launch the instance. Remember that you will be paying since this moment unless you stop it from the AWS menu—you still will incur in some costs, but minor ones and you will have the instance available for you, with all your data—or simply terminate it and don’t pay any more for it.
   After everything is launched, you can access the server from your computer using ssh.
   Take notice of the IP of the machine. Let’s say it is xx.xx.xxx.xxx, as an example.
   From a shell pointing to the directory where you .pem file is, type:
   ssh -i deeplearning_jupyter.pem ubuntu@ xx.xx.xxx.xxx
   When you have accessed the server machine, configure its Jupyter server by typing these commands: 
   jupyter notebook --generate-config
   sed -ie "s/#c.NotebookApp.ip = 'localhost'/#c.NotebookApp.ip =
   '*'/g" ~/.jupyter/jupyter_notebook_config.py
    Operate on the server by copying the code (for instance by git cloning the code repository) and installing any library you may require. For instance, you could install these packages for this specific project:
   sudo pip3 install tqdm sudo pip3 install conda
    Launch the Jupyter server by running the command: jupyter notebook --ip=0.0.0.0 --no-browser
    At this point, the server will run and your ssh shell will prompt you the logs from Jupyter. Among the logs, take note of the token (it is something like a sequence of numbers and letters).
    Open your browser and write in the address bar:  http:// xx.xx.xxx.xxx:8888/
   When required type the token and you are ready to use the Jupiter notebook as you were on your local machine, but it is actually operating on the server. At this point, you will have a powerful server with GPU for running all your experiments with GANs.

   ##Acknowledgements

   In concluding this chapter, we would like to thank Udacity and Mat Leonard for their
   CGAN tutorial, licensed under MIT (https:// github	.com	/udacity	/deep	-learning	/ blob/master	/LICENS	E) which provided a good starting point and a benchmark for this
   project.

   ##Summary

   In this chapter, we have discussed at length the topic of Generative Adversarial Networks, how they work, and how they can be trained and used for different purposes. As a project, we have created a conditional GAN, one that can generate different types of images, based on your input and we learned how to process some example datasets and train them in order to have a pickable class capable of creating new images on demand.