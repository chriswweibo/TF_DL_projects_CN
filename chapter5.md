# TensorFlow Deep Learning Projects



## 第5章 利用LSTM预测股票价格

本章主要讲述如何进行实值的时间序列预测。具体来说，本章会利用纽约证券交易所的历史数据来预测上市大公司的股价。

本章主要讲述如下内容：

- 如何收集股票的历史价格信息
- 如何组织数据，以进行时间序列预测
- 如何利用回归来预测某一支股票的未来价格
- 长短期记忆神经网络（LSTM）101
- LSTM是如何提高预测性能的
- 模型性能如何在Tensorboard上进行可视化展示

本章的小节按照上述要点进行划分。此外，为了使本章更易读更易理解，每个技术都会先在较简单的信号曲线（余弦信号）进行模拟实验。因为余弦信号比股票价格更具有确定性，可帮助读者理解算法的性能。

> 注意：此项目只是一个实验，只适用于已有的简单数据，无法保证真实场景中能有相同的性能，所以，请不要在真实场景中使用。请记住，投资有风险，本章无法保证读者一定会盈利。

### 输入数据集 – 余弦曲线数据和股票价格

如上所述，本节会使用两组一维的信号数据作为时间序列进行实验。第一个数据是加入了均匀噪声的余弦波信号。

下面函数用于生成余弦波信号，函数的参数为需要生成的数据的个数、信号频率，以及噪声强度。为保证实验可复现，函数体中设置了随机数种子:

```python
  def fetch_cosine_values(seq_len, frequency=0.01, noise=0.1):
	  np.random.seed(101)
	  x = np.arange(0.0, seq_len, 1.0)
	  return np.cos(2 * np.pi * frequency * x) + np.random.uniform(low=-noise, high=noise, size=seq_len)
```

输出10个数据点，振幅为0.1的余弦波信号，并加入[-0.1,0.1)的随机噪声，运行： 

```python
  print(fetch_cosine_values(10, frequency=0.1))
```

输出为：

```python
	[ 1.00327973 0.82315051 0.21471184 -0.37471266 -0.7719616 -0.93322063 -0.84762375 -0.23029438 0.35332577 0.74700479]
```

时间序列的每个点都可代表当天股票价格，是单维特征，本节会用此作为股票价格。

第二个信号数据来源于真实的金融市场。金融数据很珍贵、不易获取，所以本节用Python库`quandl`来获取这些数据。`quandl`易用、便宜（每天可免费查询XX次），且非常适用于本章的任务（预测股票收盘价格）。如果读者从事的是自动交易，则需要更多的数据来支持，例如从此库的更高版本中获取数据，或利用一些其他的库或数据源。
Quandl 是个 API, Python库是API的包装器。读者可以在提示符中执行以下命令来查看返回结果：

```
$> curl "https://www.quandl.com/api/v3/datasets/WIKI/FB/data.csv" Date,Open,High,Low,Close,Volume,Ex-Dividend,Split Ratio,Adj. Open,Adj.
High,Adj. Low,Adj. Close,Adj. Volume
2017-08-18,166.84,168.67,166.21,167.41,14933261.0,0.0,1.0,166.84,168.67,166
.21,167.41,14933261.0 
2017-08-17,169.34,169.86,166.85,166.91,16791591.0,0.0,1.0,169.34,169.86,166
.85,166.91,16791591.0 
2017-08-16,171.25,171.38,169.24,170.0,15580549.0,0.0,1.0,171.25,171.38,169.
24,170.0,15580549.0
2017-08-15,171.49,171.5,170.01,171.0,8621787.0,0.0,1.0,171.49,171.5,170.01,
171.0,8621787.0
...
```

这是CSV格式，每一行包含日期、开盘价、最高价、最低价、收盘价、调整价格、以及一些成交量指标，并按照日期由近到远排序。 由于本节需要的是调整后的收盘价，故只需要获取`Adj. Close`这一列。

> 调整后的收盘价指的是经过股息、分割、合并等调整后的收盘价格。

读者需注意，许多在线的服务提供的是未经调整的价格或直接给出开盘价，所以其提供的数据可能会与Quandl提供的不一致。

有了这些数据，需要构建一个Python函数，用Python API提取调整后的价格。本章只需要调用`quandl.get`函数，完整文档可在`https://docs.quandl.com/v1.0/docs` 查看。注意，默认是正序排序，即，价格是按照时间从远到近排序的。

这里所需函数应该能够进行缓存调用，并可以指定初始时间戳和截止时间戳，以及股票代号，以获取所需数据。代码如下:

```python
  def date_obj_to_str(date_obj):
	  return date_obj.strftime('%Y-%m-%d')
  def save_pickle(something, path):
	  if not os.path.exists(os.path.dirname(path)):
		  os.makedirs(os.path.dirname(path))
	  with open(path, 'wb') as fh:
		  pickle.dump(something, fh, pickle.DEFAULT_PROTOCOL)
  def load_pickle(path):
	  with open(path, 'rb') as fh:
	  return pickle.load(fh)
  def fetch_stock_price(symbol,from_date,to_date, cache_path="./tmp/prices/"):
	  assert(from_date <= to_date)
	  filename = "{}_{}_{}.pk".format(symbol, str(from_date), str(to_date))
	  price_filepath = os.path.join(cache_path, filename)
	  try:
		  prices = load_pickle(price_filepath)
		  print("loaded from", price_filepath)
	  except IOError:
		  historic = quandl.get("WIKI/" + symbol,
		                      start_date = date_obj_to_str(from_date),
		                      end_date = date_obj_to_str(to_date))
		  prices = historic["Adj. Close"].tolist()
		  save_pickle(prices, price_filepath)
		  print("saved into", price_filepath)
	  return prices
```

`fetch_stock_price`函数返回一个一维数组，其内容为所需股票代码的股票价格，价格按照`from_date`到`to_date`来排序。缓存在函数内完成，即，如果找不到缓存内容，则会调用`quandl`  API。`date_obj_to_str`是个辅助函数，用于将`datetime.date`转化为API所需的正确的字符串类型。

下面代码用于打印1月份Google股票价格（股票代码为“GOOG”）：

```python
  import datetime
  print(fetch_stock_price("GOOG",
						datetime.date(2017, 1, 1),
						datetime.date(2017, 1, 31)))
```

输出为:
	[786.14, 786.9, 794.02, 806.15, 806.65, 804.79, 807.91, 806.36, 807.88, 804.61, 806.07, 802.175, 805.02, 819.31, 823.87, 835.67, 832.15, 823.31, 802.32, 796.79]

为了让所有的脚本都可调用上述的函数，建议读者把这些函数都写进一个Python文件中。例如，==本书的代码都写在`tool.py`文件中。==

### 数据集格式化

==经典的机器学习算法一般用预设好大小的数据做训练（即每条数据的特征数是固定的）。但是，用时间序列数据做训练和预测时无法预先定义时间长度，因为无法让模型既能作用在10天以来的数据上，又能作用在3年以来的数据上。==

很简单，本章在保持特征维度大小不变的情况下，改变观测的数目。每个观测值都代表了时间序列上的一个时间窗口， 向右滑动一格就可以创建一个新的观测值。代码如下：

```python
  def format_dataset(values, temporal_features):
	  feat_splits = [values[i:i + temporal_features] for i in range(len(values) - temporal_features)]
	  feats = np.vstack(feat_splits)
	  labels = np.array(values[temporal_features:])
	  return feats, labels
```

给定时间序列、特征大小，此函数会创建一个滑动窗口来遍历时间序列，并生成特征和标注（每个滑动窗口后面的一个值作为这个滑动窗口所生成特征的标注）。最后，将所有观测值和标注分别按行堆叠起来。输出是列数确定的观测值矩阵，以及标注向量。

建议读者将此函数写入`tools.py`文件中，以便后续使用。

以下代码绘制出了余弦波开始的两个摆动。本节将此代码保存在了一个名为`1_visualization_data.py的`Python脚本中:

```python
  import datetime
  import matplotlib.pyplot as plt
  import numpy as np import seaborn
  from tools import fetch_cosine_values, fetch_stock_price, format_dataset
  np.set_printoptions(precision=2)

  cos_values = fetch_cosine_values(20, frequency=0.1)
  seaborn.tsplot(cos_values)
  plt.xlabel("Days since start of the experiment")
  plt.ylabel("Value of the cosine function")
  plt.title("Cosine time series over time")
  plt.show()
```

代码很简单，引入几个包后，便绘制出周期为10（频率为0.01）、20个点的余弦时间序列:
<img src="E:\我的\翻译\116_1.jpg" style="zoom:55%" align="center" /> 



为了让时间序列数据可作为机器学习算法的输入，需要将其格式化。以下代码将时间序列数据处理成五列的观测矩阵:

```python
  features_size = 5
  minibatch_cos_X, minibatch_cos_y =format_dataset(cos_values,features_size)
  print("minibatch_cos_X.shape=", minibatch_cos_X.shape)
  print("minibatch_cos_y.shape=", minibatch_cos_y.shape)
```

输入是20个点的时间序列，输出是一个15x5的观测矩阵和长度为15的标签向量。当然，如果改变特征大小，观测矩阵的行数也会发生改变。

为了便于读者理解，在这里对上述操作进行可视化展示。例如，把观测矩阵的前5个观测序列及其对应的标签（红色标记）在图中画出：

```python
  samples_to_plot = 5
  f, axarr = plt.subplots(samples_to_plot, sharex=True)
  for i in range(samples_to_plot):
	  feats = minibatch_cos_X[i, :]
	  label = minibatch_cos_y[i]
	  print("Observation {}: X={} y={}".format(i, feats, label))
	  plt.subplot(samples_to_plot, 1, i+1)
	  axarr[i].plot(range(i, features_size + i), feats, '--o')
	  axarr[i].plot([features_size + i], label, 'rx')
	  axarr[i].set_ylim([-1.1, 1.1])
  plt.xlabel("Days since start of the experiment")
  axarr[2].set_ylabel("Value of the cosine function")
  axarr[0].set_title("Visualization of some observations: Features (blue) and Labels (red)")
  plt.show()
```

画出的图如下所示：
<img src="E:\我的\翻译\117_1.jpg" style="zoom:55%" align="center" /> 

从图中可以看出，时间序列数据已被转化为观测向量，每个向量的长度是5。

接下来为读者展示将股票价格输出为时间序列时的可视化结果。首先，本节挑选了一些美国著名的公司来查看它们过去一年的股票价格走势（读者也可自行选择喜欢的公司）。下图只展示了2015和2016年的股票价格走势。由于后续也会使用到这些数据，所以需要将其缓存下来：

```python
  symbols = ["MSFT", "KO", "AAL", "MMM", "AXP", "GE", "GM", "JPM", "UPS"]
  ax = plt.subplot(1,1,1)
  for sym in symbols:
	  prices = fetch_stock_price(
	  sym, datetime.date(2015, 1, 1), datetime.date(2016, 12, 31))
	  ax.plot(range(len(prices)), prices, label=sym)
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles, labels)
  plt.xlabel("Trading days since 2015-1-1")
  plt.ylabel("Stock price [$]")
  plt.title("Prices of some American stocks in trading days of 2015 and  2016")
  plt.show()
```

价格走势图如下所示:
<img src="E:\我的\翻译\118_1.jpg" style="zoom:55%" align="center" /> 

图中的每条线都是时间序列，本节按照处理余弦波信号的方式将其转化为观测矩阵（利用`format_dataset`函数）。

这样简单的处理后，数据就准备好了，接下来可以开始进行模型部分的实操了。

### 用回归模型来预测未来的股票价格

准备好了观测矩阵和实值标签，本节先将这个问题看作回归问题。回归问题其实很简单：给定一个数值类型向量，预测一个数值类型的值。把这个问题当做回归问题，理想情况下，需要使得算法认为特征间相互独立。如果不做强制，那么，从同一个时间序列中通过滑动窗口得到的特征将被认为相互依赖。本节先从特征相互独立这个简单的假设开始，下节会展示如何利用时间的相关性来提高性能。

为了评估模型，本节构建了一个函数，输入为观测矩阵、实际的标签，以及预测的标签，输出为评估指标：均方误差**mean square error**(*MSE*)和平均绝对误差**mean absolute error**(*MAE*)。函数同时将训练、测试、预测性能在同一张图中画出，以便读者能直观地观测到模型的性能。另外，本节设置了一个基准：即，简单地将最后一天的价格作为预测价格，与模型预测给出的结果作比较。

另外，本节仍然需要一个辅助函数来将矩阵转化为一维数组。由于后续还要在多个脚本中使用此辅助函数，所以将此函数写在`tool.py`中。

```python
  def matrix_to_array(m):
	  return np.asarray(m).reshape(-1)
```

接下来是评估函数部分。评估函数写进`evaluate_ts.py`文件中，以便其他脚本调用：

```python
  import numpy as np
  from matplotlib import pylab as plt 
  from tools import matrix_to_array
  def evaluate_ts(features, y_true, y_pred):
	  print("Evaluation of the predictions:")
	  print("MSE:", np.mean(np.square(y_true - y_pred)))
	  print("mae:", np.mean(np.abs(y_true - y_pred)))
	
	  print("Benchmark: if prediction == last feature")
	  print("MSE:", np.mean(np.square(features[:, -1] - y_true))
	  print("mae:", np.mean(np.abs(features[:, -1] - y_true)))
	
	  plt.plot(matrix_to_array(y_true), 'b')
	  plt.plot(matrix_to_array(y_pred), 'r--')
	  plt.xlabel("Days")
	  plt.ylabel("Predicted and true values")
	  plt.title("Predicted (Red) VS Real (Blue)")
	  plt.show()
	
	  error = np.abs(matrix_to_array(y_pred) - matrix_to_array(y_true))
	  plt.plot(error, 'r')
	  fit = np.polyfit(range(len(error)), error, deg=1)
	  plt.plot(fit[0] * range(len(error)) + fit[1], '--')
	  plt.xlabel("Days")
	  plt.ylabel("Prediction error L1 norm")
	  plt.title("Prediction error (absolute) and trendline")
	  plt.show()
```

接下来进入构建模型的部分。

如前所述，本节先在余弦信号波上进行预测，再在股票价格上进行预测。

建议读者把下面代码写进另一个文件中，例如，写进`2_regression_cosion.py`中（读者可在以此命名的脚本中找到这些代码）。

首先，引入一些包，并为numpy和tensorflow设置一下随机数种子。

```python
  import matplotlib.pyplot as plt
  import numpy as np import tensorflow as tf
  from evaluate_ts import evaluate_ts
  from tensorflow.contrib import rnn
  from tools import fetch_cosine_values, format_dataset

  tf.reset_default_graph()
  tf.set_random_seed(101)
```

接下来，构建余弦波信号，并将其转换成一个观测矩阵。本例将特征大小设置为20（这个数大约相当于一个月的工作日天数）。此时，此回归问题可描述为：给定余弦曲线上20个点，预测下一个点的值。
本节用分别用250条观测数据作为训练集和测试集，这个数字基本等于一年可获取的数据量（一年的工作日天数小于250天）。本例只会生成一个余弦波信号曲线，并将此曲线分成两部分：前半部分用于训练，后半部分用于测试。读者可以按照自己的意愿进行重新划分，并观测以下参数变化时，模型性能的变化：

```python
  feat_dimension = 20
  train_size = 250
  test_size = 250
```

1. 这个脚本会定义一些Tensorflow所需要的参数。具体为以下参数：学习率、优化器类型、`epoch`数（训练时使用全部训练样本训练的次数）。这些值并不是最优组合，读者可自行调整以提高模型性能:

```python
  learning_rate = 0.01
  optimizer = tf.train.AdamOptimizer
  n_epochs = 10
```

2. 接下来是为训练和测试来准备观测矩阵。读者需注意，在训练和测试中会使用`float32`（4个字节长度）以加速Tensorflow。

```python
  cos_values = fetch_cosine_values(train_size + test_size + feat_dimension)
  minibatch_cos_X, minibatch_cos_y = format_dataset(cos_values,
  feat_dimension)
  train_X = minibatch_cos_X[:train_size, :].astype(np.float32)
  train_y = minibatch_cos_y[:train_size].reshape((-1,1)).astype(np.float32)
  test_X = minibatch_cos_X[train_size:, :].astype(np.float32)
  test_y = minibatch_cos_y[train_size:].reshape((-1, 1)).astype(np.float32)
```

有了这些数据，就可以为观测矩阵和标签设置占位符。由于这只是一个通用的脚本，所以不设置样本集数据量，仅设置特征数：

```python
  X_tf = tf.placeholder("float", shape=(None, feat_dimension), name="X")
  y_tf = tf.placeholder("float", shape=(None, 1), name="y")
```

下面是本项目的代码：回归算法在Tensorflow中的实现。

   1.本节用最经典的方式来实现回归算法，即，观测矩阵与权值数组相乘后加上偏差。返回的结果是一个数组，包含数据集X中所有数据的预测结果:

```python
def regression_ANN(x, weights, biases):
	return tf.add(biases, tf.matmul(x, weights))
```

2. 接下来，本节定义回归器需要训练的参数，这些参数也是`tensorflow`的变量。权重是个向量，向量的模与特征大小相同，而偏差是个标量。

> 读者需注意，初始化权重时，本节使用截断正态分布，这样既有接近零的值，又不至于太极端（可以作为普通正态分布输出）；而偏置项则设置为零。

同样的，读者可以更改初始化方式，来调整模型性能：

```python
  weights = tf.Variable(tf.truncated_normal([feat_dimension, 1], mean=0.0, stddev=1.0), name="weights")
  biases = tf.Variable(tf.zeros([1, 1]), name="bias")
```

3. 最后，本节展示如何在`tensorflow`中计算预测结果（本例较简单，模型的输出就是预测结果）、误差（本例使用MSE），以及如何进行模型训练（利用之前设置的优化器和学习率来最小化MSE）：

```python
  y_pred = regression_ANN(X_tf, weights, biases)
  cost = tf.reduce_mean(tf.square(y_tf - y_pred))
  train_op = optimizer(learning_rate).minimize(cost)
```

   接下来进入`tensorflow`部分，来介绍如何训练模型。

4. 首先进行变量初始化。接下来，写一个循环，把训练集喂给`tensorflow`（用占位符）。每轮迭代后，输出训练集上的MSE:

```python
  with tf.Session() as sess:
	  sess.run(tf.global_variables_initializer())
	#在每个epoch中，模型会使用所有的训练数据进行训练
	  for i in range(n_epochs):
		  train_cost, _ = sess.run([cost, train_op], feed_dict={X_tf: train_X, y_tf: train_y})
		  print("Training iteration", i, "MSE", train_cost)
	
	# 训练完毕后，需要检查一下模型目前在测试集上的表现
	  test_cost, y_pr = sess.run([cost, y_pred], feed_dict={X_tf: test_X, y_tf: test_y})
	  print("Test dataset:", test_cost)
   
	#评估结果
	  evaluate_ts(test_X, test_y, y_pr)
	
	#画出预测结果
	  plt.plot(range(len(cos_values)), cos_values, 'b')
	  plt.plot(range(len(cos_values)-test_size, len(cos_values)), y_pr, 'r--')
	  plt.xlabel("Days")
	  plt.ylabel("Predicted and true values")
	  plt.title("Predicted (Red) VS Real (Blue)")
	  plt.show()
```

训练完毕后，读者可以看到测试集上MSE评估结果，另外，本节还为读者展示模型的性能。

直接用脚本中设置的默认值来训练模型，其效果不如非建模方式的效果。随着迭代次数的增加，即各个参数的调整，效果也在提升。例如，把设置学习率为0.1，训练的epoch设置为1000，脚本的输出会与以下结果类似：

```
  Training iteration 0 MSE 4.39424
  Training iteration 1 MSE 1.34261
  Training iteration 2 MSE 1.28591
  Training iteration 3 MSE 1.84253
  Training iteration 4 MSE 1.66169
  Training iteration 5 MSE 0.993168 
  ...
  ...
  Training iteration 998 MSE 0.00363447
  Training iteration 999 MSE 0.00363426
  Test dataset: 0.00454513 
  Evaluation of the predictions:
  MSE: 0.00454513 mae: 0.0568501
  Benchmark: if prediction == last feature
  MSE: 0.964302 
  mae: 0.793475
```

读者可以看到，模型在训练集和测试集上的表现很相近(因此，模型没有过拟合)，MSE和MAE这两个指标均优于非模型预测。

下图展示了模型在每个时间点上预测错误率情况。可以看出错误率在正负0.15之内，且没有随着时间的变化而形成升高或降低的趋势。这是因为，在本章的开头，为余弦波信号引入的噪声值在正负0.1之内均匀分布：
<img src="E:\我的\翻译\124_1.jpg" style="zoom:55%" align="center" />

最后一个图显示了模型预测的时间序列与真实的时间序列会重叠在一起。对一个简单的线性回归来说，这个结果不错。
<img src="E:\我的\翻译\125_1.jpg" style="zoom:55%" align="center" />

接下来，本章在股票价格上使用同样的模型。建议读者把接下来的代码保存到一个新的文件中，并命名为`3_regression_stock_price.py`。这里只需要改变导入的包名，其余的不用改动。

下例用的是微软的股票价格，其股票代码是“`MSFT`”。由于本章开头的函数，可以轻松获取微软2015年和2016年的股票价格，并把价格数据格式化为观测矩阵。下面代码仍包含float32数据类型转换和训练集/测试集划分。本例应用2015年的数据进行训练，来预测2016年的股票价格：

```python
  symbol = "MSFT" feat_dimension = 20
  train_size = 252
  test_size = 252 - feat_dimension

  # Settings for tensorflow
  learning_rate = 0.05
  optimizer = tf.train.AdamOptimizer
  n_epochs = 1000
	
  # Fetch the values, and prepare the train/test split
  stock_values = fetch_stock_price(symbol, datetime.date(2015, 1, 1), datetime.date(2016, 12, 31))
  minibatch_cos_X, minibatch_cos_y = format_dataset(stock_values, feat_dimension)
  train_X = minibatch_cos_X[:train_size, :].astype(np.float32)
  train_y = minibatch_cos_y[:train_size].reshape((-1, 1)).astype(np.float32)
  test_X = minibatch_cos_X[train_size:, :].astype(np.float32)
  test_y = minibatch_cos_y[train_size:].reshape((-1, 1)).astype(np.float32)
```

经过实验，将脚本中的参数做如下设置会使模型表现最好：

```python
  learning_rate = 0.5
  n_epochs = 20000
  optimizer = tf.train.AdamOptimizer
```

脚本会有以下类似的输出：

```
  The output of the script should look like this:
  Training iteration 0 MSE 15136.7
  Training iteration 1 MSE 106385.0
  Training iteration 2 MSE 14307.3
  Training iteration 3 MSE 15565.6 ...
  ...
  ...
  Training iteration 19998 MSE 0.577189
  Training iteration 19999 MSE 0.57704
  Test dataset: 0.539493 
  Evaluation of the predictions:
  MSE: 0.539493 mae: 0.518984
  Benchmark: if prediction == last feature
  MSE: 33.7714 
  mae: 4.6968
```

这个例子中，模型依然没有过拟合，简单的回归模型一定比不用模型的效果好。开始训练的时候，损失特别高，但是随着迭代次数的增加，损失会趋于0。同样的，因为本例预测的是美元，所以用mae分数作为评估。基于模型预测的第二天的股票价格，与真实价格平均差0.5美元；而不做任何学习的价格与真实价格相差9倍之多。

接下来，本章直观地评估模型的性能，下图是模型预测的值：
<img src="E:\我的\翻译\127_1.jpg" style="zoom:150%" align="center" />
下图是绝对误差，点线代表绝对误差的趋势：
<img src="E:\我的\翻译\128_1.jpg" style="zoom:150%" align="center" />

下图是真实数据和训练集上的预测数据：
<img src="E:\我的\翻译\129_1.jpg" style="zoom:150%" align="center" />
读者需注意，这只是简单的回归模型的性能，此模型没有利用特征之间的时间相关性。那么如何更好地利用时间相关性呢？

### 长短期记忆神经网络—LSTM 101

**长短期记忆神经网络**（**Long Short-Term Memory **, *LSTM*）模型是**递归神经网络**（**Recurrent Neural Networks **, *RNNs*）的特例。由于对其全面、严谨的描述超出了本书的范围，故不再赘述。本节将只讲述其本质。

> 读者若有兴趣，可参考Packt所著以下书籍：
> `https://www.packtpub.com/big-data-and-business-intelligence/neural-network-programming-tensorflow`
> 也可参考这个页面`https://www.packtpub.com/big-data-and-business-intelligence/neural-networks-r`

简单来说，RNN适用于序列数据： 以多维信号作为输入，并生成多维输出信号。下图是一个RNN的例子，这个RNN模型能够处理五个时间步长（每个时间步长是一个输入）。下图的下半部分是RNN的输入，上半部分是输出。每个输入或输出都包含一个N维的特征：
<img src="E:\我的\翻译\130_1.jpg" style="zoom:100%" align="center" />
在RNN的内部存在许多时间阶段；每个阶段与它本身的输入和输出相连，也与上一阶段的输出相连。由于这种设置，每个当前阶段的输出不再仅是当前输入的函数，还依赖于上一阶段的输出（上一阶段的输出依赖于上一阶段的输入和上上阶段的输出，以此类推）。这种设置保证了每个输入可以影响到接下来的所有输出，换句话说，每个输出都是前面所有输入及当前输入的函数。

> 读者需注意，并不是所有的输出都会被使用。例如，在一个情感分析任务中，给定一个句子（时间序列输入信号），判定其情感倾向（积极/消极），只有最后一个输出被认为是最终输出，其他的输出不会被作为输出使用。谨记，因为只有最后的一个输出包含了整个句子所有的情感信息，所以只会使用最后一个输出。

LSTM模型是RNNs的演化：RNNs中文本过长时，训练阶段可能会有非常小或巨大的梯度在整个网络中反向传播，从而导致权重为零或无穷大：这种情况经常表述为梯度消失/梯度爆炸。为了解决这一问题，LSTMs在每个阶段都有两个输出：一个是模型的真正输出，而另一个是内部状态，被称作记忆。
每个输出都会再次作为输出进入接下来的阶段中，降低梯度消失或梯度爆炸的可能。当然，这种做法会有额外的开销：复杂度（需要训练的权重）和模型占用的内存空间更大，这就是为什么本书强烈建议用GPU设备来训练RNN模型，因为可以大大加速模型！
与回归模型不同，RNNs需要用三维信号作为输入。Tensorflow 规定数据需要按以下格式输入：

- 样本
- 时间步长
- 特征

在前面的情感分析例子中，训练张量是三维的，x轴代表所有输入的句子，y轴代表构成句子的单词，z轴代表词典中的词。例如：要对包含1M句子的英语语料库（大约20,000个不同的词汇）进行句子情感分析任务，每个句子最长包含50个词，张量维度为1M x 50 x 20K。

### 利用LSTM进行股票价格预测

LSTM可以方便我们探测信号中所包含的时间冗余信息。上一节向读者介绍了观测矩阵需要格式化成3维的张量，三个轴分别是：

- 第一个轴包含数据样本
- 第二个周包含时间序列
- 第三个周包含输入的特征

由于本章处理的是一维的信号数据，LSTM的输入张量则格式化为（None, `time_dimension`, 1）,其中`time_dimention`是时间窗口的长度。以下是代码，依然先从余弦信号开始。建议读者将文件命名为`4_rnn_cosine.py`。

1. 首先，引入包：

```python
  import matplotlib.pyplot as plt
  import numpy as np
  import tensorflow as tf
  from evaluate_ts import evaluate_ts
  from tensorflow.contrib import rnn
  from tools import fetch_cosine_values, format_dataset
  tf.reset_default_graph()
  tf.set_random_seed(101)
```

1. 接下来，设置窗口大小来给信号分块。此操作与创建观测矩阵类似。

```python
  time_dimension = 20
  train_size = 250
  test_size = 250
```

1. 然后，对Tensorflow进行一些设置。在这一步，本节先用默认的值来进行实验：

```python
  learning_rate = 0.01
  optimizer = tf.train.AdagradOptimizer
  n_epochs = 100
  n_embeddings = 64
```

1. 接下来，生成有噪声的余弦信号，并把这些数据重塑为3D张量格式 （None， `time_dimension`， 1）。代码如下：

```python
  cos_values = fetch_cosine_values(train_size + test_size + time_dimension)
  minibatch_cos_X, minibatch_cos_y = format_dataset(cos_values, time_dimension)
  train_X = minibatch_cos_X[:train_size, :].astype(np.float32)
  train_y = minibatch_cos_y[:train_size].reshape((-1, 1)).astype(np.float32)
  test_X = minibatch_cos_X[train_size:, :].astype(np.float32) test_y = minibatch_cos_y[train_size:].reshape((-1, 1)).astype(np.float32)
  train_X_ts = train_X[:, :, np.newaxis]
  test_X_ts = test_X[:, :, np.newaxis]
```

1. 为Tensorflow定义占位符：

```python
  X_tf = tf.placeholder("float", shape=(None, time_dimension, 1), name="X")
  y_tf = tf.placeholder("float", shape=(None, 1), name="y")
```

1. 接下来定义模型。本节会为用拥有不同数目embeddings的LSTM来实验。如前面章节所述，本节只使用通过线性回归后（全连接层）的最后一个输出以作为预测结果：

```python
  def RNN(x, weights, biases):
	  x_ = tf.unstack(x, time_dimension, 1)
	  lstm_cell = rnn.BasicLSTMCell(n_embeddings)
	  outputs, _ = rnn.static_rnn(lstm_cell, x_, dtype=tf.float32)
	  return tf.add(biases, tf.matmul(outputs[-1], weights))
```

1. 接下来，设置`可训练的`变量（`weights`），并设置损失函数和训练操作：

```python
  weights = tf.Variable(tf.truncated_normal([n_embeddings, 1], mean=0.0,
  stddev=1.0), name="weights")
  biases = tf.Variable(tf.zeros([1]), name="bias")
  y_pred = RNN(X_tf, weights, biases)
  cost = tf.reduce_mean(tf.square(y_tf - y_pred))
  train_op = optimizer(learning_rate).minimize(cost)
  
  # Exactly as before, this is the main loop. 
  with tf.Session() as sess: 
	  sess.run(tf.global_variables_initializer())
       
	  # For each epoch, the whole training set is feeded into the tensorflow graph
	  for i in range(n_epochs):
		  train_cost, _ = sess.run([cost, train_op], feed_dict={X_tf: train_X_ts, y_tf: train_y})
		  if i%100 == 0:
			  print("Training iteration", i, "MSE", train_cost)
               
	  # After the training, let's check the performance on the test set
	  test_cost, y_pr = sess.run([cost, y_pred], feed_dict={X_tf: test_X_ts, y_tf: test_y})
	  print("Test dataset:", test_cost)

	  # Evaluate the results
	  evaluate_ts(test_X, test_y, y_pr)
        
	  # How does the predicted look like?
	  plt.plot(range(len(cos_values)), cos_values, 'b')
	  plt.plot(range(len(cos_values)-test_size, len(cos_values)), y_pr, 'r--')
	  plt.xlabel("Days")
	  plt.ylabel("Predicted and true values")
	  plt.title("Predicted (Red) VS Real (Blue)")
	  plt.show()
```

经过超参数优化后，输出如下：

```
  Training iteration 0 MSE 0.0603129
  Training iteration 100 MSE 0.0054377
  Training iteration 200 MSE 0.00502512
  Training iteration 300 MSE 0.00483701 ...
  Training iteration 9700 MSE 0.0032881
  Training iteration 9800 MSE 0.00327899
  Training iteration 9900 MSE 0.00327195
  Test dataset: 0.00416444
  Evaluation of the predictions:
  MSE: 0.00416444
  mae: 0.0545878

```

模型的表现与本章用简单线性回归得到的模型表现很相近。那么，在股票价格这种不那么可预测的信号数据上，LSTM是否会表现的更好一些呢？接下来，本节会用之前小节中获取的时间序列数据，来比较模型的性能。
接下来，对之前的代码进行修改，把获取余弦数据替换成获取股票价格数据。修改几行以加载股票价格数据：

```python
  stock_values = fetch_stock_price(symbol, datetime.date(2015, 1, 1), datetime.date(2016, 12, 31))
  minibatch_cos_X, minibatch_cos_y = format_dataset(stock_values, time_dimension)
  train_X = minibatch_cos_X[:train_size, :].astype(np.float32)
  train_y = minibatch_cos_y[:train_size].reshape((-1, 1)).astype(np.float32)
  test_X = minibatch_cos_X[train_size:, :].astype(np.float32)
  test_y = minibatch_cos_y[train_size:].reshape((-1, 1)).astype(np.float32)
  train_X_ts = train_X[:, :, np.newaxis]
  test_X_ts = test_X[:, :, np.newaxis]
```

由于这个信号的振动范围更广，故还需要调整生成初始权重时的分布。建议读者设置为：

```python
  weights = tf.Variable(tf.truncated_normal([n_embeddings, 1], mean=0.0, stddev=10.0), name="weights")
```

经过几次测试，我们发现将参数作如下设置，模型的表现会最好：

```
  learning_rate = 0.1
  n_epochs = 5000
  n_embeddings = 256

```

使用上述参数，模型的输出如下：

```
  Training iteration 200 MSE 2.39028
  Training iteration 300 MSE 1.39495
  Training iteration 400 MSE 1.00994 ...
  Training iteration 4800 MSE 0.593951
  Training iteration 4900 MSE 0.593773
  Test dataset: 0.497867
  Evaluation of the predictions:
  MSE: 0.497867
  mae: 0.494975


```

LSTM的结果相比于之前的模型，有8%的提升（在测试集上的MSE）。读者需谨记，模型的结果是预测的价格！越多的参数需要训练也意味着需要比之前的模型更耗费时间（需要在有GPU的笔记本上花费几分钟）。

最后，本节讲述Tensorboard的使用。为了打印日志，需要添加以下代码：

1. 在文件的开头，引入包后：

```python
  import os
  tf_logdir = "./logs/tf/stock_price_lstm" 
  os.makedirs(tf_logdir, exist_ok=1)
```

1. RNN函数整体需要在LSTM的命名空间中，即：

```python
  def RNN(x, weights, biases):
	  with tf.name_scope("LSTM"):
		  x_ = tf.unstack(x, time_dimension, 1)
		  lstm_cell = rnn.BasicLSTMCell(n_embeddings)
		  outputs, _ = rnn.static_rnn(lstm_cell, x_, dtype=tf.float32)  
		  return tf.add(biases, tf.matmul(outputs[-1], weights))
```

1. 类似的，损失函数也需要写在Tensorflow的范围内。同样的，本节会在`tensorflow`图中添加`mae`的计算方法：

```python
  y_pred = RNN(X_tf, weights, biases)
  with tf.name_scope("cost"):
	  cost = tf.reduce_mean(tf.square(y_tf - y_pred))
	  train_op = optimizer(learning_rate).minimize(cost)
	  tf.summary.scalar("MSE", cost)
	  with tf.name_scope("mae"):
		  mae_cost = tf.reduce_mean(tf.abs(y_tf - y_pred))
		  tf.summary.scalar("mae", mae_cost)
4. 经过上面几步，主函数如下：
  with tf.Session() as sess:
	  writer = tf.summary.FileWriter(tf_logdir, sess.graph)  
	  merged = tf.summary.merge_all()
	  sess.run(tf.global_variables_initializer())
  
	  # For each epoch, the whole training set is feeded into the tensorflow graph
	  for i in range(n_epochs):
		  summary, train_cost, _ = sess.run([merged, cost, train_op], feed_dict={X_tf:train_X_ts, y_tf: train_y})
		  writer.add_summary(summary, i)
		  if i%100 == 0:
			  print("Training iteration", i, "MSE", train_cost)
  		  
	  # After the training, let's check the performance on the test set
	  test_cost, y_pr = sess.run([cost, y_pred], feed_dict={X_tf: test_X_ts, y_tf:test_y})
	  print("Test dataset:", test_cost)
```

这样就可以将每个块的范围分离出来，并为可训练的变量生成一份报告。
接下来加载`tensorboard`：

```
  $> tensorboard --logdir=./logs/tf/stock_price_lstm

```

  打开浏览器，输入`localhost:6006`，读者在第一个选项卡中可以看到MSE和MAE的曲线：
  <img src="E:\我的\翻译\137_1.jpg" style="zoom:55%" align="center" /> 
图中趋势不错，曲线在一开始下降，然后趋于平稳。读者也可以查看`tensorflow`图（在**GRAPH**选项卡中）。在这个选项卡中，读者可以看到模型的各个组成部分如何互相连接，以及所做的运算是如何相互影响的。读者也可以放大查看LSTM是如何在Tensorflow建立的：
<img src="E:\我的\翻译\138_1.jpg" style="zoom:70%" align="center" /> 
到此为止，这个项目就结束了。

### 问题思考

- 把LSTM替换成RNN，再替换成GRU。哪个表现最好？
- 不预测收盘价格，而去预测第二天的高/低价格，读者在训练模型的时候可以用相同的特征（或者读者可以用收盘价格作为输入）。
- 优化模型以适用于其他的股票：用一个适用于所有股票的通用模型比较好，还是为每个股票做一个模型比较好？
- 调优再训练。在此例中，本章预测了一年的股票价格。如果每个月/周/天训练一次模型，模型效果有什么提升吗？
- 如果读者有些金融背景，可以试着建立一个简单的交易模拟器，并按照预测结果来交易。初始资金是$100，一年后，你是赚了还是赔了？

### 总结

本章展示了如何进行时间序列的预测：具体地，本章带领读者观察到RNN在真实的股票价格数据上表现如何。下一章将讲述RNN的另一个应用，例如，如何自动地将句子从一种语言翻译成另一种语言。