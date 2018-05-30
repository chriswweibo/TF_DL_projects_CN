## 第9章 构建TensorFlow推荐系统

A recommender system is an algorithm that makes personalized suggestions to users based
on their past interactions with the software. The most famous example is the "customers
who bought X also bought Y" type of recommendation on Amazon and other e-commerce
websites.
In the past few years, recommender systems have gained a lot of importance: it has become
clear for the online businesses that the better the recommendations they give on their
websites, the more money they make. This is why today almost every website has a block
with personalized recommendations.
In this chapter, we will see how we can use TensorFlow to build our own recommender
system.
In particular, we will cover the following topics:
Basics of recommender systems
Matrix Factorization for recommender systems
Bayesian Personalized Ranking
Advanced recommender systems based on Recurrent Neural Nets
By the end of this chapter, you will know how to prepare data for training a recommender
system, how to build your own models with TensorFlow, and how to perform a simple
evaluation of the quality of these models.
Building a TensorFlow Recommender System Chapter 9
[ 213 ]
Recommender systems
The task of a recommender system is to take a list of all possible items and rank them
according to preferences of particular users. This ranked list is referred to as a personalized
ranking, or, more often, as a recommendation.
For example, a shopping website may have a section with recommendations where users
can see items that they may find relevant and could decide to buy. Websites selling tickets
to concerts may recommend interesting shows, and an online music player may suggest
songs that the user is likely to enjoy. Or a website with online courses, such as
Coursera.org, may recommend a course similar to ones the user has already finished:
Course recommendation on website
The recommendations are typically based on historical data: the past transaction history,
visits, and clicks that the users have made. So, a recommender system is a system that takes
this historical data and uses machine learning to extract patterns in the behavior of the users
and based on that comes up with the best recommendations.
Companies are quite interested in making the recommendations as good as possible: this
usually makes users engaged by improving their experience. Hence, it brings the revenue
up. When we recommend an item that the user otherwise would not notice, and the user
buys it, not only do we make the user satisfied, but we also sell an item that we would not
otherwise have sold.
Building a TensorFlow Recommender System Chapter 9
[ 214 ]
This chapter project is about implementing multiple recommender system algorithms using
TensorFlow. We will start with classical time-proven algorithms and then go deeper and try
a more complex model based on RNN and LSTM. For each model in this chapter, we will
first give a short introduction and then we implement this model in TensorFlow.
To illustrate these ideas, we use the Online Retail Dataset from the UCI Machine Learning
repository. This dataset can be downloaded from http:/ / archive. ics. uci. edu/ ml/
datasets/ online+retail.
The dataset itself is an Excel file with the following features:
InvoiceNo: The invoice number, which is used to uniquely identify each
transaction
StockCode: The code of the purchased item
Description: The name of the product
Quantity: The number of times the item is purchased in the transaction
UnitPrice: Price per item
CustomerID: The ID of the customer
Country: The name of the customer's country of the customer
It consists of 25,900 transactions, with each transaction containing around 20 items. This
makes approximately 540,000 items in total. The recorded transactions were made by 4,300
users starting from December 2010 up until December 2011.
To download the dataset, we can either use the browser and save the file or use wget:
wget
http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Ret
ail.xlsx
For this project, we will use the following Python packages:
pandas for reading the data
numpy and scipy for numerical data manipulations
tensorflow for creating the models
implicit for the baseline solution
[optional] tqdm for monitoring the progress
[optional] numba for speeding up the computations
Building a TensorFlow Recommender System Chapter 9
[ 215 ]
If you use Anaconda, then you should already have numba installed, but if not, a simple pip
install numba will get this package for you. To install implicit, we again use pip:
pip install implicit
Once the dataset is downloaded and the packages are installed, we are ready to start. In the
next section, we will review the Matrix Factorization techniques, then prepare the dataset,
and finally implement some of them in TensorFlow.
Matrix factorization for recommender
systems
In this section, we will go over traditional techniques for recommending systems. As we
will see, these techniques are really easy to implement in TensorFlow, and the resulting
code is very flexible and easily allows modifications and improvements.
For this section, we will use the Online Retail Dataset. We first define the problem we want
to solve and establish a few baselines. Then we implement the classical Matrix factorization
algorithm as well as its modification based on Bayesian Personalized Ranking.
Dataset preparation and baseline
Now we are ready to start building a recommender system.
First, declare the imports:
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
Let us read the dataset:
df = pd.read_excel('Online Retail.xlsx')
Building a TensorFlow Recommender System Chapter 9
[ 216 ]
Reading xlsx files may take a while. To save time when we next want to read the file, we
can save the loaded copy into a pickle file:
import pickle
with open('df_retail.bin', 'wb') as f_out:
pickle.dump(df, f_out)
This file is a lot faster to read, so for loading, we should use the pickled version:
with open('df_retail.bin', 'rb') as f_in:
df = pickle.load(f_in)
Once the data is loaded, we can have a look at the data. We can do this by invoking the
head function:
df.head()
We then see the following table:
If we take a closer look at the data, we can notice the following problems:
The column names are in capital letters. This is a bit unusual, so we may
lowercase them.
Some of the transactions are returns: they are not of interest to us, so we should
filter them out.
Finally, some of the transactions belong to unknown users. We can assign some
common ID for these users, for example, -1. Also, unknown users are encoded as
NaNs, this is why the CustomerID column is encoded as float—so we need to
convert it to an integer.
Building a TensorFlow Recommender System Chapter 9
[ 217 ]
These problems can be fixed with the following code:
df.columns = df.columns.str.lower()
df =
df[~df.invoiceno.astype('str').str.startswith('C')].reset_index(drop=True)
df.customerid = df.customerid.fillna(-1).astype('int32')
Next, we should encode all item IDs (stockcode) with integers. One of the ways to do it is
to build a mapping from each code to some unique index number:
stockcode_values = df.stockcode.astype('str')
stockcodes = sorted(set(stockcode_values))
stockcodes = {c: i for (i, c) in enumerate(stockcodes)}
df.stockcode = stockcode_values.map(stockcodes).astype('int32')
Now after we have encoded the items, we can split the dataset into train, validation, and
test parts. Since we have e-commerce transactions data, the most sensible way to do the
split is based on time. So we will use:
Training set: before 2011.10.09 (around 10 months of data, approximately 378,500
rows)
Validation set: between 2011.10.09 and 2011.11.09 (one month of data,
approximately 64,500 rows)
Test set: after 2011.11.09 (also one month, approximately 89,000 rows)
For that we just filter the dataframes:
df_train = df[df.invoicedate < '2011-10-09']
df_val = df[(df.invoicedate >= '2011-10-09') &
(df.invoicedate <= '2011-11-09') ]
df_test = df[df.invoicedate >= '2011-11-09']
In this section, we will consider the following (very simplified) recommendation scenario:

1. The user enters the website.
2. We present five recommendations.
3. The user assesses the lists, maybe buys some things from there, and then
  continues shopping as usual.
  So we need to build a model for the second step. To do so, we use the training data and then
  simulate the second and third steps using the validation set. To evaluate whether our
  recommendation was good or not, we count the number of recommended items that the
  user has actually bought.
  Building a TensorFlow Recommender System Chapter 9
  [ 218 ]
  Our evaluation measure is then the number of successful recommendations (the items the
  user has actually bought) divided by the number of total recommendations we made. This
  is called precision—a common measure of evaluating the performance of machine learning
  models.
  For this project we use precision. Of course, it is a rather simplistic way of evaluating the
  performance, and there are different ways of doing this. Other metrics you may want to use
  include MAP (Mean Average Precision), NDCG (Normalized Discounted Cumulative
  Gain), and so on. For simplicity, however, we do not use them in this chapter.
  Before we jump into using machine learning algorithm for this task, let us first establish a
  basic baseline. For example, we can calculate how many times each item was bought, then
  take the most frequent five items, and recommend these items to all the users.
  With pandas it is easy to do:
  top = df_train.stockcode.value_counts().head(5).index.values
  This gives us an array of integers—stockcode codes:
  array([3527, 3506, 1347, 2730, 180])
  Now we use this array to recommend it to all the users. So we repeat the top array as many
  times as there are transactions in the validation dataset, and then we use this as the
  recommendations and calculate the precision metric to evaluate the quality.
  For repeating we use the tile function from numpy:
  num_groups = len(df_val.invoiceno.drop_duplicates())
  baseline = np.tile(top, num_groups).reshape(-1, 5)
  The tile function takes in an array and repeats it num_group times. After reshaping, it
  gives us the following array:
  array([[3527, 3506, 1347, 2730, 180],
  [3527, 3506, 1347, 2730, 180],
  [3527, 3506, 1347, 2730, 180],
  ...,
  [3527, 3506, 1347, 2730, 180],
  [3527, 3506, 1347, 2730, 180],
  [3527, 3506, 1347, 2730, 180]])
  Now we are ready to calculate the precision of this recommendation.
  Building a TensorFlow Recommender System Chapter 9
  [ 219 ]
  However, there is a complication: the way the items are stored makes it difficult to calculate
  the number of correctly classified elements per group. Using groupby from pandas is one
  way of solving the problem:
  Group by invoiceno (this is our transaction ID)
  For each transaction make a recommendation
  Record the number of correct predictions per each group
  Calculate the overall precision
  However, this way is often very slow and inefficient. It may work fine for this particular
  project, but for slightly larger datasets it becomes a problem.
  The reason it is slow is the way groupby is implemented in pandas: it internally performs
  sorting, which we do not need. However, we can improve the speed by exploiting the way
  the data is stored: we know that the elements of our dataframe are always ordered. That is,
  if a transaction starts at a certain row number i, then it ends at the number i + k, where k
  is the number of items in this transaction. In other words, all the rows between i and i + k
  belong to the same invoiceid.
  So we need to know where each transaction starts and where it ends. For this purpose, we
  keep a special array of length n + 1, where n is the number of groups (transactions) we
  have in our dataset.
  Let us call this array indptr. For each transaction t:
  indptr[t] returns the number of the row in the dataframe where the transaction
  starts
  indptr[t + 1] returns the row where it ends
  This way of representing the groups of various length is inspired by the
  CSR algorithm—Compressed Row Storage (sometimes Compressed
  Sparse Row). It is used to represent sparse matrices in memory. You can
  read about it more in the Netlib documentation—http:/ / netlib. org/
  linalg/ html_ templates/ node91. html. You may also recognize this name
  from scipy—it is one of the possible ways of representing matrices in the
  scipy.sparse package: https:/ / docs. scipy. org/ doc/ scipy- 0. 14. 0/
  reference/ generated/ scipy. sparse. csr_ matrix. html.
  Building a TensorFlow Recommender System Chapter 9
  [ 220 ]
  Creating such arrays is not difficult in Python: we just need to see where the current
  transaction finishes and the next one starts. So at each row index, we can compare the
  current index with the previous one, and if it is different, record the index. This can be done
  efficiently using the shift method from pandas:
  def group_indptr(df):
  indptr, = np.where(df.invoiceno != df.invoiceno.shift())
  indptr = np.append(indptr, len(df)).astype('int32')
  return indptr
  This way we get the pointers array for the validation set:
  val_indptr = group_indptr(df_val)
  Now we can use it for the precision function:
  from numba import njit
  @njit
  def precision(group_indptr, true_items, predicted_items):
  tp = 0
  n, m = predicted_items.shape
  for i in range(n):
  group_start = group_indptr[i]
  group_end = group_indptr[i + 1]
  group_true_items = true_items[group_start:group_end]
  for item in group_true_items:
  for j in range(m):
  if item == predicted_items[i, j]:
  tp = tp + 1
  continue
  return tp / (n * m)
  Here the logic is straightforward: for each transaction we check how many items we
  predicted correctly. The total amount of correctly predicted items is stored in tp. At the end
  we divide tp by the total number of predictions, which is the size of the prediction matrix,
  that is, number of transactions times five in our case.
  Building a TensorFlow Recommender System Chapter 9
  [ 221 ]
  Note the @njit decorator from numba. This decorator tells numba that the code should be
  optimized. When we invoke this function for the first time, numba analyzes the code and
  uses the just-in-time (JIT) compiler to translate the function to native code. When the
  function is compiled, it runs multiple orders of magnitude faster—comparable to native
  code written in C.
  Numba's @jit and @njit decorators give a very easy way to improve the
  speed of the code. Often it is enough just to put the @jit decorator on a
  function to see a significant speed-up. If a function takes time to compute,
  numba is a good way to improve the performance.
  Now we can check what is the precision of this baseline:
  val_items = df_val.stockcode.values
  precision(val_indptr, val_items, baseline)
  Executing this code should produce 0.064. That is, in 6.4% of the cases we made the correct
  recommendation. This means that the user ended up buying the recommended item only in
  6.4% cases.
  Now when we take a first look at the data and establish a simple baseline, we can proceed
  to more complex techniques such as matrix factorization.
  Matrix factorization
  In 2006 Netflix, a DVD rental company, organized the famous Netflix competition. The goal
  of this competition was to improve their recommender system. For this purpose, the
  company released a large dataset of movie ratings. This competition was notable in a few
  ways. First, the prize pool was one million dollars, and that was one of the main reasons it
  became famous. Second, because of the prize, and because of the dataset itself, many
  researchers invested their time into this problem and that significantly advanced the state of
  the art in recommender systems.
  It was the Netflix competition that showed that recommenders based on matrix
  factorization are very powerful, can scale to a large number of training examples, and yet
  are not very difficult to implement and deploy.
  The paper Matrix factorization techniques for recommender systems by Koren and others
  (2009) nicely summarizes the key findings, which we will also present in this chapter.
  Building a TensorFlow Recommender System Chapter 9
  [ 222 ]
  Imagine we have the rating of a movie rated by user . We can model this rating by:
  .
  Here we decompose the rating into four factors:
  is the global bias
  is the bias of the item (in case of Netflix—movie)
  is the bias of the user
  is the inner product between the user vector and the item vector
  The last factor—the inner product between the user and the item vectors—is the reason this
  technique is called Matrix Factorization.
  Let us take all the user vectors , and put them into a matrix as rows. We then will
  have an matrix, where is the number of users and is the dimensionality of
  the vectors. Likewise, we can take the item vectors and put them into a matrix as
  rows. This matrix has the size , where is the number of items, and is again
  the dimensionality of the vectors. The dimensionality is a parameter of the model, which
  allows us to control how much we want to compress the information. The smaller is, the
  less information is preserved from the original rating matrix.
  Lastly, we take all the known ratings and put them into a matrix —this matrix is of
  size. Then this matrix can be factorized as
  .
  Without the biases part, this is exactly what we have when we compute in the
  preceding formula.
  Building a TensorFlow Recommender System Chapter 9
  [ 223 ]
  To make the predicted rating as close as possible to the observed rating rating , we
  minimize the squared error between them. That is, our training objective is the following:
  This way of factorizing the rating matrix is sometimes called SVD because it is inspired by
  the classical Singular Value Decomposition method—it also optimizes the sum of squared
  errors. However, the classical SVD often tends to overfit to the training data, which is why
  here we include the regularization term in the objective.
  After defining the optimization problem, the paper then talks about two ways of solving it:
  Stochastic Gradient Descent (SGD)
  Alternating Least Squares (ALS)
  Later in this chapter, we will use TensorFlow to implement the SGD method ourselves and
  compare it to the results of the ALS method from the implicit library.
  However, the dataset we use for this project is different from the Netflix competition
  dataset in a very important way—we do not know what the users do not like. We only
  observe what they like. That is why next we will talk about ways to handle such cases.
  Implicit feedback datasets
  In case of the Netflix competition, the data there relies on the explicit feedback given by the
  users. The users went to the website and explicitly told them how much they like the movie
  by giving it a rating from 1 to 5.
  Typically it is quite difficult to make users do that. However, just by visiting the website
  and interacting with it, the users already generated a lot of useful information, which can be
  used to infer their interests. All the clicks, page visits, and past purchases tell us about the
  preferences of the user. This kind of data is called implicit - the users do not explicitly tell
  us what they like, but instead, they indirectly convey this information to us by using the
  system. By collecting this interaction information we get implicit feedback datasets.
  The Online Retail Dataset we use for this project is exactly that. It tells us what the users
  previously bought, but does not tell us what the users do not like. We do not know if the
  users did not buy an item because they did not like it, or just because they did not know the
  item existed.
  Building a TensorFlow Recommender System Chapter 9
  [ 224 ]
  Luckily for us, with minor modification, we still can apply the Matrix Factorization
  techniques to implicit datasets. Instead of the explicit ratings, the matrix takes values of 1
  and 0—depending on whether there was an interaction with the item or not. Additionally, it
  is possible to express the confidence that the value 1 or 0 is indeed correct, and this is
  typically done by counting how many times the users have interacted with the item. The
  more times they interact with it, the larger our confidence becomes.
  So, in our case all values that the user has bought get the value 1 in the matrix, and all the
  rest are 0's. Thus we can see this is a binary classification problem and implement an SGDbased
  model in TensorFlow for learning the user and item matrices.
  But before we do that, we will establish another baseline have stronger than the previous
  one. We will use the implicit library, which uses ALS.
  Collaborative Filtering for Implicit Feedback Datasets by Hu et al (2008) gives a
  good introduction to the ALS method for implicit feedback datasets. We
  do not focus on ALS in this chapter, but if you want to learn how ALS is
  implemented in libraries such as implicit, this paper is definitely a great
  source. At the time of writing, the paper was accessible via http:/ /
  yifanhu. net/ PUB/ cf. pdf.
  First, we need to prepare the data in the format implicit expects—and for that we need to
  construct the user-item matrix X. For that we need to translate both users and items to IDs,
  so we can map each user to a row of X, and each item—to the column of X.
  We have already converted items (the column stockcode) to integers. How we need to
  perform the same on the user IDs (the column customerid):
  df_train_user = df_train[df_train.customerid != -1].reset_index(drop=True)
  customers = sorted(set(df_train_user.customerid))
  customers = {c: i for (i, c) in enumerate(customers)}
  df_train_user.customerid = df_train_user.customerid.map(customers)
  Note that in the first line we perform the filtering and keep only known users there—these
  are the users we will use for training the model afterward. Then we apply the same
  procedure to the users in the validation set:
  df_val.customerid = df_val.customerid.apply(lambda c: customers.get(c, -1))
  Building a TensorFlow Recommender System Chapter 9
  [ 225 ]
  Next we use these integer codes to construct the matrix X:
  uid = df_train_user.customerid.values.astype('int32')
  iid = df_train_user.stockcode.values.astype('int32')
  ones = np.ones_like(uid, dtype='uint8')
  X_train = sp.csr_matrix((ones, (uid, iid)))
  The sp.csr_matrix is a function from the scipy.sparse package. It takes in the rows
  and column indicies plus the corresponding value for each index pair, and constructs a
  matrix in the Compressed Storage Row format.
  Using sparse matrices is a great way to reduce the space consumption of
  data matrices. In recommender systems there are many users and many
  items. When we construct a matrix, we put zeros for all the items the user
  has not interacted with. Keeping all these zeros is wasteful, so sparse
  matrices give a way to store only non-zero entries. You can read more
  about them in the scipy.sparse package documentation at https:/ /
  docs. scipy. org/ doc/ scipy/ reference/ sparse. html.
  Now let us use implicit to factorize the matrix X and learn the user and item vectors:
  from implicit.als import AlternatingLeastSquares
  item_user = X_train.T.tocsr()
  als = AlternatingLeastSquares(factors=128, regularization=0.000001)
  als.fit(item_user)
  To use ALS we use the AlternatingLeastSquares class. It takes two parameters:
  factors: this is the dimensionality of the user and item vectors, which we called
  previously k
  regularization: the L2 regularization parameter to avoid overfitting
  Then we invoke the fit function to learn the vectors. Once the training is done, these
  vectors are easy to get:
  als_U = als.user_factors
  als_I = als.item_factors
  After getting the U and I matrices, we can use them to make recommendations to the user,
  and for that, we simply calculate the inner product between the rows of each matrix. We
  will see soon how to do it.
  Building a TensorFlow Recommender System Chapter 9
  [ 226 ]
  Matrix factorization methods have a problem: they cannot deal with new users. To
  overcome this problem, we can simply combine it with the baseline method: use the
  baseline to make a recommendation to new and unknown users, but apply Matrix
  Factorization to known users.
  So, first we select the IDs of known users in the validation set:
  uid_val = df_val.drop_duplicates(subset='invoiceno').customerid.values
  known_mask = uid_val != -1
  uid_val = uid_val[known_mask]
  We will make recommendations only to these users. Then we copy the baseline solution,
  and replace the prediction for the known users by values from ALS:
  imp_baseline = baseline.copy()
  pred_all = als_U[uid_val].dot(als_I.T)
  top_val = (-pred_all).argsort(axis=1)[:, :5]
  imp_baseline[known_mask] = top_val
  prevision(val_indptr, val_items, imp_baseline)
  Here we get the vectors for each user ID in the validation set and multiply them with all the
  item vectors. Next, for each user we select top five items according to the score.
  This outputs 13.9%. This is a lot stronger baseline than our previous baseline of 6%. This
  should be a lot more difficult to outperform, but next, we nonetheless try to do it.
  SGD-based matrix factorization
  Now we are finally ready to implement the matrix factorization model in TensorFlow. Let
  us do this and see if we can improve the baseline by implicit. Implementing ALS in
  TensorFlow is not an easy task: it is better suited for gradient-based methods such as SGD.
  This is why we will do exactly that, and leave ALS to specialized implementations.
  Here we implement the formula from the previous sections:
  .
  Building a TensorFlow Recommender System Chapter 9
  [ 227 ]
  Recall that the objective there was the following:
  Note that in this objective we still have the squared error, which is no longer the case for us
  since we model this as a binary classification problem. With TensorFlow it does not really
  matter, and the optimization loss can easily be changed.
  In our model we will use the log loss instead—it is better suited for binary classification
  problems than squared error.
  The p and q vectors make up the U and I matrices, respectively. What we need to do is to
  learn these U and I matrices. We can store the full matrices U and I as a TensorFlow
  Variable's and then use the embedding layer to look up the appropriate p and q vectors.
  Let us define a helper function for declaring embedding layers:
  def embed(inputs, size, dim, name=None):
  std = np.sqrt(2 / dim)
  emb = tf.Variable(tf.random_uniform([size, dim], -std, std), name=name)
  lookup = tf.nn.embedding_lookup(emb, inputs)
  return lookup
  This function creates a matrix of the specified dimension, initializes it with random values,
  and finally uses the lookup layer to convert user or item indexes into vectors.
  We use this function as a part of the model graph:
# parameters of the model
num_users = uid.max() + 1
num_items = iid.max() + 1
num_factors = 128
lambda_user = 0.0000001
lambda_item = 0.0000001
K = 5
lr = 0.005
Building a TensorFlow Recommender System Chapter 9
[ 228 ]
graph = tf.Graph()
graph.seed = 1
with graph.as_default():
# this is the input to the model
place_user = tf.placeholder(tf.int32, shape=(None, 1))
place_item = tf.placeholder(tf.int32, shape=(None, 1))
place_y = tf.placeholder(tf.float32, shape=(None, 1))
# user features
user_factors = embed(place_user, num_users, num_factors,
"user_factors")
user_bias = embed(place_user, num_users, 1, "user_bias")
user_bias = tf.reshape(user_bias, [-1, 1])
# item features
item_factors = embed(place_item, num_items, num_factors,
"item_factors")
item_bias = embed(place_item, num_items, 1, "item_bias")
item_bias = tf.reshape(item_bias, [-1, 1])
global_bias = tf.Variable(0.0, name='global_bias')
# prediction is dot product followed by a sigmoid
pred = tf.reduce_sum(user_factors * item_factors, axis=2)
pred = tf.sigmoid(global_bias + user_bias + item_bias + pred)
reg = lambda_user * tf.reduce_sum(user_factors * user_factors) + \
lambda_item * tf.reduce_sum(item_factors * item_factors)
# we have a classification model, so minimize logloss
loss = tf.losses.log_loss(place_y, pred)
loss_total = loss + reg
opt = tf.train.AdamOptimizer(learning_rate=lr)
step = opt.minimize(loss_total)
init = tf.global_variables_initializer()
The model gets three inputs:
place_user: The user IDs
place_item: The item IDs
place_y: The labels of each (user, item) pair
Building a TensorFlow Recommender System Chapter 9
[ 229 ]
Then we define:
user_factors: The user matrix
user_bias: The bias of each user
item_factors: The item matrix
item_bias: The bias of each item
global_bias: The global bias
Then, we put together all the biases and take the dot product between the user and item
factors. This is our prediction, which we then pass through the sigmoid function to get
probabilities.
Finally, we define our objective function as a sum of the data loss and regularization loss
and use Adam for minimizing this objective.
The model has the following parameters:
num_users and num_items: The number of users (items). They specify the
number of rows in U and I matrices, respectively.
num_factors: The number of latent features for users and items. This specifies
the number of columns in both U and I.
lambda_user and lambda_item: The regularization parameters.
lr: Learning rate for the optimizer.
K: The number of negative examples to sample for each positive case (see the
explanation in the following section).
Now let us train the model. For that, we need to cut the input into small batches. Let us use
a helper function for that:
def prepare_batches(seq, step):
n = len(seq)
res = []
for i in range(0, n, step):
res.append(seq[i:i+step])
return res
This will turn one array into a list of arrays of specified size.
Building a TensorFlow Recommender System Chapter 9
[ 230 ]
Recall that our dataset is based on implicit feedback, and the number positive
instances—interactions that did occur—is very small compared to the number of negative
instances—the interactions that did not occur. What do we do with it? The solution is
simple: we use negative sampling. The idea behind it is to sample only a small fraction of
negative examples. Typically, for each positive example, we sample K negative examples,
and K is a tunable parameter. And this is exactly what we do here.
So let us train the model:
session = tf.Session(config=None, graph=graph)
session.run(init)
np.random.seed(0)
for i in range(10):
train_idx_shuffle = np.arange(uid.shape[0])
np.random.shuffle(train_idx_shuffle)
batches = prepare_batches(train_idx_shuffle, 5000)
progress = tqdm(total=len(batches))
for idx in batches:
pos_samples = len(idx)
neg_samples = pos_samples * K
label = np.concatenate([
np.ones(pos_samples, dtype='float32'),
np.zeros(neg_samples, dtype='float32')
]).reshape(-1, 1)
# negative sampling
neg_users = np.random.randint(low=0, high=num_users,
size=neg_samples, dtype='int32')
neg_items = np.random.randint(low=0, high=num_items,
size=neg_samples, dtype='int32')
batch_uid = np.concatenate([uid[idx], neg_users]).reshape(-1, 1)
batch_iid = np.concatenate([iid[idx], neg_items]).reshape(-1, 1)
feed_dict = {
place_user: batch_uid,
place_item: batch_iid,
place_y: label,
}
_, l = session.run([step, loss], feed_dict)
progress.update(1)
progress.set_description('%.3f' % l)
Building a TensorFlow Recommender System Chapter 9
[ 231 ]
progress.close()
val_precision = calculate_validation_precision(graph, session, uid_val)
print('epoch %02d: precision: %.3f' % (i+1, val_precision))
We run the model for 10 epochs, then for each epoch we shuffle the data randomly and cut
it into batches of 5000 positive examples. Then for each batch, we generate K * 5000 negative
examples (K = 5 in our case) and put positive and negative examples together in one array.
Finally, we run the model, and at each update step, we monitor the training loss using
tqdm. The tqdm library provides a very nice way to monitor the training progress.
This is the output we produce when we use the tqdm jupyter notebook widgets:
At the end of each epoch, we calculate precision—to monitor how our model is performing
for our defined recommendation scenario. The calculate_validation_precision
function is used for that. It is implemented in a similar way to what we did previously with
implicit:
We first extract the matrices and the biases
Then put them together to get the score for each (user, item) pair
Finally, we sort these pairs and keep the top five ones
For this particular case we do not need the global bias as well as the user bias: adding them
will not change the order of items per user. This is how this function can be implemented:
def get_variable(graph, session, name):
v = graph.get_operation_by_name(name)
v = v.values()[0]
v = v.eval(session=session)
return v
def calculate_validation_precision(graph, session, uid):
Building a TensorFlow Recommender System Chapter 9
[ 232 ]
U = get_variable(graph, session, 'user_factors')
I = get_variable(graph, session, 'item_factors')
bi = get_variable(graph, session, 'item_bias').reshape(-1)
pred_all = U[uid_val].dot(I.T) + bi
top_val = (-pred_all).argsort(axis=1)[:, :5]
imp_baseline = baseline.copy()
imp_baseline[known_mask] = top_val
return precision(val_indptr, val_items, imp_baseline)
This is the output we get:
epoch 01: precision: 0.064
epoch 02: precision: 0.086
epoch 03: precision: 0.106
epoch 04: precision: 0.127
epoch 05: precision: 0.138
epoch 06: precision: 0.145
epoch 07: precision: 0.150
epoch 08: precision: 0.149
epoch 09: precision: 0.151
epoch 10: precision: 0.152
By the sixth epoch it beats the previous baseline, and by the tenth, it reaches 15.2%.
Matrix factorization techniques usually give a very strong baseline solution for
recommender systems. But with a small adjustment, the same technique can produce even
better results. Instead of optimizing a loss for binary classification, we can use a different
loss designed specifically for ranking problems. In the next section, we will learn about this
kind of loss and see how to make this adjustment.
Bayesian personalized ranking
We use Matrix factorization methods for making a personalized ranking of items for each
user. However, to solve this problem we use a binary classification optimization
criterion—the log loss. This loss works fine and optimizing it often produces good ranking
models. What if instead we could use a loss specifically designed for training a ranking
function?
Building a TensorFlow Recommender System Chapter 9
[ 233 ]
Of course, it is possible to use an objective that directly optimizes for ranking. In the paper
BPR: Bayesian Personalized Ranking from Implicit Feedback by Rendle et al (2012), the authors
propose an optimization criterion, which they call BPR-Opt.
Previously, we looked at individual items in separation from the other items. That is, we
tried to predict the rating of an item, or the probability that the item i will be interesting to
the user u. These kinds of ranking models are usually called "point-wise": they use
traditional supervised learning methods such as regression or classification to learn the
score, and then rank the items according to this score. This is exactly what we did in the
previous section.
BPR-Opt is different. Instead, it looks at the pairs of items. If we know that user u has
bought item i, but never bought item j, then most likely u is more interested in i than in j.
Thus, when we train a model, the score it produces for i should be higher than the
score for j. In other words, for the scoring model we want .
Therefore, for training this algorithm we need triples (user, positive item, negative item).
For such triple (u, i, j) we define the pair-wise difference in scores as:
where and is scores for (u, i) and (u, j), respectively.
When training, we adjust parameters of our model in such a way that at the end item i does
rank higher than item j. We do this by optimizing the following objective:
Where are the differences, is the sigmoid function, and is all the parameters of
the model.
It is straightforward to change our previous code to optimize this loss. The way we
compute the score for (u, i) and (u, j) is the same: we use the biases and the inner product
between the user and item vectors. Then we compute the difference between the scores and
feed the difference into the new objective.
Building a TensorFlow Recommender System Chapter 9
[ 234 ]
The difference in the implementation is also not large:
For BPR-Opt we do not have place_y, but instead, we will have
place_item_pos and place_item_neg for the positive and the negative items,
respectively.
We no longer need the user bias and the global bias: when we compute the
difference, these biases cancel each other out. What is more, they are not really
important for ranking—we have noted that previously when computing the
predictions for the validation data.
Another slight difference in implementation is that because we now have two inputs items,
and these items have to share the embeddings, we need to define and create the
embeddings slightly differently. For that we modify the embed helper function, and
separate the variable creation and the lookup layer:
def init_variable(size, dim, name=None):
std = np.sqrt(2 / dim)
return tf.Variable(tf.random_uniform([size, dim], -std, std),
name=name)
def embed(inputs, size, dim, name=None):
emb = init_variable(size, dim, name)
return tf.nn.embedding_lookup(emb, inputs)
Finally, let us see how it looks in the code:
num_factors = 128
lambda_user = 0.0000001
lambda_item = 0.0000001
lambda_bias = 0.0000001
lr = 0.0005
graph = tf.Graph()
graph.seed = 1
with graph.as_default():
place_user = tf.placeholder(tf.int32, shape=(None, 1))
place_item_pos = tf.placeholder(tf.int32, shape=(None, 1))
place_item_neg = tf.placeholder(tf.int32, shape=(None, 1))
# no place_y
user_factors = embed(place_user, num_users, num_factors,
"user_factors")
# no user bias anymore as well as no global bias
Building a TensorFlow Recommender System Chapter 9
[ 235 ]
item_factors = init_variable(num_items, num_factors,
"item_factors")
item_factors_pos = tf.nn.embedding_lookup(item_factors, place_item_pos)
item_factors_neg = tf.nn.embedding_lookup(item_factors, place_item_neg)
item_bias = init_variable(num_items, 1, "item_bias")
item_bias_pos = tf.nn.embedding_lookup(item_bias, place_item_pos)
item_bias_pos = tf.reshape(item_bias_pos, [-1, 1])
item_bias_neg = tf.nn.embedding_lookup(item_bias, place_item_neg)
item_bias_neg = tf.reshape(item_bias_neg, [-1, 1])
# predictions for each item are same as previously
# but no user bias and global bias
pred_pos = item_bias_pos + \
tf.reduce_sum(user_factors * item_factors_pos, axis=2)
pred_neg = item_bias_neg + \
tf.reduce_sum(user_factors * item_factors_neg, axis=2)
pred_diff = pred_pos—pred_neg
loss_bpr =—tf.reduce_mean(tf.log(tf.sigmoid(pred_diff)))
loss_reg = lambda_user * tf.reduce_sum(user_factors * user_factors) +\
lambda_item * tf.reduce_sum(item_factors_pos * item_factors_pos)+\
lambda_item * tf.reduce_sum(item_factors_neg * item_factors_neg)+\
lambda_bias * tf.reduce_sum(item_bias_pos) + \
lambda_bias * tf.reduce_sum(item_bias_neg)
loss_total = loss_bpr + loss_reg
opt = tf.train.AdamOptimizer(learning_rate=lr)
step = opt.minimize(loss_total)
init = tf.global_variables_initializer()
The way to train this model is also slightly different. The authors of the BPR-Opt paper
suggest using the bootstrap sampling instead of the usual full-pass over all the data, that is,
at each training step we uniformly sample the triples (user, positive item, negative item)
from the training dataset.
Luckily, this is even easier to implement than the full-pass:
session = tf.Session(config=None, graph=graph)
session.run(init)
size_total = uid.shape[0]
size_sample = 15000
Building a TensorFlow Recommender System Chapter 9
[ 236 ]
np.random.seed(0)
for i in range(75):
for k in range(30):
idx = np.random.randint(low=0, high=size_total, size=size_sample)
batch_uid = uid[idx].reshape(-1, 1)
batch_iid_pos = iid[idx].reshape(-1, 1)
batch_iid_neg = np.random.randint(
low=0, high=num_items, size=(size_sample, 1), dtype='int32')
feed_dict = {
place_user: batch_uid,
place_item_pos: batch_iid_pos,
place_item_neg: batch_iid_neg,
}
_, l = session.run([step, loss_bpr], feed_dict)
val_precision = calculate_validation_precision(graph, session, uid_val)
print('epoch %02d: precision: %.3f' % (i+1, val_precision))
After around 70 iterations it reaches the precision of around 15.4%. While it is not
significantly different from the previous model (it reached 15.2%), it does open a lot of
possibilities for optimizing directly for ranking. More importantly, we show how easy it is
to adjust the existent method such that instead of optimizing the point-wise loss it
optimizes a pair-wise objective.
In the next section, we will go deeper and see how recurrent neural networks can model
user actions as sequences and how we can use them as recommender systems.
RNN for recommender systems
A recurrent neural networks (RNN) is a special kind of neural network for modeling
sequences, and it is quite successful in a number applications. One such application is
sequence generation. In the article The Unreasonable Effectiveness of Recurrent Neural
Networks, Andrej Karpathy writes about multiple examples where RNNs show very
impressive results, including generation of Shakespeare, Wikipedia articles, XML, Latex,
and even C code!
Building a TensorFlow Recommender System Chapter 9
[ 237 ]
Since they have proven useful in a few applications already, the natural question to ask is
whether we can apply RNNs to some other domains. What about recommender systems,
for example? This is the question the authors of the recurrent neural networks Based
Subreddit Recommender System report have asked themselves (see https:/ / cole- maclean.
github. io/ blog/ RNN- Based- Subreddit- Recommender- System/ ). The answer is yes, we can
use RNNs for that too!
In this section, we will try to answer this question as well. For this part we consider a
slightly different recommendation scenario than previously:
1. The user enters the website.
2. We present five recommendations.
3. After each purchase, we update the recommendations.
  This scenario needs a different way of evaluating the results. Each time the user buys
  something, we can check whether this item was among the suggested ones or not. If it was,
  then our recommendation is considered successful. So we can calculate how many
  successful recommendations we have made. This way of evaluating performance is called
  Top-5 accuracy and it is often used for evaluating classification models with a large number
  of target classes.
  Historically RNNs are used for language models, that is, for predicting what will be the
  most likely next word given in the sentence so far. And, of course, there is already an
  implementation of such a language model in the TensorFlow model repository located at
  https:/ / github. com/ tensorflow/ models (in the tutorials/rnn/ptb/ folder). Some of
  the code samples in the remaining of this chapter are heavily inspired by this example.
  So let us get started.
  Data preparation and baseline
  Like previously, we need to represent the items and users as integers. This time, however,
  we need to have a special placeholder value for unknown users. Additionally, we need a
  special placeholder for items to represent "no item" at the beginning of each transaction. We
  will talk more about it later in this section, but for now, we need to implement the encoding
  such that the 0 index is reserved for special purposes.
  Building a TensorFlow Recommender System Chapter 9
  [ 238 ]
  Previously we were using a dictionary, but this time let us implement a special
  class, LabelEncoder, for this purpose:
  class LabelEncoder:
  def fit(self, seq):
  self.vocab = sorted(set(seq))
  self.idx = {c: i + 1 for i, c in enumerate(self.vocab)}
  def transform(self, seq):
  n = len(seq)
  result = np.zeros(n, dtype='int32')
  for i in range(n):
  result[i] = self.idx.get(seq[i], 0)
  return result
  def fit_transform(self, seq):
  self.fit(seq)
  return self.transform(seq)
  def vocab_size(self):
  return len(self.vocab) + 1
  The implementation is straightforward and it largely repeats the code we used previously,
  but this time it is wrapped in a class, and also reserves 0 for special needs—for example, for
  elements that are missing in the training data.
  Let us use this encoder to convert the items to integers:
  item_enc = LabelEncoder()
  df.stockcode = item_enc.fit_transform(df.stockcode.astype('str'))
  df.stockcode = df.stockcode.astype('int32')
  Then we perform the same train-validation-test split: first 10 months we use for training,
  one for validation and the last one—for testing.
  Next, we encode the user ids:
  user_enc = LabelEncoder()
  user_enc.fit(df_train[df_train.customerid != -1].customerid)
  df_train.customerid = user_enc.transfrom(df_train.customerid)
  df_val.customerid = user_enc.transfrom(df_val.customerid)
  Building a TensorFlow Recommender System Chapter 9
  [ 239 ]
  Like previously, we use the most frequently bought items for the baseline. However, this
  time the scenario is different, which is why we also adjust the baseline slightly. In
  particular, if one of the recommended items is bought by the user, we remove it from the
  future recommendations.
  Here is how we can implement it:
  from collections import Counter
  top_train = Counter(df_train.stockcode)
  def baseline(uid, indptr, items, top, k=5):
  n_groups = len(uid)
  n_items = len(items)
  pred_all = np.zeros((n_items, k), dtype=np.int32)
  for g in range(n_groups):
  t = top.copy()
  start = indptr[g]
  end = indptr[g+1]
  for i in range(start, end):
  pred = [k for (k, c) in t.most_common(5)]
  pred_all[i] = pred
  actual = items[i]
  if actual in t:
  del t[actual]
  return pred_all
  In the preceding code, indptr is the array of pointers—the same one that we used for
  implementing the precision function previously.
  So now we apply this to the validation data and produce the results:
  iid_val = df_val.stockcode.values
  pred_baseline = baseline(uid_val, indptr_val, iid_val, top_train, k=5)
  The baseline looks as follows:
  array([[3528, 3507, 1348, 2731, 181],
  [3528, 3507, 1348, 2731, 181],
  [3528, 3507, 1348, 2731, 181],
  ...,
  [1348, 2731, 181, 454, 1314],
  Building a TensorFlow Recommender System Chapter 9
  [ 240 ]
  [1348, 2731, 181, 454, 1314],
  [1348, 2731, 181, 454, 1314]], dtype=int32
  Now let us implement the top-k accuracy metric. We again use the @njit decorator from
  numba to speed this function up:
  @njit
  def accuracy_k(y_true, y_pred):
  n, k = y_pred.shape
  acc = 0
  for i in range(n):
  for j in range(k):
  if y_pred[i, j] == y_true[i]:
  acc = acc + 1
  break
  return acc / n
  To evaluate the performance of the baseline, just invoke with the true labels and the
  predictions:
  accuracy_k(iid_val, pred_baseline)
  It prints 0.012, that is, only in 1.2% cases we make a successful recommendation. Looks
  like there is a lot of room for improvement!
  The next step is breaking the long array of items into separate transactions. We again can
  reuse the pointer array, which tells us where each transaction starts and where it ends:
  def pack_items(users, items_indptr, items_vals):
  n = len(items_indptr)—1
  result = []
  for i in range(n):
  start = items_indptr[i]
  end = items_indptr[i+1]
  result.append(items_vals[start:end])
  return result
  Now we can unwrap the transactions and put them into a separate dataframe:
  train_items = pack_items(indptr_train, indptr_train,
  df_train.stockcode.values)
  df_train_wrap = pd.DataFrame()
  Building a TensorFlow Recommender System Chapter 9
  [ 241 ]
  df_train_wrap['customerid'] = uid_train
  df_train_wrap['items'] = train_items
  To have a look at what we have at the end, use the head function:
  df_train_wrap.head()
  This shows the following:
  These sequences have varying lengths, and this is a problem for RNNs. So, we need to
  convert them into fixed-length sequences, which we can easily feed to the model later.
  In case the original sequence is too short, we need to pad it with zeros. If the sequence is too
  long, we need to cut it or split it into multiple sequences.
  Lastly, we also need to represent the state when the user has entered the website but has not
  bought anything yet. We can do this by inserting the dummy zero item—an item with index
  0, which we reserved for special purposes, just like this one. In addition to that, we can also
  use this dummy item to pad the sequences that are too small.
  We also need to prepare the labels for the RNN. Suppose we have the following sequence:
  We want to produce a sequence of fixed length 5. With padding in the beginning, the
  sequence we use for training will look as follows:
  Here we pad the original sequence with zero at the beginning and do not include the last
  element—the last element will only be included in the target sequence. So the target
  sequence—the output we want to predict—should look as follows:
  Building a TensorFlow Recommender System Chapter 9
  [ 242 ]
  It may look confusing at the beginning, but the idea is simple. We want to construct the
  sequences in such a way that for the position i in X, the position i in Y contains the element
  we want to predict. For the preceding example we want to learn the following rules:
- both are at the position 0 in X and Y
  —both are at the position 1 in X and Y
  and so on
  Now imagine we have a smaller sequence of length 2, which we need to pad to a sequence
  of length 5:
  In this case, we again pad the input sequence with 0 in the beginning, and also with some
  zeros at the end:
  .
  We transform the target sequence Y similarly:
  .
  If the input is too long, for example , we can cut it into multiple
  sequences:
  To perform such a transformation, we write a function pad_seq. It adds the needed
  amount of zeros at the beginning and at the end of the sequence. Then we pad_seq in
  another function - prepare_training_data—the function that creates the matrices X and
  Y for each sequence:
  def pad_seq(data, num_steps):
  data = np.pad(data, pad_width=(1, 0), mode='constant')
  Building a TensorFlow Recommender System Chapter 9
  [ 243 ]
  n = len(data)
  if n <= num_steps:
  pad_right = num_steps—n + 1
  data = np.pad(data, pad_width=(0, pad_right), mode='constant')
  return data
  def prepare_train_data(data, num_steps):
  data = pad_seq(data, num_steps)
  X = []
  Y = []
  for i in range(num_steps, len(data)):
  start = i—num_steps
  X.append(data[start:i])
  Y.append(data[start+1:i+1])
  return X, Y
  What is left to do is invoking the prepare_training_data function for each sequence in
  the training history, and then put the results together in X_train and Y_train matrices:
  train_items = df_train_wrap['items']
  X_train = []
  Y_train = []
  for i in range(len(train_items)):
  X, Y = prepare_train_data(train_items[i], 5)
  X_train.extend(X)
  Y_train.extend(Y)
  X_train = np.array(X_train, dtype='int32')
  Y_train = np.array(Y_train, dtype='int32')
  At this point, we have finished data preparation. Now we are ready to finally create an
  RNN model that can process this data.
  Building a TensorFlow Recommender System Chapter 9
  [ 244 ]
  RNN recommender system in TensorFlow
  The data preparation is done and now we take the produced matrices X_train and
  Y_train and use them for training a model. But of course, we need to create the model
  first. In this chapter, we will use a recurrent neural network with LSTM cells (Long Short-
  Term Memory). LSTM cells are better than plain RNN cells because they can capture longterm
  dependencies better.
  A great resource to learn more about LSTMs is the blog post
  "Understanding LSTM Networks" by Christopher Olah, which is available
  at https:/ / colah. github. io/ posts/ 2015- 08- Understanding- LSTMs/ . In
  this chapter, we do not go into theoretical details about how LSTM and
  RNN work and only look at using them in TensorFow.
  Let us start with defining a special configuration class that holds all the important training
  parameters:
  class Config:
  num_steps = 5
  num_items = item_enc.vocab_size()
  num_users = user_enc.vocab_size()
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  hidden_size = 200
  embedding_size = 200
  batch_size = 20
  config = Config()
  Here the Config class defines the following parameters:
  num_steps—This is the size of the fixed-length sequences
  num_items—The number of items in our training data (+1 for the dummy 0 item)
  num_users—The number of users (again +1 for the dummy 0 user)
  init_scale—Scale of the weights parameters, needed for the initialization
  learning_rate—The rate at which we update the weights
  max_grad_norm—The maximally allowed norm of the gradient, if the gradient
  exceeds this value, we clip it
  num_layers—The number of LSTM layers in the network
  Building a TensorFlow Recommender System Chapter 9
  [ 245 ]
  hidden_size—The size of the hidden dense layer that converts the output of
  LSTM to output probabilities
  embedding_size—The dimensionality of the item embeddings
  batch_size—The number of sequences we feed into the net in a single training
  step
  Now we finally implement the model. We start off by defining two useful helper
  functions—we will use them for adding the RNN part to our model:
  def lstm_cell(hidden_size, is_training):
  return rnn.BasicLSTMCell(hidden_size, forget_bias=0.0,
  state_is_tuple=True, reuse=not is_training)
  def rnn_model(inputs, hidden_size, num_layers, batch_size, num_steps,
  is_training):
  cells = [lstm_cell(hidden_size, is_training) for
  i in range(num_layers)]
  cell = rnn.MultiRNNCell(cells, state_is_tuple=True)
  initial_state = cell.zero_state(batch_size, tf.float32)
  inputs = tf.unstack(inputs, num=num_steps, axis=1)
  outputs, final_state = rnn.static_rnn(cell, inputs,
  initial_state=initial_state)
  output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])
  return output, initial_state, final_state
  Now we can use the rnn_model function to create our model:
  def model(config, is_training):
  batch_size = config.batch_size
  num_steps = config.num_steps
  embedding_size = config.embedding_size
  hidden_size = config.hidden_size
  num_items = config.num_items
  place_x = tf.placeholder(shape=[batch_size, num_steps], dtype=tf.int32)
  place_y = tf.placeholder(shape=[batch_size, num_steps], dtype=tf.int32)
  embedding = tf.get_variable("items", [num_items, embedding_size],
  dtype=tf.float32)
  inputs = tf.nn.embedding_lookup(embedding, place_x)
  output, initial_state, final_state = \
  rnn_model(inputs, hidden_size, config.num_layers, batch_size,
  num_steps, is_training)
  Building a TensorFlow Recommender System Chapter 9
  [ 246 ]
  W = tf.get_variable("W", [hidden_size, num_items], dtype=tf.float32)
  b = tf.get_variable("b", [num_items], dtype=tf.float32)
  logits = tf.nn.xw_plus_b(output, W, b)
  logits = tf.reshape(logits, [batch_size, num_steps, num_items])
  loss = tf.losses.sparse_softmax_cross_entropy(place_y, logits)
  total_loss = tf.reduce_mean(loss)
  tvars = tf.trainable_variables()
  gradient = tf.gradients(total_loss, tvars)
  clipped, _ = tf.clip_by_global_norm(gradient, config.max_grad_norm)
  optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
  global_step = tf.train.get_or_create_global_step()
  train_op = optimizer.apply_gradients(zip(clipped, tvars),
  global_step=global_step)
  out = {}
  out['place_x'] = place_x
  out['place_y'] = place_y
  out['logits'] = logits
  out['initial_state'] = initial_state
  out['final_state'] = final_state
  out['total_loss'] = total_loss
  out['train_op'] = train_op
  return out
  In this model there are multiple parts, which is described as follows:
1. First, we specify the inputs. Like previously, these are IDs, which later we convert
  to vectors by using the embeddings layer.
2. Second, we add the RNN layer followed by a dense layer. The LSTM layer learns
  the temporary patters in purchase behavior, and the dense layer converts this
  information into a probability distribution over all possible items.
3. Third, since our model is multi-class classification model, we optimize the
  categorical cross-entropy loss.
4. Finally, LSTMs are known to have problems with exploding gradients, which is
  why we perform gradient clipping when performing the optimization.
  Building a TensorFlow Recommender System Chapter 9
  [ 247 ]
  The function returns all the important variables in a dictionary—so, later on, we will be able
  to use them when training and validating the results.
  The reason this time we create a function, and not just global variables like previously, is to
  be able to change the parameters between training and testing phases. During training, the
  batch_size and num_steps variables could take any value, and, in fact, they are tunable
  parameters of the model. On the contrary, during testing, these parameters could take only
  one possible value: 1. The reason is that when the user buys something, it is always one
  item at a time, and not several, so num_steps is one. The batch_size is also one for the
  same reason.
  For this reason, we create two configs: one for training, and one for validation:
  config = Config()
  config_val = Config()
  config_val.batch_size = 1
  config_val.num_steps = 1
  Now let us define the computational graph for the model. Since we want to learn the
  parameters during training, but then use them in a separate model with different
  parameters during testing, we need to make the learned parameters shareable. These
  parameters include embeddings, LSTM, and the weights of the dense layer. To make both
  models share the parameters, we use a variable scope with reuse=True:
  graph = tf.Graph()
  graph.seed = 1
  with graph.as_default():
  initializer = tf.random_uniform_initializer(-config.init_scale,
  config.init_scale)
  with tf.name_scope("Train"):
  with tf.variable_scope("Model", reuse=None,
  initializer=initializer):
  train_model = model(config, is_training=True)
  with tf.name_scope("Valid"):
  with tf.variable_scope("Model", reuse=True,
  initializer=initializer):
  val_model = model(config_val, is_training=False)
  init = tf.global_variables_initializer()
  Building a TensorFlow Recommender System Chapter 9
  [ 248 ]
  The graph is ready. Now we can train the model, and for this purpose, we create
  a run_epoch helper function:
  def run_epoch(session, model, X, Y, batch_size):
  fetches = {
  "total_loss": model['total_loss'],
  "final_state": model['final_state'],
  "eval_op": model['train_op']
  }
  num_steps = X.shape[1]
  all_idx = np.arange(X.shape[0])
  np.random.shuffle(all_idx)
  batches = prepare_batches(all_idx, batch_size)
  initial_state = session.run(model['initial_state'])
  current_state = initial_state
  progress = tqdm(total=len(batches))
  for idx in batches:
  if len(idx) < batch_size:
  continue
  feed_dict = {}
  for i, (c, h) in enumerate(model['initial_state']):
  feed_dict[c] = current_state[i].c
  feed_dict[h] = current_state[i].h
  feed_dict[model['place_x']] = X[idx]
  feed_dict[model['place_y']] = Y[idx]
  vals = session.run(fetches, feed_dict)
  loss = vals["total_loss"]
  current_state = vals["final_state"]
  progress.update(1)
  progress.set_description('%.3f' % loss)
  progress.close()
  The initial part of the function should already be familiar to us: it first creates a dictionary of
  variables that we are interested to get from the model and also shuffle the dataset.
  Building a TensorFlow Recommender System Chapter 9
  [ 249 ]
  The next part is different though: since this time we have an RNN model (LSTM cell, to be
  exact), we need to keep its state across runs. To do it we first get the initial state—which
  should be all zeros—and then make sure the model gets exactly these values. After each
  step, we record the final step of the LSTM and re-enter it to the model. This way the model
  can learn typical behavior patterns.
  Again, like previously, we use tqdm to monitor progress, and we display both how many
  steps we have already taken during the epoch and the current training loss.
  Let us train this model for one epoch:
  session = tf.Session(config=None, graph=graph)
  session.run(init)
  np.random.seed(0)
  run_epoch(session, train_model, X_train, Y_train,
  batch_size=config.batch_size)
  One epoch is enough for the model to learn some patterns, so now we can see whether it
  was actually able to do it. For that we first write another helper function, which will
  emulate our recommendation scenario:
  def generate_prediction(uid, indptr, items, model, k):
  n_groups = len(uid)
  n_items = len(items)
  pred_all = np.zeros((n_items, k), dtype=np.int32)
  initial_state = session.run(model['initial_state'])
  fetches = {
  "logits": model['logits'],
  "final_state": model['final_state'],
  }
  for g in tqdm(range(n_groups)):
  start = indptr[g]
  end = indptr[g+1]
  current_state = initial_state
  feed_dict = {}
  for i, (c, h) in enumerate(model['initial_state']):
  feed_dict[c] = current_state[i].c
  feed_dict[h] = current_state[i].h
  Building a TensorFlow Recommender System Chapter 9
  [ 250 ]
  prev = np.array([[0]], dtype=np.int32)
  for i in range(start, end):
  feed_dict[model['place_x']] = prev
  actual = items[i]
  prev[0, 0] = actual
  values = session.run(fetches, feed_dict)
  current_state = values["final_state"]
  logits = values['logits'].reshape(-1)
  pred = np.argpartition(-logits, k)[:k]
  pred_all[i] = pred
  return pred_all
  What we do here is the following:
5. First, we initialize the prediction matrix, its size like in the baseline, is the number
  of items in the validation set times the number of recommendations.
6. Then we run the model for each transaction in the dataset.
7. Each time we start with the dummy zero item and the empty zero LSTM state.
8. Then one by one we predict the next possible item and put the actual item the
  user bought as the previous item—which we will feed into the model on the next
  step.
9. Finally, we take the output of the dense layer and get top-k most likely
  predictions as our recommendation for this particular step.
  Let us execute this function and look at its performance:
  pred_lstm = generate_prediction(uid_val, indptr_val, iid_val, val_model,
  k=5)
  accuracy_k(iid_val, pred_lstm)
  We see the output 7.1%, which is seven times better than the baseline.
  This is a very basic model, and there is definitely a lot of room for improvement: we can
  tune the learning rate and train for a few more epochs with gradually decreasing learning
  rate. We can change the batch_size, num_steps, as well as all other parameters. We also
  do not use any regularization—neither weight decay nor dropout. Adding it should be
  helpful.
  Building a TensorFlow Recommender System Chapter 9
  [ 251 ]
  But most importantly, we did not use any user information here: the recommendations
  were based solely on the patterns of items. We should be able to get additional
  improvement by including the user context. After all, the recommender systems should be
  personalized, that is, tailored for a particular user.
  Right now our X_train matrix contains only items. We should include another input, for
  example U_train, which contains the user IDs:
  X_train = []
  U_train = []
  Y_train = []
  for t in df_train_wrap.itertuples():
  X, Y = prepare_train_data(t.items, config.num_steps)
  U_train.extend([t.customerid] * len(X))
  X_train.extend(X)
  Y_train.extend(Y)
  X_train = np.array(X_train, dtype='int32')
  Y_train = np.array(Y_train, dtype='int32')
  U_train = np.array(U_train, dtype='int32')
  Let us change the model now. The easiest way to incorporate user features is to stack
  together user vectors with item vectors and put the stacked matrix to LSTM. It is quite easy
  to implement, we just need to modify a few lines of the code:
  def user_model(config, is_training):
  batch_size = config.batch_size
  num_steps = config.num_steps
  embedding_size = config.embedding_size
  hidden_size = config.hidden_size
  num_items = config.num_items
  num_users = config.num_users
  place_x = tf.placeholder(shape=[batch_size, num_steps], dtype=tf.int32)
  place_u = tf.placeholder(shape=[batch_size, 1], dtype=tf.int32)
  place_y = tf.placeholder(shape=[batch_size, num_steps], dtype=tf.int32)
  item_embedding = tf.get_variable("items", [num_items, embedding_size],
  dtype=tf.float32)
  item_inputs = tf.nn.embedding_lookup(item_embedding, place_x)
  user_embedding = tf.get_variable("users", [num_items, embedding_size],
  dtype=tf.float32)
  u_repeat = tf.tile(place_u, [1, num_steps])
  user_inputs = tf.nn.embedding_lookup(user_embedding, u_repeat)
  Building a TensorFlow Recommender System Chapter 9
  [ 252 ]
  inputs = tf.concat([user_inputs, item_inputs], axis=2)
  output, initial_state, final_state = \
  rnn_model(inputs, hidden_size, config.num_layers, batch_size,
  num_steps, is_training)
  W = tf.get_variable("W", [hidden_size, num_items], dtype=tf.float32)
  b = tf.get_variable("b", [num_items], dtype=tf.float32)
  logits = tf.nn.xw_plus_b(output, W, b)
  logits = tf.reshape(logits, [batch_size, num_steps, num_items])
  loss = tf.losses.sparse_softmax_cross_entropy(place_y, logits)
  total_loss = tf.reduce_mean(loss)
  tvars = tf.trainable_variables()
  gradient = tf.gradients(total_loss, tvars)
  clipped, _ = tf.clip_by_global_norm(gradient, config.max_grad_norm)
  optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
  global_step = tf.train.get_or_create_global_step()
  train_op = optimizer.apply_gradients(zip(clipped, tvars),
  global_step=global_step)
  out = {}
  out['place_x'] = place_x
  out['place_u'] = place_u
  out['place_y'] = place_y
  out['logits'] = logits
  out['initial_state'] = initial_state
  out['final_state'] = final_state
  out['total_loss'] = total_loss
  out['train_op'] = train_op
  return out
  Building a TensorFlow Recommender System Chapter 9
  [ 253 ]
  The changes between the new implementation and the previous model are shown in bold.
  In particular, the differences are the following:
  We add place_u—The placeholder that takes the user ID as input
  Rename embeddings to item_embeddings—not to confuse them with
  user_embeddings, which we added a few lines after that
  Finally, we concatenate user features with item features
  The rest of the model code stays unchanged!
  Initialization is similar to the previous model:
  graph = tf.Graph()
  graph.seed = 1
  with graph.as_default():
  initializer = tf.random_uniform_initializer(-config.init_scale,
  config.init_scale)
  with tf.name_scope("Train"):
  with tf.variable_scope("Model", reuse=None,
  initializer=initializer):
  train_model = user_model(config, is_training=True)
  with tf.name_scope("Valid"):
  with tf.variable_scope("Model", reuse=True,
  initializer=initializer):
  val_model = user_model(config_val, is_training=False)
  init = tf.global_variables_initializer()
  session = tf.Session(config=None, graph=graph)
  session.run(init)
  The only difference here is that we invoke a different function when creating the model. The
  code for training one epoch of the model is very similar to the previous one. The only things
  that we change are the extra parameters of the function, which we add into the feed_dict
  inside:
  def user_model_epoch(session, model, X, U, Y, batch_size):
  fetches = {
  "total_loss": model['total_loss'],
  "final_state": model['final_state'],
  "eval_op": model['train_op']
  }
  Building a TensorFlow Recommender System Chapter 9
  [ 254 ]
  num_steps = X.shape[1]
  all_idx = np.arange(X.shape[0])
  np.random.shuffle(all_idx)
  batches = prepare_batches(all_idx, batch_size)
  initial_state = session.run(model['initial_state'])
  current_state = initial_state
  progress = tqdm(total=len(batches))
  for idx in batches:
  if len(idx) < batch_size:
  continue
  feed_dict = {}
  for i, (c, h) in enumerate(model['initial_state']):
  feed_dict[c] = current_state[i].c
  feed_dict[h] = current_state[i].h
  feed_dict[model['place_x']] = X[idx]
  feed_dict[model['place_y']] = Y[idx]
  feed_dict[model['place_u']] = U[idx].reshape(-1, 1)
  vals = session.run(fetches, feed_dict)
  loss = vals["total_loss"]
  current_state = vals["final_state"]
  progress.update(1)
  progress.set_description('%.3f' % loss)
  progress.close()
  Let us train this new model for one epoch:
  session = tf.Session(config=None, graph=graph)
  session.run(init)
  np.random.seed(0)
  user_model_epoch(session, train_model, X_train, U_train, Y_train,
  batch_size=config.batch_size)
  The way we use the model is also almost the same as previous:
  def generate_prediction_user_model(uid, indptr, items, model, k):
  n_groups = len(uid)
  n_items = len(items)
  pred_all = np.zeros((n_items, k), dtype=np.int32)
  initial_state = session.run(model['initial_state'])
  Building a TensorFlow Recommender System Chapter 9
  [ 255 ]
  fetches = {
  "logits": model['logits'],
  "final_state": model['final_state'],
  }
  for g in tqdm(range(n_groups)):
  start = indptr[g]
  end = indptr[g+1]
  u = uid[g]
  current_state = initial_state
  feed_dict = {}
  feed_dict[model['place_u']] = np.array([[u]], dtype=np.int32)
  for i, (c, h) in enumerate(model['initial_state']):
  feed_dict[c] = current_state[i].c
  feed_dict[h] = current_state[i].h
  prev = np.array([[0]], dtype=np.int32)
  for i in range(start, end):
  feed_dict[model['place_x']] = prev
  actual = items[i]
  prev[0, 0] = actual
  values = session.run(fetches, feed_dict)
  current_state = values["final_state"]
  logits = values['logits'].reshape(-1)
  pred = np.argpartition(-logits, k)[:k]
  pred_all[i] = pred
  return pred_all
  Finally, we run this function to generate the predictions for the validation set, and calculate
  the accuracy of these recommendations:
  pred_lstm = generate_prediction_user_model(uid_val, indptr_val, iid_val,
  val_model, k=5)
  accuracy_k(iid_val, pred_lstm)
  The output we see is 0.252, which is 25%. We naturally expect it to be better, but the
  improvement was quite drastic: almost four times better than the previous model, and 25
  better than the naive baseline. Here we skip the model check on the hold-out test set, but
  you can (and generally should) do it yourself to make sure the model does not overfit.
  Building a TensorFlow Recommender System Chapter 9
  [ 256 ]
  Summary
  In this chapter, we covered recommender systems. We first looked at some background
  theory, implemented simple methods with TensorFlow, and then discussed some
  improvements such as the application of BPR-Opt to recommendations. These models are
  important to know and very useful to have when implementing the actual recommender
  systems.
  In the second section, we tried to apply the novel techniques for building recommender
  systems based on Recurrent Neural Nets and LSTMs. We looked at the user's purchase
  history as a sequence and were able to use sequence models to make successful
  recommendations.
  In the next chapter, we will cover Reinforcement Learning. This is one of the areas where
  the recent advances of Deep Learning have significantly changed the state-of-the-art: the
  models now are able to beat humans in many games. We will look at the advanced models
  that caused the change and we will also learn how to use TensorFlow to implement real AI.