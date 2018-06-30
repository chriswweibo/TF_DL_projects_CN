## 第7章 训练像人类一样的聊天机器人 

This chapter will show you how to train an automatic chatbot that will be able to answer
simple and generic questions, and how to create an endpoint over HTTP for providing the
answers via an API. More specifically, we will show:

- What's the corpus and how to preprocess the corpus
- How to train a chatbot and how to test it
- How to create an HTTP endpoint to expose the API

### 项目简介

Chatbots are becoming increasingly used as a way to provide assistance to users. Many
companies, including banks, mobile/landline companies and large e-sellers now use
chatbots for customer assistance and for helping users in pre-sales. The Q&A page is not
enough anymore: each customer is nowadays expecting an answer to his very own question
which maybe is not covered or only partially covered in the Q&A. Also, chatbots are a great
tool for companies which don't need to provide additional customer service capacity for
trivial questions: they really look like a win-win situation!
Chatbots have become very popular tools ever since deep learning became popular. Thanks
to deep learning, we're now able to train the bot to provide better and personalized
questions, and, in the last implementation, to retain a per-user context.
Cutting it short, there are mainly two types of chatbot: the first is a simple one, which tries
to understand the topic, always providing the same answer for all questions about the same
topic. For example, on a train website, the questions Where can I find the timetable of the
City_A to City_B service? and What's the next train departing from City_A? will likely get the
same answer, that could read Hi! The timetable on our network is available on this page: <link>.

[ 160 ]
Basically, behind the scene, this types of chatbots use classification algorithms to
understand the topic (in the example, both questions are about the timetable topic). Given
the topic, they always provide the same answer. Usually, they have a list of N topics and N
answers; also, if the probability of the classified topic is low (the question is too vague, or
it's on a topic not included in the list), they usually ask the user to be more specific and
repeat the question, eventually pointing out other ways to do the question (send an email or
call the customer service number, for example).
The second type of chatbots is more advanced, smarter, but also more complex. For those,
the answers are built using an RNN, in the same way that machine translation is performed
(see the previous chapter). Those chatbots are able to provide more personalized answers,
and they may provide a more specific reply. In fact, they don't just guess the topic, but with
an RNN engine they're able to understand more about the user's questions and provide the
best possible answer: in fact, it's very unlikely you'll get the same answers with two
different questions using these types if chatbots.
In this chapter, we will try to build a chatbot of the second type using an RNN similarly to
what we've done in the previous chapter with the machine translation system. Also, we will
show how to put the chatbot behind an HTTP endpoint, in order to use the chatbot as a
service from your website, or, more simply, from your command line.

### 输入词表

Unfortunately, we haven't found any consumer-oriented dataset that is open source and
freely available on the Internet. Therefore, we will train the chatbot with a more generic
dataset, not really focused on customer service. Specifically, we will use the Cornell Movie
Dialogs Corpus, from the Cornell University. The corpus contains the collection of
conversations extracted from raw movie scripts, therefore the chatbot will be able to give
answer more to fictional questions than real ones. The Cornell corpus contains more than
200,000 conversational exchanges between 10+ thousands of movie characters, extracted
from 617 movies.

> The dataset is available here: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html。
> We would like to thank the authors for having released the corpus: that
> makes experimentation, reproducibility and knowledge sharing easier.

The dataset comes as a .zip archive file. After decompressing it, you'll find several files in
it:

- `README.txt` contains the description of the dataset, the format of the corpora files, the details on the collection procedure and the author's contact.

- `Chameleons.pdf` is the original paper for which the corpus has been released.Although the goal of the paper is strictly not around chatbots, it studies the language used in dialogues, and it's a good source of information to understanding more

- `movie_conversations.txt` contains all the dialogues structure. For each conversation, it includes the ID of the two characters involved in the discussion, the ID of the movie and the list of sentences IDs (or utterances, to be more precise) in chronological order. For example, the first line of the file is:

  `u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']`

  That means that user u0 had a conversation with user u2 in the movie m0 and the conversation had 4 utterances: 'L194', 'L195', 'L196' and 'L197'

- `movie_lines.txt` contains the actual text of each utterance ID and the person who produced it. For example, the utterance L195 is listed here as:

  `L195 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ Well, I thought we'd start with pronunciation, if that's okay with you`.

  So, the text of the utterance L195 is Well, I thought we'd start with pronunciation, if

  that's okay with you. And it was pronounced by the character u2 whose name is CAMERON in the movie m0.

- `movie_titles_metadata.txt` contains information about the movies, including the title, year, IMDB rating, the number of votes in IMDB and the genres. For example, the movie m0 here is described as:

  `m0 +++$+++ 10 things i hate about you +++$+++ 1999 +++$+++ 6.90 +++$+++ 62847 +++$+++ ['comedy', 'romance']`

  So, the title of the movie whose ID is m0 is 10 things i hate about you, it's from 1999,  it's a comedy with romance and it received almost 63 thousand votes on IMDB with an average score of 6.9 (over 10.0)

* `movie_characters_metadata.txt` contains information about the movie characters, including the name the title of the movie where he/she appears, the gender (if known) and the position in the credits (if known). For example, the character “u2” appears in this file with this description:

  `u2 +++$+++ CAMERON +++$+++ m0 +++$+++ 10 things i hate about you +++$+++ m +++$+++ 3`

  The character `u2` is named CAMERON, it appears in the movie m0 whose title is `10 things i hate about you`, his gender is male and he's the third person appearing in the credits.

- `raw_script_urls.txt` contains the source URL where the dialogues of each movie can be retrieved. For example, for the movie `m0` that's it:

  `m0 +++$+++ 10 things i hate about you +++$+++`

  `http://www.dailyscript.com/scripts/10Things.html`

  As you will have noticed, most files use the token `+++$+++` to separate the fields. Beyond that, the format looks pretty straightforward to parse. Please take particular care while parsing the files: their format is not UTF-8 but ISO-8859-1.

### 创建训练集

Let's now create the training set for the chatbot. We'd need all the conversations between
the characters in the correct order: fortunately, the corpora contains more than what we
actually need. For creating the dataset, we will start by downloading the zip archive, if it's
not already on disk. We'll then decompress the archive in a temporary folder (if you're
using Windows, that should be `C:\Temp`), and we will read just the `movie_lines.txt` and
the `movie_conversations.txt` files, the ones we really need to create a dataset of
consecutive utterances.
Let's now go step by step, creating multiple functions, one for each step, in the file
`corpora_downloader.py`. The first function we need is to retrieve the file from the
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


This function does exactly that: it checks whether the “`README.txt`” file is available locally;
if not, it downloads the file (thanks for the urlretrieve function in the urllib.request module)
and it decompresses the zip (using the zipfile module).
The next step is read the conversation file and extract the list of utterance IDS. As a
reminder, its format is: `u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']`,
therefore what we're looking for is the fourth element of the list after we split it on the
token `+++$+++` . Also, we'd need to clean up the square brackets and the apostrophes to
have a clean list of IDs. For doing that, we shall import the re module, and the function will
look like this.

```python
import re
def read_conversations(storage_path, storage_dir):
    filename = storage_path + "/" + storage_dir + "/cornell movie-dialogs corpus/movie_conversations.txt"
    with open(filename, "r", encoding="ISO-8859-1") as fh:
        conversations_chunks = [line.split(" +++$+++ ") for line in fh]
    return [re.sub('[\[\]\']', '', el[3].strip()).split(", ") for el in
            conversations_chunks]
```


As previously said, remember to read the file with the right encoding, otherwise, you'll get
an error. The output of this function is a list of lists, each of them containing the sequence of
utterance IDS in a conversation between characters. Next step is to read and parse the
movie_lines.txt file, to extract the actual utterances texts. As a reminder, the file looks
like this line:
`L195 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ Well, I thought we'd start with pronunciation, if that's okay with you`.
Here, what we're looking for are the first and the last chunks.

```python
def read_lines(storage_path, storage_dir):
    filename = storage_path + "/" + storage_dir + "/cornell movie-dialogs corpus/movie_lines.txt"
    with open(filename, "r", encoding="ISO-8859-1") as fh:
        lines_chunks = [line.split(" +++$+++ ") for line in fh]
    return {line[0]: line[-1].strip() for line in lines_chunks}
```



The very last bit is about tokenization and alignment. We'd like to have a set whose
observations have two sequential utterances. In this way, we will train the chatbot, given
the first utterance, to provide the next one. Hopefully, this will lead to a smart chatbot, able
to reply to multiple questions. Here's the function:

```python
def get_tokenized_sequencial_sentences(list_of_lines, line_text):
    for line in list_of_lines:
        for i in range(len(line) - 1):
            yield (line_text[line[i]].split(" "),
                   line_text[line[i+1]].split(" "))
```


Its output is a generator containing a tuple of the two utterances (the one on the right
follows temporally the one on the left). Also, utterances are tokenized on the space
character.
Finally, we can wrap up everything into a function, which downloads the file and unzip it
(if not cached), parse the conversations and the lines, and format the dataset as a generator.
As a default, we will store the files in the /tmp directory:

```python
def retrieve_cornell_corpora(storage_path="/tmp",
                             storage_dir="cornell_movie_dialogs_corpus"):
    download_and_decompress("http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip", storage_path, storage_dir)
    conversations = read_conversations(storage_path, storage_dir)
    lines = read_lines(storage_path, storage_dir)
    return tuple(zip(*list(get_tokenized_sequencial_sentences(conversations, 
                                                              lines))))
```


At this point, our training set looks very similar to the training set used in the translation
project, in the previous chapter. Actually, it's not just similar, it's the same format with the
same goal. We can, therefore, use some pieces of code we've developed in the previous
chapter. For example, the `corpora_tools.py` file can be used here without any change
(also, it requires the `data_utils.py`).
Given that file, we can dig more into the corpora, with a script to check the chatbot input.
To inspect the corpora, we can use the `corpora_tools.py` we made in the previous
chapter, and the file we've previously created. Let's retrieve the Cornell Movie Dialog
Corpus, format the corpora and print an example and its length:

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

This code prints an example of two tokenized consecutive utterances, and the number of
examples in the dataset, that is more than 220,000:

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

Let's now clean the punctuation in the sentences, lowercase them and limits their size to 20
words maximum (that is examples where at least one of the sentences is longer than 20
words are discarded). This is needed to standardize the tokens:
```python
clean_sen_l1 = [clean_sentence(s) for s in sen_l1]
clean_sen_l2 = [clean_sentence(s) for s in sen_l2]
filt_clean_sen_l1, filt_clean_sen_l2 = filter_sentence_length(clean_sen_l1,
                                                              clean_sen_l2)
print("# Filtered Corpora length (i.e. number of sentences)")
print(len(filt_clean_sen_l1))
assert len(filt_clean_sen_l1) == len(filt_clean_sen_l2)
```


This leads us to almost 140,000 examples:
```
# Filtered Corpora length (i.e. number of sentences)
140261
```


Then, let's create the dictionaries for the two sets of sentences. Practically, they should look
the same (since the same sentence appears once on the left side, and once in the right side)
except there might be some changes introduced by the first and last sentences of a
conversation (they appear only once). To make the best out of our corpora, let's build two
dictionaries of words and then encode all the words in the corpora with their dictionary
indexes:
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


That prints the following output. We also notice that a dictionary of 15 thousand entries
doesn't contain all the words and more than 16 thousand (less popular) of them don't fit
into it:
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

As the final step, let's add paddings and markings to the sentences:
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


And that, as expected, prints:
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

After we're done with the corpora, it's now time to work on the model. This project requires
again a sequence to sequence model, therefore we can use an RNN. Even more, we can
reuse part of the code from the previous project: we'd just need to change how the dataset is
built, and the parameters of the model. We can then copy the training script built in the
previous chapter, and modify the `build_dataset` function, to use the Cornell dataset.
Mind that the dataset used in this chapter is bigger than the one used in the previous,
therefore you may need to limit the corpora to a few dozen thousand lines. On a 4 years old
laptop with 8GB RAM, we had to select only the first 30 thousand lines, otherwise, the
program ran out of memory and kept swapping. As a side effect of having fewer examples,
even the dictionaries are smaller, resulting in less than 10 thousands words each.

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


By inserting this function into the `train_translator.py` file (from the previous chapter)
and rename the file as `train_chatbot.py`, we can run the training of the chatbot.

[ 168 ]
After a few iterations, you can stop the program and you'll see something similar to this
output:

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


Again, if you change the settings, you may end up with a different perplexity. To obtain
these results, we set the RNN size to 256 and 2 layers, the batch size of 128 samples, and the
learning rate to 1.0.
At this point, the chatbot is ready to be tested. Although you can test the chatbot with the
same code as in the `test_translator.py` of the previous chapter, here we would like to
do a more elaborate solution, which allows exposing the chatbot as a service with APIs.

### 聊天机器人API

First of all, we need a web framework to expose the API. In this project, we've chosen Bottle,
a lightweight simple framework very easy to use.

> To install the package, run pip install bottle from the command
> line. To gather further information and dig into the code, take a look at the
> project webpage, https://bottlepy.org.

Let's now create a function to parse an arbitrary sentence provided by the user as an
argument. All the following code should live in the test_chatbot_aas.py file. Let's start
with some imports and the function to clean, tokenize and prepare the sentence using the
dictionary:

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


The function `prepare_sentence` does the following:

- Tokenizes the input sentence
- Cleans it (lowercase and punctuation cleanup)
- Converts tokens to dictionary IDs
- Add markers and paddings to reach the default length

Next, we will need a function to convert the predicted sequence of numbers to an actual
sentence composed of words. This is done by the function `decode`, which runs the
prediction given the input sentence and with softmax predicts the most likely output.
Finally, it returns the sentence without paddings and markers (a more exhaustive
description of the function is provided in the previous chapter):

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
Finally, the main function, that is, the function to run in the script:

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


Initially, it loads the dictionary and prepares the inverse dictionary. Then, it uses the Bottle
API to create an HTTP GET endpoint (under the /api URL). The route decorator sets and
enriches the function to run when the endpoint is contacted via HTTP GET. In this case, the
`api()` function is run, which first reads the sentence passed as HTTP parameter, then calls
the `prepare_sentence` function, described above, and finally runs the decoding step.
What's returned is a dictionary containing both the input sentence provided by the user and
the reply of the chatbot.
Finally, the webserver is turned on, on the localhost at port 8080. Isn't very easy to have a
chatbot as a service with Bottle?
It's now time to run it and check the outputs. To run it, run from the command line:

```shell
$> python3 –u test_chatbot_aas.py
```

Then, let's start querying the chatbot with some generic questions, to do so we can use
CURL, a simple command line; also all the browsers are ok, just remember that the URL
should be encoded, for example, the space character should be replaced with its encoding,
that is, %20.
Curl makes things easier, having a simple way to encode the URL request. Here are a
couple of examples:

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


> If the system doesn't work with your browser, try encoding the URL, for example:
> ```shell
> $> curl -X GET
> http://127.0.0.1:8080/api?sentence=how%20are%20you?
> {"data": [{"out": "that ' s okay .", "in": "how are you?"}]}.
> ```

Replies are quite funny; always remember that we trained the chatbox on movies, therefore
the type of replies follow that style.
To turn off the webserver, use Ctrl + C.

#### 课后作业

Following are the home assignments:
Can you create a simple webpage which queries the chatbot via JS?
Many other training sets are available on the Internet; try to see the differences of
answers between the models. Which one is the best for a customer service bot?
Can you modify the model, to be trained as a service, that is, by passing the
sentences via HTTP GET/POST?

### 小结

In this chapter, we've implemented a chatbot, able to respond to questions through an
HTTP endpoint and a GET API. It's another great example of what we can do with RNN. In
the next chapter, we're moving to a different topic: how to create a recommender system
using Tensorflow.