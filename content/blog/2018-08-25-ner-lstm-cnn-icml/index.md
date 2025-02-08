+++
title = "End-to-end Sequence-Labeling via Bi-directional LSTM CNNs CRF Tutorial"
description = "This is a PyTorch tutorial for the ACL'16 paper End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF, presented at ICML 2018"
categories =  [
	"natural-language-processing"
]
date = "2018-08-25"
author = "Anirudh Ganesh"
[taxonomies]
tags = [
	"ner-icml-project",
	"deep-learning",
	"text-classification"
]
[extra]
toc = true
katex = true
+++

## Introduction
If you’ve ever dipped your toes into natural language processing (NLP), you’ve probably come across tasks like part-of-speech (POS) tagging and named entity recognition (NER). These are the bread and butter of understanding text—things like figuring out whether "Apple" in a sentence refers to a fruit or a tech giant. But let’s be honest, the traditional ways of handling these tasks? Not exactly user-friendly. They usually involve manually crafting features and preprocessing data for hours. It’s tedious, time-consuming, and doesn’t always translate well when you try to apply it to new types of text.

That’s where a groundbreaking paper from ACL 2016 comes in: End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF by Xuezhe Ma and Eduard Hovy[^1]. Don’t let the fancy title scare you—this paper is a game-changer. It proposes a neural network model that completely flips the script on the traditional approach. Instead of hand-picking features, it combines character-level representations (thanks to Convolutional Neural Networks, or CNNs) with word-level context (handled by Bi-directional LSTMs). The cherry on top? A Conditional Random Field (CRF) layer that ties everything together for structured predictions.

What’s so cool about this? Well, for starters, it’s an end-to-end model, which means it handles the entire process without requiring you to do all that annoying feature engineering. You just give it data, and it learns what it needs—automatically. It’s versatile too, working across a range of sequence labeling tasks.

And the results? Pretty impressive. This model hit 97.55% accuracy on POS tagging with the Penn Treebank WSJ dataset and scored a solid 91.21% F1 on NER using the CoNLL 2003 dataset. Those are state-of-the-art numbers, by the way.

Since its debut, this paper has left a big mark on the NLP community. It’s inspired a wave of follow-up research and implementations, all riffing on this idea of combining different neural network components into a single powerful system. This emphasis on end-to-end learning has made sequence labeling faster, easier, and way more adaptable.

## Reproducibility
This is why I took this paper as a good candidate to present for reproducibility[^2].


It’s a standout example of a complex yet impactful model, combining multiple neural network components into an end-to-end framework. The paper doesn’t just offer strong theoretical contributions; it also provides practical results that have influenced a significant portion of subsequent NLP research. By breaking down the BiLSTM-CNN-CRF model step-by-step, I wanted to demonstrate how a state-of-the-art system could be reproduced with modern tools like PyTorch, ensuring others could verify the results and, ideally, build upon them.

Reproducibility in this case meant more than just rerunning the experiments—it meant making the implementation accessible, understandable, and adaptable to different use cases. During my preparation, I paid close attention to the potential pain points: ensuring the code was platform-agnostic, tuning hyperparameters to match those used in the paper, and structuring the codebase so others could easily tweak it for related tasks. The aim was to not only replicate the model’s impressive benchmark performance but to encourage further experimentation and adoption by the community.

At the ICML workshop, this example opened up broader conversations about what it takes to make machine learning research truly reproducible. The discussion often circled back to the tools we use to share code and results. For instance, while frameworks like PyTorch offer flexibility and power, simply uploading a GitHub repository isn’t enough if it lacks proper documentation or doesn’t account for different computational setups. The group also explored the importance of including detailed training procedures and hyperparameter configurations—often overlooked but critical for achieving comparable results.

## Tutorial

{% alert(note=true) %}
Feel free to check out the linked repository to follow along with this article to implement your own state of the art NER pipeline
{%end%}

### Data Preparation
Before we dive into the model, let’s prepare the data. We’ll use the CoNLL 2003 dataset, which contains text tagged for four types of named entities: PERSON, LOCATION, ORGANIZATION, and MISC.

The dataset uses the BIO tagging scheme, where:

* B-TYPE indicates the beginning of an entity of type TYPE.
* I-TYPE indicates the continuation of the entity.
* O indicates that the word is not part of any entity.

For example,
```mathematica
U.N.         NNP  I-NP  I-ORG
official     NN   I-NP  O
Ekeus        NNP  I-NP  I-PER
heads        VBZ  I-VP  O
for          IN   I-PP  O
Baghdad      NNP  I-NP  I-LOC
```

We also preprocess the text by replacing all digits with 0. This step ensures that the model focuses on meaningful textual patterns instead of numeric details, which are often irrelevant in NER tasks.

```python
def zero_digits(s):
    return re.sub('\\d', '0', s)

def load_sentences(path, zeros):
    sentences = []
    sentence = []
    for line in open(path, 'r', encoding='utf-8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    return sentences
```

To align with the paper, we convert the tags to the BIOES scheme, which provides finer granularity by adding:

* E-TYPE for the end of an entity.
* S-TYPE for single-token entities.

```python
def iob_iobes(tags):
    """
    the function is used to convert
    BIO -> BIOES tagging
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags
```

### Mappings and Word Embeddings
To train our model, we need to convert words, characters, and tags into numeric representations. This mapping allows us to leverage efficient matrix operations in PyTorch. For word embeddings, we use pre-trained GloVe vectors (100-dimensional) trained on Wikipedia and Gigaword data. These embeddings provide a semantic representation of words, which significantly boosts the model’s performance.

```python
word_to_id, id_to_word = create_mapping(train_sentences)
char_to_id, id_to_char = create_mapping(train_sentences, level='char')
tag_to_id, id_to_tag = create_mapping(train_sentences, level='tag')
```

We also load GloVe embeddings into a matrix and initialize missing embeddings randomly:

```python
word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), 100))
for word in word_to_id:
    if word in glove_embeds:
        word_embeds[word_to_id[word]] = glove_embeds[word]
```

### Model Architecture

The BiLSTM-CNN-CRF model has three main components:

* CNN for Character Embeddings: Generates a character-level representation for each word using convolutional and max-pooling layers. This step captures morphological patterns like prefixes and suffixes.

* BiLSTM for Word Encoding: Combines word embeddings (GloVe + character embeddings) and processes them bidirectionally. This helps capture context from both the left and right of each word.

* CRF for Structured Prediction: Ensures that predictions respect tagging constraints (e.g., I-ORG cannot follow I-PER).

The model that we are presenting is a complicated one, since its a hybridized network using LSTMs and CNNs. So in order to break down the complexity, we have attempted to simplify the process by splitting up operations into individual functions that we can go over part by part. This hopefully makes the whole thing more easily digestable and gives a more intuitive understanding of the whole process.

#### Initialization of weights
We start with the init_embedding function, which just initializes the embedding layer by pooling from a random sample.

The distribution is pooled from $-\sqrt{\frac{3}{V}}$ to $+\sqrt{\frac{3}{V}}$ where $V$ is the embedding dimension size.
```python
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform(input_embedding, -bias, bias)
```

The LSTM layers are initialized by uniform sampling from $-\sqrt{\frac{6}{r + c}}$ to $+\sqrt{\frac{6}{r + c}}$. Where $r$ is the number of rows, $c$ is the number of columns (based on the shape of the weight matrix).
```python
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform(input_linear.weight, -bias, bias)
```

#### CRF Layer
We have two options:

* **softmax**: normalize the scores into a vector such that can be interpreted as the probability that the word belongs to class. Eventually, the probability of a sequence of tag $y$ is the product of all tags.

* **linear-chain CRF**: the first method makes local choices. In other words, even if we capture some information from the context thanks to the bi-LSTM, the tagging decision is still local. We don’t make use of the neighbooring tagging decisions. Given a sequence of words $w_1, ..., w_m$, a sequence of score vectors
$s_1, ..., s_m$ and a sequence of tags $y_1, ..., y_m$, a linear-chain CRF defines a global score $C \in \R$ such that

$$
C(y_1, ..., y_m) = b[y_1] + \sum_{t=1}^m s_t[y_t] + \sum_{t=1}^{m-1}T[y_t, y_{t+1}] + e[y_m]
$$

where $T$ is a transition matrix in $\R^{9x9}$ and $e, b \in \R^9$ are vectors of scores that capture the cost of beginning or ending with a given tag. The use of the matrix $T$ captures linear (one step) dependencies between tagging decisions.

The motivation behind CRFs was to generate sentence level likelihoods for optimal tags. What that means is for each word we estimate maximum likelihood and then we use the Viterbi algorithm to decode the tag sequence optimally.

**Advantages of CRF over Softmax:**

* Softmax doesn't value any dependencies, this is a problem since NER the context heavily influences the tag that is assigned. This is solved by applying CRF as it takes into account the full sequence to assign the tag.
* Example: I-ORG cannot directly follow I-PER.

<figure>
{{ image(url="simple_crf_network.svg", alt="A simple CRF network") }}
<figcaption>
A simple CRF network
</figcaption>
</figure>

The figure shows a simple CRF network, in our case we have the inputs feeding in from our BiLSTMs, but otherwise the structure largely remains the same.

#### Score Calculation
CRF computes a conditional probability. Let $y$ be a tag sequence and x an input of sequence of words. Then we compute
$$
P(y|x) = \frac{exp(Score(x, y))}{\sum_{y'}exp(Score(x, y'))}
$$

Where the score is determined by defining some log potentials $log\psi_i(x, y)$ such that
$$
Score(x, y) = \sum_i log\psi_i(x, y)
$$

```python
def log_sum_exp(vec):
    '''
    This function calculates the score explained above for the forward algorithm
    vec 2D: 1 * tagset_size
    '''
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
```

This is the score function for our sentences. This function takes a list of ground truths that tell us what the corresponding tags are and the features which contains the supposed tagged parts of the function. Which is then used to compute the score.

```python
def score_sentences(self, feats, tags):
    # tags is ground_truth, a list of ints, length is len(sentence)
    # feats is a 2D tensor, len(sentence) * tagset_size
    r = torch.LongTensor(range(feats.size()[0]))
    if self.use_gpu:
        r = r.cuda()
        pad_start_tags = torch.cat([torch.cuda.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        pad_stop_tags = torch.cat([tags, torch.cuda.LongTensor([self.tag_to_ix[STOP_TAG]])])
    else:
        pad_start_tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        pad_stop_tags = torch.cat([tags, torch.LongTensor([self.tag_to_ix[STOP_TAG]])])

    score = torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])

    return score
```

#### Viterbi decode

Viterbi decode is basically applying dynamic programming to choosing our tag sequence. Let’s suppose that we have the solution

$$ \tilde s_t(y_t) = argmax_{y_t, ..., y_m} C(y_t, ..., y_m) = argmax_{y_{t+1}}s_t[y_t] + T[y_t, y_{t+1}] + \tilde s_{t+1}(y^{t+1}) $$

Then, we can easily define the probability of a given sequence of tags as

$$
\mathbb{P}(y_1, ..., y_m) = \frac{e^{C(y_t, ..., y_m)}}{Z}
$$

```python
def forward_calc(self, sentence, chars, chars2_length, d):

    '''
    The function calls viterbi decode and generates the
    most probable sequence of tags for the sentence
    '''

    # Get the emission scores from the BiLSTM
    feats = self._get_lstm_features(sentence, chars, chars2_length, d)
    # viterbi to get tag_seq

    # Find the best path, given the features.
    if self.use_crf:
        score, tag_seq = self.viterbi_decode(feats)
    else:
        score, tag_seq = torch.max(feats, 1)
        tag_seq = list(tag_seq.cpu().data)

    return score, tag_seq
```

### Model Details

#### CNN model for generating character embeddings
Consider the word 'cat', we pad it on both ends to get our maximum word length ( this is mainly an implementation quirk since we can't have variable length layers at run time, our algorithm will ignore the pads).
<figure>
{{ image(url="convolution_model_details.svg", alt="Convolution Model for generating character embeddings") }}
<figcaption>
    Convolution Model for generating character embeddings
</figcaption>
</figure>
We then apply a convolution layer on top that generates spatial coherence across characters, we use a maxpool to extract meaningful features out of our convolution layer. This now gives us a dense vector representation of each word. This representation will be concatenated with the pre-trained GloVe embeddings using a simple lookup.

This snippet shows us how the CNN is implemented in pytorch
```python
self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(3, char_embedding_dim), padding=(2,0))
```

#### LSTM model that generates tags for the given sequence

The word-embeddings( glove+char embedding ) that we generated above, we feed to a bi-directional LSTM model. The LSTM model has 2 layers,

<figure>
{{ image(url="lstm_based_tag_generation.svg", alt="LSTMs for Tag Generation") }}
<figcaption>
    LSTMs for Tag Generation
</figcaption>
</figure>

* The forward layer takes in a sequence of word vectors and generates a new vector based on what it has seen so far in the forward direction (starting from the start word up until current word) this vector can be thought of as a summary of all the words it has seen.

* The backwards layer does the same but in opposite direction, i.e., from the end of the sentence to the current word.

* The forward vector and the backwards vector at current word concatanate to generate a unified representation.

This snippet shows us how the BiLSTM is implemented in pytorch

```python
self.lstm = nn.LSTM(embedding_dim+self.out_channels, hidden_dim, bidirectional=True)
```

#### Main Model Implementation

The get_lstm_features function returns the LSTM's tag vectors. The function performs all the steps mentioned above for the model.

Steps:

* It takes in characters, converts them to embeddings using our character CNN.
* We concat Character Embeeding with glove vectors, use this as features that we feed to Bidirectional-LSTM.
* The Bidirectional-LSTM generates outputs based on these set of features.
* The output are passed through a linear layer to convert to tag space

```python
if self.char_mode == 'LSTM':
    chars_embeds = self.char_embeds(chars2).transpose(0, 1)
    packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, chars2_length)
    lstm_out, _ = self.char_lstm(packed)
    outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
    outputs = outputs.transpose(0, 1)
    chars_embeds_temp = Variable(torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2)))))
    if self.use_gpu:
        chars_embeds_temp = chars_embeds_temp.cuda()
    for i, index in enumerate(output_lengths):
        chars_embeds_temp[i] = torch.cat((outputs[i, index-1, :self.char_lstm_dim], outputs[i, 0, self.char_lstm_dim:]))
    chars_embeds = chars_embeds_temp.clone()
    for i in range(chars_embeds.size(0)):
        chars_embeds[d[i]] = chars_embeds_temp[i]

if self.char_mode == 'CNN':
    chars_embeds = self.char_embeds(chars2).unsqueeze(1)
    ## Creating Character level representation using Convolutional Neural Netowrk
    ## followed by a Maxpooling Layer
    chars_cnn_out3 = self.char_cnn3(chars_embeds)
    chars_embeds = nn.functional.max_pool2d(chars_cnn_out3,
    kernel_size=(chars_cnn_out3.size(2), 1)).view(chars_cnn_out3.size(0), self.out_channels)
```

### Training the model

We train the model using stochastic gradient descent (SGD) with a learning rate of 0.015 and momentum of 0.9. To avoid overfitting, we apply dropout and use gradient clipping

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.015, momentum=0.9)
```

We also calculate the negative log-likelihood as our loss function:

```python
def neg_log_likelihood(sentence, tags):
    feats = get_lstm_features(sentence)
    forward_score = forward_algorithm(feats)
    gold_score = score_sentence(feats, tags)
    return forward_score - gold_score
```

### Evaluation and Testing

After training, we evaluate the model using precision, recall, and F1-score. Here’s an example of the model tagging new sentences:

```python
sentence = "Jay is from India."
prediction = model.predict(sentence)
print(prediction)
```

This gives us the output:
```yaml
Jay : PER
is : NA
from : NA
India : LOC
```

... and that's it! That gets us a state of the art NER pipeline!

## Closing Thoughts
Presenting this work at the workshop was an incredibly rewarding experience. It wasn’t just about showcasing the BiLSTM-CNN-CRF model or diving into the nitty-gritty of the implementation details in PyTorch, but seeing researchers and practitioners engage with this tutorial, ask thoughtful questions, and discuss their own challenges made all the preparation worth it.

If you have any thoughts, questions, or feedback, feel free to reach out—I’d love to hear from you!

{% alert(note=true) %}
The work in this article was presented at ICML 2018, MLTrain Workshop "Enabling Reproducability in Machine Learning".
{%end%}

[^1]: Link to the original paper: [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://aclanthology.org/P16-1101/)
[^2]: Link to the repository: [NER-LSTM-CNN-Pytorch](https://github.com/TheAnig/NER-LSTM-CNN-Pytorch)
