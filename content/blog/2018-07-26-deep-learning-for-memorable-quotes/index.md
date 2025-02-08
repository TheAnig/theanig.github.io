+++
title = "Deep Learning Methods for Quotable Text"
description = " The goal of this study is to find out what makes a given sentence more memorable and quotable than others. Traditional methods of linguistic analysis have been done historically and they’ve had limited sucess. The aim of this project is to find out whether a machine learning solely on its own, is able to learn more effectively. Part of Documentation for Social Media and Text Analysis project under Prof.  Wei Xu at The Ohio State University"
date = "2018-07-26"
author = "Anirudh Ganesh"
[taxonomies]
tags = [
	"memorable-quotes-project", 
	"deep-learning", 
	"text-classification",
	"natural-language-processing"
]
[extra]
toc = true
katex = true
+++

## Abstract
> The goal of this study is to find out what makes a given sentence more memorable and quotable
> than others. Traditional methods of linguistic analysis have been done historically and they’ve had
> limited sucess. The aim of this study is to find out whether a machine learning solely on its own, is
> able to learn more effectively.
<!--more-->
## Introduction
It is widely known fact that in literature there are some quotes that are more memorable than others. In order to capture this essence of what makes the quote memorable, techniques have been devised based on linguistic expertise to hand pick features that is believed to contribute to making the quote memorable. The limitation of this approach is not only the fact that these generated features scale poorly from dataset to dataset, their performance is also upper-bounded by the quality of the features that are used.

In this project, I try to overcome some of these limitations by applying a supervised machine learning technique that ''learns'' what makes the quotes memorable.

## Literature Review

### Text Aesthetics 2014 

<cite>“Automatic prediction of text aesthetics and interestingness” D Ganguly et al (2014)</cite>

- Used statistical features (discussed below)
- Kindle dataset similar to the one I’ve acquired
- 56% accuracy
- A comparative study on Naive Bayes, one class SVM, binary SVM
- One class performed best
- Used the “mapping convergence” algorithm
- They apply SVDD to reduce impact of outliers
- Rely on average length of word, positional difference, POS features include adjectives and adverbs, sentiword net distance, and semantic distance.
- Used a distance formula extensively, no idea why, only appears in this paper (probably something to do with MC SVM?)

$$F(p) = \frac{2}{N*(N-1)}*\sum_{i=1}^N\sum_{j=i+1}^N\frac{d(j) - d(i)}{j-i}$$

### Bendersky 2012 

<cite> “A Dictionary of Wisdom and Wit: Learning to Extract Quotable Phrases” Bendersky et al (2012) </cite>

- ( Most closely resembles the idea I had in mind)
- Establishes features used in subsequent works by other people
- Basic pipeline involves,
- Take a book, feed sentences to Naive Bayes filter that uses log likelihood to reject most of the “unquotable” text.
- The naive Bayes filter is trained on external sets of quotes independent of the quotes from the book
- The output set of sentences from the NB filter is then fed into a classifier.
- Used a quotable language model involving selected lexical, POS and punctuation features.
- The model is a unigram  model with sentence likelihood
- Paper strongly focuses on what makes quotable phrase quotable
- Naive Bayes filter
	- Reduces noise (around 60% of text fed to it from a book is rejected)
	- High recall is important as that's the only thing used
	- Measured by testing against a foreign set of quotable phrases to see how many pass
- Quotable Phrase detection (classifier)
	- Pass a set of sentences from NB filter to single perceptron layer
	- Its labelled dataset and is trained using features mentioned above
	- They use “quotable” classified sentences and post them on /r/Quotes subreddit then average the score to see how well their classifier performs

### Memorability 2012

<cite>“You had me at hello: How phrasing affects memorability” Cristian Danescu-Niculescu-Mizil et al (2012).</cite>

- Used a corpus of movie quotes
- Also used a corpus of common language (news article) “Brown corpus”.
- Use two sets of features: generality and distinctiveness.
- Distinctiveness is calculated using likelihood w.r.t the Brown corpus.
- Generality is calculated using personal pronouns, indefinite articles, tense etc.
- Performed a comparison of %age of memorable and non-memorable quotes that contain the above features
- Use 1,2,3-gram Language Models and 1,2,3-gram with POS tagged Language models
- Of each quote pair memorable is found 60% of the time to be distinct using their distinctiveness measure
- Also included slogans from advertising and ad campaigns
- Used distinctiveness metric and generalisability metric
- Used SVM to get 60% accuracy, concluded that there is something memorable about these quotes

### Koto 2014
<cite> “Memorable spoken quote corpora of TED public speaking” Koto et al (2014) </cite>

- They crossed the domains of quotability/memorability of text with speech processing
- Used TED talk dataset
- Popularity of quote is estimated by the amount of “shares” it has on the website
- Concluded that there is something that influences memorability of text using F0 metric from INTERSPEECH ‘09

### Koto 2015 

<cite> “A Study On Natural Expressive Speech: Automatic Memorable Spoken Quote Detection” Koto et al (2015) </cite>

- Continuation of 2014 work, added features from Bendersky 2012 to extract quotability
- Introduced two new POS features in quote detection
- Dropped punctuation features as the data is speech based, added some acoustic features to improve speech processing
- Compared Naive Bayes, Neural Networks and SVMs
- Used Bendersky 2012’s features on the TED talk data set as baseline performance for the 3 methods
- Naive Bayes gets them highest performance,
- The feature set gives them 70.4% accuracy over 60% of baseline

### Koto 2016
<cite> “Automatic Detection of Memorable Spoken Quotes” Koto et al (2016) </cite>
- Extension of 2015 work, performed some analysis on best features to haveselect.


## Method
At its core, the given task can be formulated as a text classification problem. As such, I plan to employ the traditional methods for text classification and then tweak them for the task at hand. From current literature available literature that compares the performance of these techniques on the standard IMDb dataset, I have chosen TextCNN (using the GloVe embeddings) as focal technique that will be applied to the problem to see how it compares to the existing techniques discussed above.

TextCNN model was chosen because of its flexibility that allows for a lot more granular tweaking which can help increase performance for our given task. 

The test hypothesis for this project is to check whether this model is able to match the performance of the traditional techniques in their respective data and see how it performs.

Traditionally all the papers listed above in the literature review employ a balanced dataset, however this is not indicative of the real-world scenario where quotable text is much rarer than non-quotable text. Hence I have decided to improve on this by testing various amounts of imbalanced datasets (imbalanced in favour of non-quotable text)

Also since the aim is to create a generalizable model that can be applied to a wide collection of tasks I start with balanced dataset and then see the impact of creating imbalance.

I test a 50-50 split, a 60-40 split, 70-30 split, 80-20 split and 90-10 split on the generated dataset.

### TextCNN

The TextCNN model hyperparameters were tweaked according to Zhang's paper [<cite>"A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification" Zhang et al.</cite>] , with GloVe 6B, 300dim vectors.

Filter Size of 3, 4, 5 was used as it unconditionally yields the best result as shown in the paper. A batch size of 50 was used and a dropout of 0.5 was added to prevent overfitting. The model that we are using here is also double channeled.

## Experiments

### Data
The data used in the experiments are as follows,

- LitQuotes - around 2300 book quotes were scraped off of LitQuotes.com and paired up with non-quotes from same books available via Project Gutenberg. 
- Quotationspage.com - 5100 general quotes which are paired with random non-quote newspaper dataset.
- Data from Memorability 2012's paper - 2197 one-sentence memorable quotes paired with surrounding non-memorable quotes from the same movie, spoken by the same character and containing the same number of words.

This gets us around 9,000 pairs of quotes and non-quotes.

### Metrics

To measure the sucess of the system, I used F-1 score this was important as due to the skewing of the dataset, accuracy would be a bad indicator of performance as in a 90-10 split, a dumb classifier that predicts one label all the time would also yields a 90% accuracy which doesn't accurately represent the fact that the classifier hasn't learnt any useful feature from our dataset.

### Baselines

The baseline we take is 70.4% which is the best that has been achieved on a similar dataset. Though I have augmented that data set with a substantial amounts of quotes, the paper only has 2000 or so memorable quotes, while in this project I have augmented that to around 9000. This is solely because 2000 quotes is very little for Deep Learning based methods to learn anything meaningful from.

### Performance and Results of the method

- Balanced (50-50) Dataset
	- F-Score :  0.9417280643
- 60-40 Dataset
	- F-Score :  0.914163090129
- 70-30 Dataset
	- F-Score :  0.898954703833
- 80-20 Dataset
	- F-Score :  0.892349356804
- 90-10 Dataset
	- F-Score :  0.863253402463

This is a substaintial increase from the baseline, thus proving what I set out to, i.e., deep learning based methods are extremely accurate in discerning the quotability measure of a given sentence. These results support the hypotheis hence, in Future Work, I would like to see what are the additional features that the Deep Learning based methods generate that are able to capture the essence of quotability more accurately than the domain experts.

Note: Since the data for quotable text was limited, during the various higher-order splits, I just increased the amount of non-quotable text to dilute the quotable text data. So a 90-10 split has 74,000 total of quotable and non-quotable text.

## Conclusions

Though one of the main contributing factors that contributed to me choosing TextCNN was its speed and relative ease of implementation, after the experiments it is evident that the CNN is able to perform much better than human generated features for the same classifcation task. As to exactly what more information is a CNN able to extract that the linguists might have overlooked is uncertain, we can make some speculations to conclude what exactly is going on behind the scenes that justifies this performance bump. One of the biggest contributing factor that I would assume would be the GloVe word embeddings, that have a very interesting property of capturing word's meaning numerically. This leads to the CNN understanding the implications of words in a quotable vs a non-quotable text, this combined with the ability of CNNs to auto-generate abstract representations of certain advanced grammatical structures, must have contributed in its ability to yield better results. <cite> "Modeling Interestingness with Deep Neural Networks" Gao, J., Pantel, P., Gamon, M., He, X., & Deng, L. (2014) </cite>


## Replicability

In the interest of replicability of the project I will be putting up all the relavant code up on a repository [^1]



{% alert(note=true) %}
This article was originally published on Dec 5, 2017 8:43 AM.
{%end%}

[^1]:Link to the repository: [memorable-quotes](https://github.com/TheAnig/memorable-quotes)