---
title: Deep Learning Methods for Quotable Text
layout: post
description: Documentation for Social Media and Text Analysis project under Prof.
  Wei Xu
categories: natural-language-processing
tags: memorable-quotes-project, deep-learning, text-classification
---

##### This article was originally published on Dec 5, 2017 8:43 AM.

### Abstract
> The goal of this project is to find out what makes a given sentence more memorable and quotable
> than others. Traditional methods of linguistic analysis have been done historically and they’ve had
> limited sucess. The aim of this project is to find out whether a machine learning solely on its own, is
> able to learn more effectively.

## Introduction
It is widely known fact that in literature there are some quotes that are more memorable than others. In order to capture this essence of what makes the quote memorable, techniques have been devised based on linguistic expertise to hand pick features that is believed to contribute to making the quote memorable. The limitation of this approach is not only the fact that these generated features scale poorly from dataset to dataset, their performance is also upper-bounded by the quality of the features that are used.

In this project, I try to overcome some of these limitations by applying a supervised machine learning technique that ''learns'' what makes the quotes memorable.

## Literature Review

#### Text Aesthetics 2014 

<cite>“Automatic prediction of text aesthetics and interestingness” D Ganguly et al (2014)</cite>

* Used statistical features (discussed below)
* Kindle dataset similar to the one I’ve acquired
* 56% accuracy
* A comparative study on Naive Bayes, one class SVM, binary SVM
* One class performed best
* Used the “mapping convergence” algorithm
* They apply SVDD to reduce impact of outliers
* Rely on average length of word, positional difference, POS features include adjectives and adverbs, sentiword net distance, and semantic distance.
* Used a distance formula extensively, no idea why, only appears in this paper (probably something to do with MC SVM?)

$$F(p) = \frac{2}{N*(N-1)}*\sum_{i=1}^N\sum_{j=i+1}^N\frac{d(j) - d(i)}{j-i}$$

#### Bendersky 2012 

<cite> “A Dictionary of Wisdom and Wit: Learning to Extract Quotable Phrases” Bendersky et al (2012) </cite>

* ( Most closely resembles the idea I had in mind)
* Establishes features used in subsequent works by other people
* Basic pipeline involves,
* Take a book, feed sentences to Naive Bayes filter that uses log likelihood to reject most of the “unquotable” text.
* The naive Bayes filter is trained on external sets of quotes independent of the quotes from the book
* The output set of sentences from the NB filter is then fed into a classifier.
* Used a quotable language model involving selected lexical, POS and punctuation features.
* The model is a unigram  model with sentence likelihood
* Paper strongly focuses on what makes quotable phrase quotable

* Naive Bayes filter
	* Reduces noise (around 60% of text fed to it from a book is rejected)
	* High recall is important as that's the only thing used
	* Measured by testing against a foreign set of quotable phrases to see how many pass

* Quotable Phrase detection (classifier)
	* Pass a set of sentences from NB filter to single perceptron layer
	* Its labelled dataset and is trained using features mentioned above
	* They use “quotable” classified sentences and post them on /r/Quotes subreddit then average the score to see how well their classifier performs

#### Memorability 2012

<cite>“You had me at hello: How phrasing affects memorability” Cristian Danescu-Niculescu-Mizil et al (2012).</cite>

* Used a corpus of movie quotes
* Also used a corpus of common language (news article) “Brown corpus”.
* Use two sets of features: generality and distinctiveness.
* Distinctiveness is calculated using likelihood w.r.t the Brown corpus.
* Generality is calculated using personal pronouns, indefinite articles, tense etc.
* Performed a comparison of %age of memorable and non-memorable quotes that contain the above features
* Use 1,2,3-gram Language Models and 1,2,3-gram with POS tagged Language models
* Of each quote pair memorable is found 60% of the time to be distinct using their distinctiveness measure
* Also included slogans from advertising and ad campaigns
* Used distinctiveness metric and generalisability metric
* Used SVM to get 60% accuracy, concluded that there is something memorable about these quotes

#### Koto 2014
<cite> “Memorable spoken quote corpora of TED public speaking” Koto et al (2014) </cite>

* They crossed the domains of quotability/memorability of text with speech processing
* Used TED talk dataset
* Popularity of quote is estimated by the amount of “shares” it has on the website
* Concluded that there is something that influences memorability of text using F0 metric from INTERSPEECH ‘09

#### Koto 2015 

<cite> “A Study On Natural Expressive Speech: Automatic Memorable Spoken Quote Detection” Koto et al (2015) </cite>

* Continuation of 2014 work, added features from Bendersky 2012 to extract quotability
* Introduced two new POS features in quote detection
* Dropped punctuation features as the data is speech based, added some acoustic features to improve speech processing
* Compared Naive Bayes, Neural Networks and SVMs
* Used Bendersky 2012’s features on the TED talk data set as baseline performance for the 3 methods
* Naive Bayes gets them highest performance,
* The feature set gives them 70.4% accuracy over 60% of baseline

#### Koto 2016
<cite> “Automatic Detection of Memorable Spoken Quotes” Koto et al (2016) </cite>
* Extension of 2015 work, performed some analysis on best features to haveselect.


## Method
At its core, the given task can be formulated as a text classification problem. As such, I plan to employ the traditional methods for text classification and then tweak them for the task at hand. From current literature available literature that compares the performance of these techniques on the standard IMDb dataset, I have chosen TextCNN (using the GloVe embeddings) and an LSTM classifier as focal techniques that will be applied to the problem to see how it compares to the existing techniques discussed above.

TextCNN model was chosen because of its flexibility that allows for a lot more granular tweaking which can help increase performance for our given task. LSTM Classifier was also used to give another model to compare it to.

The test hypothesis for this project is to check whether this model is able to match the performance of the traditional techniques in their respective data and see how it performs.

Traditionally all the papers listed above in the literature review employ a balanced dataset, however this is not indicative of the real-world scenario where quotable text is much rarer than non-quotable text. Hence I have decided to improve on this by testing various amounts of imbalanced datasets (imbalanced in favour of non-quotable text)

Also since the aim is to create a generalizable model that can be applied to a wide collection of tasks I start with balanced dataset and then see the impact of creating imbalance.

I test a 50-50 split, a 60-40 split, 70-30 split, 80-20 split and 90-10 split on the generated dataset.

### TextCNN

The TextCNN model hyperparameters were tweaked according to Zhang's paper [<cite>"A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification" Zhang et al.</cite>] , with GloVe 6B, 300dim vectors.

Filter Size of 3, 4, 5 was used as it unconditionally yields the best result as shown in the paper. A batch size of 50 was used and a dropout of 0.5 was added to prevent overfitting. The model that we are using here is also double channeled.

### LSTM Classifier

## Experiments

### Data
The data used in the experiments are as follows,

* LitQuotes - around 2300 book quotes were scraped off of LitQuotes.com and paired up with non-quotes from same books available via Project Gutenberg. 

* Quotationspage.com - 5100 general quotes which are paired with random non-quote newspaper dataset.

* Data from Memorability 2012's paper - 2197 one-sentence memorable quotes paired with surrounding non-memorable quotes from the same movie, spoken by the same character and containing the same number of words.

This gets us around 9,000 pairs of quotes and non-quotes.