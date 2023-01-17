import math, random

# PLEASE do not delete or modify the comments that divide the code
# into sections, like the following comment.

################################################################################
# Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

def start_pad(c):
    ''' Returns a padding string of length c to append to the front of text
        as a pre-processing step to building n-grams. c = n-1 '''
    return '~' * c

def ngram_partitions(size, text):
    total_ngrams = []
    for partition in range(0, int(len(text)-size)):
        ngram = (text[partition:partition+size], text[partition+size])
        total_ngrams.append(ngram)
    return total_ngrams

def ngrams(c, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-c context and the second is the character '''

    padded_text = start_pad(c) + text

    return ngram_partitions(c, padded_text)


# def ngram_step(text:str, startIndex:int, endingIndex:int, total_ngrams:list):
#
#     if endingIndex >= len(text):
#         return total_ngrams
#
#     ngram = (text[startIndex: endingIndex], text[endingIndex])
#     total_ngrams.append(ngram)
#     return ngram_step(text, startIndex + 1, endingIndex + 1, total_ngrams)

# def ngram_step(text,start_index, ending_index, total_ngrams):
#     for index in range(start_index, ending_index):
#         text[]



def create_ngram_model(model_class, path, c=2, k=0):
    ''' Creates and returns a new n-gram model trained on the entire text
        found in the path file '''
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def create_ngram_model_lines(model_class, path, c=2, k=0):
    '''Creates and returns a new n-gram model trained line by line on the
        text found in the path file. '''
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model

################################################################################
# Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, c, k):
        self.context_len = c
        self.k_smoothing_co = k
        self.vocab = set()
        self.gram_collection = []
    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''

        newly_collected_grams = ngrams(self.context_len, text)

        for ngram in newly_collected_grams:
            self.vocab.add(ngram[1])
            self.gram_collection.append(ngram)

    #helper
    def ngramInstanceCounter(self, context, char):
        instances = 0

        for ngrams in self.gram_collection:
            context_of_gram = ngrams[0]
            char_of_gram = ngrams[1]

            if context_of_gram == context:
                if char == char_of_gram or char == "*":
                    instances += 1

        return instances

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''

        times_context_found = self.ngramInstanceCounter(context, "*")

        if times_context_found == 0:
            return 1 / (len(self.vocab))

        times_char_after_context_found = self.ngramInstanceCounter(context, char)

        current_probability = times_char_after_context_found / times_context_found

        return current_probability

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''

        r = random.random()

        summated_prob = 0
        for letter in sorted(self.vocab):
            summated_prob += self.prob(context,letter)

            if summated_prob > r:
                return letter

        return "ERROR"

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''

        randomized_text = ""

        for times in range(length):
            padded_text = (start_pad(self.context_len) + randomized_text)[times:self.context_len+times]

            randomized_text += self.random_char(padded_text)

        return randomized_text

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        ngrams_collection = ngrams(self.context_len, text)

        chance_of_text = 1.0
        for ngram in ngrams_collection:
            chance_of_text *= self.prob(ngram[0], ngram[1])

        if chance_of_text == 0:
            chance_of_text = float("inf")

        entropy = (-1/self.context_len) * math.log2(chance_of_text)

        return 2 ** entropy

################################################################################
# N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, c, k):
        pass

    def get_vocab(self):
        pass

    def update(self, text):
        pass

    def prob(self, context, char):
        pass

################################################################################
# Your N-Gram Model Experimentations
################################################################################

# print(ngrams(3,"abcde"))
#
# m = NgramModel(1, 0)
# m.update("abab")
# print(m.get_vocab())
# # {’b’, ’a’}
# m.update("abcd")
# print(m.get_vocab())
# #{’b’, ’a’, ’c’, ’d’}
# print(m.prob("a", "b"))
# #1.0
# print(m.prob("~", "c"))
# #0.0
# print(m.prob("b", "c"))
# # 0.5

# m = NgramModel(0, 0)
# m.update("abab")
# m.update("abcd")
# random.seed(1)
# print([m.random_char("") for i in range(25)]
# )


m = NgramModel(1, 0)
m.update("abab")
m.update("abcd")
random.seed(1)
print(m.random_text(25))

# m = create_ngram_model(NgramModel, "shakespeare_input.txt", 2)
# m.random_text(250)
# m = create_ngram_model(NgramModel, "shakespeare_input.txt", 3)
# m.random_text(250)
# m = create_ngram_model(NgramModel, "shakespeare_input.txt", 4)
# m.random_text(250)
# m = create_ngram_model(NgramModel, "shakespeare_input.txt", 7)
# m.random_text(250)

m = NgramModel(1, 0)
m.update("abab")
m.update("abcd")
print(m.perplexity("abcd"))
# 1.189207115002721
print(m.perplexity("abca"))
# inf
print(m.perplexity("abcda"))
# 1.515716566510398
