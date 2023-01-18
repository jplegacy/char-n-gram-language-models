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
        self.context_length = c
        self.k_smoothing_co = k
        self.vocab = set()
        self.gram_collection = {}
    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''

        newly_collected_grams = ngrams(self.context_length, text)

        for ngram in newly_collected_grams:
            self.vocab.add(ngram[1])
            self.dictionary_entry_controller(ngram[0], ngram[1])

    def dictionary_entry_controller(self, context, char):
        if not self.is_gram_collected(context, "*"):
            self.gram_collection[context] = {}

        if not self.is_gram_collected(context, char):
            self.gram_collection[context][char] = 0

        self.gram_collection[context][char] += 1

    def is_gram_collected(self, context, char):
        if context not in self.gram_collection.keys():
            return False

        if char != "*" and char not in self.gram_collection[context].keys() :
            return False

        return True

    def gram_entries(self, context, char):
        if not self.is_gram_collected(context, char):
            return 0

        elif char == "*":
            instances = 0
            for chars in self.gram_collection[context].keys():
                instances += self.gram_collection[context][chars]

            return instances

        return self.gram_collection[context][char]

    # def ngramInstances(self, context, char):
    #     if not self.context_collected(context):
    #         return 0
    #
    #     context_instances = self.gram_collection[context]
    #
    #     if char == "*":
    #         return len(context_instances)
    #
    #     counter = 0
    #     for item in context_instances:
    #         if item == counter:
    #             counter+=1
    #
    #     return counter

    #helper
    # def ngramInstanceCounter(self, context, char):
    #     instances = 0
    #
    #     for ngrams in self.gram_collection:
    #         context_of_gram = ngrams[0]
    #         char_of_gram = ngrams[1]
    #
    #         if context_of_gram == context:
    #             if char == char_of_gram or char == "*":
    #                 instances += 1
    #
    #     return instances

    def prob(self, context, char):
        ''' Returns the pro bability of char appearing after context '''

        times_context_found = self.gram_entries(context, "*") + self.k_smoothing_co * len(self.vocab)

        if times_context_found == 0:
            return 1 / (len(self.vocab))

        times_char_after_context_found = self.gram_entries(context, char) + self.k_smoothing_co

        current_probability = times_char_after_context_found / times_context_found

        return current_probability

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''

        r = random.random()

        summated_prob = 0
        for letter in sorted(self.vocab):
            summated_prob += self.prob(context, letter)

            if summated_prob > r:
                return letter

        return "ERROR"

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''

        randomized_text = ""

        for times in range(length):
            padded_text = (start_pad(self.context_length) + randomized_text)[times:self.context_length + times]

            randomized_text += self.random_char(padded_text)

        return randomized_text

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''

        ngrams_collection = ngrams(self.context_length, text)

        print(ngrams_collection)
        print(self.gram_collection)

        chance_of_text = 1
        for ngram in ngrams_collection:

            chance_of_text *= self.prob(ngram[0], ngram[1])

        if chance_of_text == 0.0:
            return float("inf")

        entropy = (-1 / self.context_length) * (math.log2(chance_of_text)/len(text))

        return 2 ** entropy

################################################################################
# N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, c, k):
        self.K_smoothing_coe = 0
        self.vocab = set()
        self.longest_context = c
        self.gram_collections = []

        for gram in range(0, self.longest_context):
            self.gram_collections.append(NgramModel(gram, k))


        self.gram_weight = 1/c

    def update(self, text):
        for gram in self.gram_collections:
            gram.update(text)

    def prob(self, context, char):
        summated_prob = 0
        for x, gram in enumerate(self.gram_collections):
            summated_prob += self.gram_weight * gram.prob(context, char)

        return summated_prob

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
# print(m.gram_collection)
#
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
#
#
# m = NgramModel(1, 0)
# m.update("abab")
# m.update("abcd")
# random.seed(1)
# print(m.random_text(25))

# m = create_ngram_model(NgramModel, "shakespeare_input.txt", 2)
# print(m.random_text(250))
# m = create_ngram_model(NgramModel, "shakespeare_input.txt", 3)
# print(m.random_text(250))
# m = create_ngram_model(NgramModel, "shakespeare_input.txt", 4)
# print(m.random_text(250))
# m = create_ngram_model(NgramModel, "shakespeare_input.txt", 7)
# print(m.random_text(250))

# m = NgramModel(1, 0)
# m.update("abab")
# m.update("abcd")
# print(m.perplexity("abcd"))
# # # 1.189207115002721
# print(m.perplexity("abca"))
# # # inf
# print(m.perplexity("abcda"))
# # # 1.515716566510398


# m = NgramModel(1, 1)
# (m.update("abab"))
# (m.update("abcd"))
# print(m.prob("a", "a"))
# # 0.14285714285714285
# print(m.prob("a", "b"))
# # 0.5714285714285714
# print(m.prob("c", "d"))
# # 0.4
# print(m.prob("d", "a"))
# # 0.25


m = NgramModelWithInterpolation(1, 0)
m.update("abab")
print(m.prob("a", "a"))
# 0.25
print(m.prob("a", "b"))
# 0.75
# m = NgramModelWithInterpolation(2, 1)
# m.update(’abab’)
# m.update(’abcd’)
# m.prob(’~a’, ’b’)
# 0.4682539682539682
# m.prob(’ba’, ’b’)
# 0.4349206349206349
# m.prob(’~c’, ’d’)
# 0.27222222222222225
# m.prob(’bc’, ’d’)
# 0.3222222222222222