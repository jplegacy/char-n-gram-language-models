import math
import random

# PLEASE do not delete or modify the comments that divide the code
# into sections, like the following comment.

################################################################################
# Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']


def start_pad(c):
    """ Returns a padding string of length c to append to the front of text
        as a pre-processing step to building n-grams. c = n-1 """
    return '~' * c


def ngrams(c, text):
    """ Returns the ngrams of the text as tuples where the first element is
        the length-c context and the second is the character """

    padded_text = start_pad(c) + text

    total_ngrams = []
    for partition in range(0, int(len(padded_text) - c)):
        ngram = (padded_text[partition:partition + c], padded_text[partition + c])
        total_ngrams.append(ngram)
    return total_ngrams


def create_ngram_model(model_class, path, c=2, k=0):
    """ Creates and returns a new n-gram model trained on the entire text
        found in the path file """
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model


def create_ngram_model_lines(model_class, path, c=2, k=0):
    """Creates and returns a new n-gram model trained line by line on the
        text found in the path file. """
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model

################################################################################
# Basic N-Gram Model
################################################################################


class NgramModel(object):
    """ A basic n-gram model using add-k smoothing """

    def __init__(self, c, k):
        self.context_length = c
        self.k_smoothing_co = k
        self.vocab = set()
        self.gram_collection = {}

    def get_vocab(self):
        """ Returns the set of characters in the vocab """
        return self.vocab

    def update(self, text):
        """ Updates the model n-grams based on text """

        newly_collected_grams = ngrams(self.context_length, text)

        for ngram in newly_collected_grams:
            self.vocab.add(ngram[1])
            self.gram_counter(ngram[0], ngram[1])

    def gram_counter(self, context, char):
        """ Updates the model n-grams based on text """

        if not self.is_gram_collected(context, "*"):
            self.gram_collection[context] = {}

        if not self.is_gram_collected(context, char):
            self.gram_collection[context][char] = 0

        self.gram_collection[context][char] += 1

    def is_gram_collected(self, context, char):
        if context not in self.gram_collection.keys():
            return False

        if char != "*" and char not in self.gram_collection[context].keys():
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

    def prob(self, context, char):
        """ Returns the pro bability of char appearing after context """

        times_context_found = self.gram_entries(context, "*") + self.k_smoothing_co * len(self.vocab)

        if times_context_found == 0  :
            return 1 / (len(self.vocab))

        times_char_after_context_found = self.gram_entries(context, char) + self.k_smoothing_co

        current_probability = times_char_after_context_found / times_context_found

        return current_probability

    def random_char(self, context):
        """ Returns a random character based on the given context and the
            n-grams learned by this model """

        r = random.random()

        summated_prob = 0
        for letter in sorted(self.vocab):
            summated_prob += self.prob(context, letter)

            if summated_prob > r:
                return letter

        return "ERROR"

    def random_text(self, length):
        """ Returns text of the specified character length based on the
            n-grams learned by this model """

        randomized_text = ""

        for times in range(length):
            padded_text = (start_pad(self.context_length) + randomized_text)[times:self.context_length + times]

            randomized_text += self.random_char(padded_text)

        return randomized_text

    def perplexity(self, text):
        """ Returns the perplexity of text based on the n-grams learned by
            this model """

        ngrams_collection = ngrams(self.context_length, text)

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
    """ An n-gram model with interpolation """

    def __init__(self, c, k, lamda_list=None):
        self.K_smoothing_coe = 0
        self.vocab = set()
        self.longest_context = c
        self.gram_collections = []

        for gram in range(0, self.longest_context + 1):
            self.gram_collections.append(NgramModel(gram, k))

        if lamda_list is None:
            self.gram_weight = [1/(self.longest_context + 1) for n in range(c+1)]
        else:
            self.gram_weight = lamda_list

    def update(self, text):
        for gram in self.gram_collections:
            gram.update(text)

    def prob(self, context, char):
        summated_prob = 0
        for gram in self.gram_collections:
            partitioned_context = context[-gram.context_length:]
            if gram.context_length == 0:
                partitioned_context = ''

            summated_prob += gram.prob(partitioned_context, char)

        return self.gram_weight[gram.context_length] * summated_prob

    # def average_perplexity(self, text):
    #     total_perplexities = 0
    #     for gram in self.gram_collections:
    #         total_perplexities += gram.perplexity(text)
    #     return total_perplexities/len(self.gram_collections)

################################################################################
# Your N-Gram Model Experimentations
################################################################################


def test_ngrams():
    print(ngrams(3, "abcde"))


def test_update():
    m = NgramModel(1, 0)
    m.update("abab")
    print(m.get_vocab())
    # {’b’, ’a’}
    m.update("abcd")
    print(m.get_vocab())
    # {’b’, ’a’, ’c’, ’d’}
    print(m.prob("a", "b"))
    # 1.0
    print(m.prob("~", "c"))
    # 0.0
    print(m.prob("b", "c"))
    # 0.5


def test_random_char():
    m = NgramModel(0, 0)
    m.update("abab")
    m.update("abcd")
    random.seed(1)
    print([m.random_char("") for i in range(25)])


def test_random_test():
    m = NgramModel(1, 0)
    m.update("abab")
    m.update("abcd")
    random.seed(1)
    print(m.random_text(25))


def test_random_test_shakespear_text():
    m = create_ngram_model(NgramModel, "shakespeare_input.txt", 2)
    print(m.random_text(250))
    m = create_ngram_model(NgramModel, "shakespeare_input.txt", 3)
    print(m.random_text(250))
    m = create_ngram_model(NgramModel, "shakespeare_input.txt", 4)
    print(m.random_text(250))
    m = create_ngram_model(NgramModel, "shakespeare_input.txt", 7)
    print(m.random_text(250))


def test_perplexity():
    m = NgramModel(1, 0)
    m.update("abab")
    m.update("abcd")
    print(m.perplexity("abcd"))
    # # 1.189207115002721
    print(m.perplexity("abca"))
    # # inf
    print(m.perplexity("abcda"))
    # # 1.515716566510398


def test_prob_with_k():
    m = NgramModel(1, 1)
    (m.update("abab"))
    (m.update("abcd"))
    print(m.prob("a", "a"))
    # 0.14285714285714285
    print(m.prob("a", "b"))
    # 0.5714285714285714
    print(m.prob("c", "d"))
    # 0.4
    print(m.prob("d", "a"))
    # 0.25


def test_prob_with_interpolation():
    m = NgramModelWithInterpolation(1, 0)
    m.update("abab")
    print(m.prob("a", "a"))
    # 0.25
    print(m.prob("a", "b"))
    # 0.75
    m = NgramModelWithInterpolation(2, 1)
    m.update("abab")
    m.update("abcd")
    print(m.prob("~a", "b"))
    # 0.4682539682539682
    print(m.prob("ba", "b"))
    # 0.4349206349206349
    print(m.prob("~c", "d"))
    # 0.27222222222222225
    print(m.prob("bc", "d"))
    # 0.3222222222222222


parallel_models = {}
model_guesses = {}

#########################
#   PARAMETERS
#########################
MAX_MODEL_N_SIZE = 3

# LAMDA_LIST MUST BE SAME SIZE AS INPUTTED MAX N SIZE + 1
LAMDA_LIST = [1/4, 1/4, 1/4, 1/4]

MODEL_K_SMOOTHING = 0


def create_models():
    for country in COUNTRY_CODES:
        parallel_models[country] = NgramModelWithInterpolation(MAX_MODEL_N_SIZE, MODEL_K_SMOOTHING, LAMDA_LIST)
        model_guesses[country] = {}
        with open("./train/" + country + '.txt', encoding='ISO-8859-1') as training_text:
            text = training_text.read().replace('\n', '')
            parallel_models[country].update(text)


def evaluate_models():
    for country in COUNTRY_CODES:
        for country_to_guess in COUNTRY_CODES:
            with open("./val/"+country_to_guess+'.txt', encoding='ISO-8859-1') as valuation_text:
                text = valuation_text.read().replace('\n', '')
                model_guesses[country][country_to_guess] = 0

                ngrams_of_text = ngrams(MAX_MODEL_N_SIZE, text)
                for gram in ngrams_of_text:
                    probability_of_lang = parallel_models[country].prob(gram[0], gram[1])
                    model_guesses[country][country_to_guess] += probability_of_lang


def results():
    for key in model_guesses.keys():
        print("MODEL: ", key)
        for country_guess in model_guesses[key].keys():
            print(key, " guess for ", country_guess, " was ", model_guesses[key][country_guess])


create_models()
evaluate_models()
results()