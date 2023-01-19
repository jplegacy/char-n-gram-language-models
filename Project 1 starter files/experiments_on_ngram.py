import ngram_lm
import random
#
# print(ngram_lm.ngrams(3, "abcde"))
#
#
# m = ngram_lm.NgramModel(1, 0)
# m.update("abab")
# print(m.get_vocab())
# # {’b’, ’a’}
# m.update("abcd")
# print(m.get_vocab())
# # {’b’, ’a’, ’c’, ’d’}
# print(m.prob("a", "b"))
# # 1.0
# print(m.prob("~", "c"))
# # 0.0
# print(m.prob("b", "c"))
# # 0.5
#
#
# m = ngram_lm.NgramModel(0, 0)
# m.update("abab")
# m.update("abcd")
# random.seed(1)
# print([m.random_char("") for i in range(25)])
#
#
# m = ngram_lm.NgramModel(1, 0)
# m.update("abab")
# m.update("abcd")
# random.seed(1)
# print(m.random_text(25))
#
#
# m = ngram_lm.create_ngram_model(ngram_lm.NgramModel, "shakespeare_input.txt", 2)
# print("\n",m.random_text(250))
#
# print("##################SPLITER######################")
#
# m = ngram_lm.create_ngram_model(ngram_lm.NgramModel, "shakespeare_input.txt", 3)
# print(m.random_text(250))
#
# print("##################SPLITER######################")
#
# m = ngram_lm.create_ngram_model(ngram_lm.NgramModel, "shakespeare_input.txt", 4)
# print(m.random_text(250))
#
# print("##################SPLITER######################")
#
# m = ngram_lm.create_ngram_model(ngram_lm.NgramModel, "shakespeare_input.txt", 7)
# print(m.random_text(250))
#
#
# m = ngram_lm.NgramModel(1, 0)
# m.update("abab")
# m.update("abcd")
# print(m.perplexity("abcd"))
# # # 1.189207115002721
# print(m.perplexity("abca"))
# # # inf
# print(m.perplexity("abcda"))
# # # 1.515716566510398

m = ngram_lm.create_ngram_model(ngram_lm.NgramModel, "shakespeare_input.txt", 2, 1)
with open("shakespeare_sonnets.txt") as f:
    print(m.perplexity(f.read()))

with open("nytimes_article.txt", encoding='ISO-8859-1') as f:
    print(m.perplexity(f.read()))

with open("shakespeare_input.txt", encoding='ISO-8859-1') as f:
    print(m.perplexity(f.read()))


#
# m = ngram_lm.NgramModel(1, 1)
# (m.update("abab"))
# (m.update("abcd"))
# print(m.prob("a", "a"))
# # 0.14285714285714285
# print(m.prob("a", "b"))
# # 0.5714285714285714+
# print(m.prob("c", "d"))
# # 0.4
# print(m.prob("d", "a"))
# # 0.25
#
#
# m = ngram_lm.NgramModelWithInterpolation(1, 0)
# m.update("abab")
# print(m.prob("a", "a"))
# # 0.25
# print(m.prob("a", "b"))
# # 0.75
# m = ngram_lm.NgramModelWithInterpolation(2, 1)
# m.update("abab")
# m.update("abcd")
# print(m.prob("~a", "b"))
# # 0.4682539682539682
# print(m.prob("ba", "b"))
# # 0.4349206349206349
# print(m.prob("~c", "d"))
# # 0.27222222222222225
# print(m.prob("bc", "d"))
# # 0.3222222222222222
