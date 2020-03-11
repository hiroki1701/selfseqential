from __future__ import unicode_literals
import spacy
nlp = spacy.load('en_core_web_sm')
import textacy
# sentence = 'the back tire of an old style motorcycle is resting in a metal stand'
pattern = r'<VERB>?<ADV>*<VERB>+'
# doc = textacy.Doc(sentence, lang='en_core_web_sm')
# lists = textacy.extract.pos_regex_matches(doc, pattern)
# for list in lists:
#     print(list.text)
#
# exit()


text = u'''the back tire of an old style motorcycle is resting in a metal stand'''

tokens = nlp(text)

for i in tokens.noun_chunks:
    print(i,i[-1])
print('----------------')
doc = textacy.Doc(text, lang='en_core_web_sm')
lists = textacy.extract.pos_regex_matches(doc, pattern)
for list in lists:
    print(list.text)
print('----------------')
exit()


text = u'''bunk bed with a narrow shelf sitting underneath it '''

tokens = nlp(text)

for i in tokens.noun_chunks:
    print(i)
print('----------------')
text = u'''a giraffe in a enclosed area is watched by some people'''

tokens = nlp(text)

for i in tokens.noun_chunks:
    print(i)
print('----------------')
text = u'''the giraffe is being kept by itself indoors'''

tokens = nlp(text)

for i in tokens.noun_chunks:
    print(i)

exit()


split_tokens = []
for token in tokens:
    print(token.text, token.pos_, token.dep_)
    split_tokens.append(str(token.text))
    if token.pos_ == 'PROPN' or token.pos_ == 'NOUN':
        print('--')
        split_tokens.append('|')


print('----------------')
print(split_tokens)
print('----------------')

text = u'''a small dog is curled up on top of the shoes'''

tokens = nlp(text)

split_tokens = []
for token in tokens:
    print(token.text, token.pos_, token.dep_)
    split_tokens.append(str(token.text))
    if token.pos_ == 'PROPN' or token.pos_ == 'NOUN':
        print('--')
        split_tokens.append('|')


print('----------------')
print(split_tokens)