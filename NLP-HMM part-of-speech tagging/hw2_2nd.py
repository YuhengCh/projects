#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sys
import pprint, time

input_train = sys.argv[1]
input_dev = sys.argv[2]
input_test = sys.argv[3]

# ## Task1: Vocabulary Creation

# In[151]:


## reading training text
f = open(input_train, "r")
text = [line.strip() for line in f.readlines()]


# In[138]:


# text[:20]


# In[152]:


## split each word and its tag into list
split = [item.split('\t') for item in text] 


# In[6]:


# split[:20]


# In[505]:


## create a dictionary to count word occurences
dict1 = {}
for item in split:
    if len(item) > 1:
        key = item[1]
        if key not in dict1:
            dict1[key] = 1
        else:
            dict1[key] += 1


# In[10]:


## sort dictionary in descending order
dict_sorted = {k: v for k, v in sorted(dict1.items(), key=lambda item: item[1], reverse = True)}

## replace rare words whose occurence less than 3 with a special token ‘< unk >’.
rare_words_occ = 0
rare_words = []
for k,v in dict_sorted.items():
    if v < 3:
        rare_words_occ += v
        rare_words.append([k,v])

## delete rare words and add unkown token to vocabulary and update vocabulary
for item in rare_words:
      dict_sorted.pop(item[0])
updict = {'<unk>': rare_words_occ}
dict_vocab = {**updict, **dict_sorted}
# dict_sorted.update({'<unk>': rare_words_occ})
# print('size of the vocabulary is', len(dict_vocab))


# In[11]:


## not removing puntuations and transform the dictionary into correct vocab form
i = 0
vocab = []
for k, v in dict_vocab.items():
    word = k + ' \t ' + str(i) + ' \t ' + str(v)
    vocab.append(word)
    i += 1
# print('total size of vocabulary is', len(vocab))


# In[12]:


## output the vocabulary into a txt file named vocab.txt
file = open('vocab.txt','w')
for item in vocab:
	file.write(item+"\n")
file.close()


# In[13]:


print('The selected threshold for unknown words replacement is', 3)
print('The total size of vocabulary before replacement is', len(dict1))
print('The total size of vocabulary after replacement is', len(vocab))
print('The total occurrences of the special token ‘< unk >’ after replacement is', rare_words_occ)


# ## Task 2: Model Learning

# In[14]:


## get unique tags from training dataset
tags1 = set([x[2] for x in split if len(x) > 1])
# len(tags1)


# In[15]:


## get unique words from training dataset
words1 = set([x[1] for x in split if len(x) > 1])
# len(words1)


# In[84]:


## unknown words list
unk_word = [x[0] for x in rare_words]


# In[58]:


## create a dataframe for transition
tags = []
for item in tags1:
    tags.append(item)
row = tags.insert(0, 'start')
df = pd.DataFrame(columns=tags,
                  index=tags)
df_tran = df.drop(df.columns[0],axis=1)

## create a dataframe for emission
words = []
for item in words1:
    words.append(item)
df = pd.DataFrame(columns=words,
                  index=tags)
df_emm = df.drop(index='start')

# print('df done')
# In[935]:


## transition function
def calculate_transition(current, next, training):   
    denominator = 0
    count = 0
    if current == 'start':
        # count_current = 0
        for item in training:
            if len(item) > 1:
                if item[2] == next:
                    denominator += 1
                    if item[0] == '1':
                        count += 1
    else:
        for i in range(len(training)):
            if len(training[i]) > 1:
                if training[i][2] == current:
                    denominator += 1
                    index = i + 1
                    if (index < len(training)) and (len(training[index])>1):
                        if training[index][2] == next:
                            count += 1
        i += 1
    return count/denominator


## emission funtion
def calculate_emission(word, tag, training):
    denominator = 0
    count = 0
    for item in training:
        if (len(item) > 1) and (item[2] == tag):
            denominator += 1
            if item[1] == word:
#         if (len(item) > 1) and (item[1] == word) and (item[2] == tag):
                count += 1
    return [count/denominator, count, denominator]


# In[943]:


# calculate_emission('THE', 'DT', split)


# In[1012]:

## separate unknown words and words that are in vocab
## i will find emission for unknown words based on unk/states
in_vocab = []
notin_vocab = []
for item in split:
    if len(item) > 1 and item[1] in dict_sorted:
        in_vocab.append(item)
    else:
        notin_vocab.append(item)

split2_word_tag, split2_tag = [], []
for item in split:
    if len(item) < 2:
        split2_tag.append('')
    else:
        split2_tag.append(item[2])

for item in in_vocab:
    if len(item) < 2:
        split2_word_tag.append('')
    else:
        split2_word_tag.append(item[1] + ' '+item[2])
# len(split2_word_tag)


# In[1014]:


from collections import defaultdict
dct = defaultdict(int)

for key in split2_word_tag:
    dct[key] += 1
# dct


# In[233]:


# calculate_transition('start','WP$', split)


# In[115]:


# calculate_emission('PLC', 'NNP', test)


# In[286]:


## built transition dictionary: {(s, s'): t(s'|s)}
transition_dict = {}
for i in df_tran.index:
    for j in df_tran.columns:
        prob = calculate_transition(i, j, split)
        transition_dict[(i,j)] = prob
        df_tran.at[i,j] = prob

# print('transition done')
# In[131]:


## separate unknown words and words that are in vocab
## i will find emission for unknown words based on unk/states
# in_vocab = []
# notin_vocab = []
# for item in split:
#     if len(item) > 1 and item[1] in dict_sorted:
#         in_vocab.append(item)
#     else:
#         notin_vocab.append(item)


# In[163]:


## count tags and tags with words functions
def calculate_denominator(tag, training):
    denominator = 0
    for item in training:
        if (len(item) > 1) and (item[2] == tag):
            denominator += 1
    return denominator
def calculate_count(word, tag, training):
    count = 0
    for item in training:
        if (len(item) > 1) and (item[1] == word) and (item[2] == tag):
            count += 1
    return count


# In[164]:


# calculate_count('.', '.', split)


# In[158]:


# calculate_denominator('NNP', split)


# In[961]:


d = {}
for tag in list(tags1):
    d[tag] = split2_tag.count(tag)


# In[1033]:


## build emission dictionary: {(s,x): e(x|s)}
## when the word is unknown, calculate e(s|'unk')
start = time.time()
emission_dict2 = {}
z = 0
for key, value in dict_vocab.items():
    start = time.time()
    for item in list(tags1):
#         start = time.time()
        word = key
        tag = item
        if (tag, word) in emission_dict2:
            continue
        prob = dct[word + ' ' + tag] / d[tag]
        #calculate_emission(word, tag, split)
        if prob != 0:
            emission_dict2[(tag,word)] = prob
        else:
            continue
    end = time.time()
  
    z += 1

end = time.time()
difference = end-start
 
# print("Time taken in seconds: ", difference)


# In[1040]:


dict_tag = {}
for item in list(tags1):
    val = calculate_denominator(item, split)
    dict_tag[item] = val
for item in notin_vocab:
    if len(item) > 1:
        tag = item[2]
        word = item[1]
        if (tag, 'unk') in emission_dict2:
            continue
        else:
            prob = calculate_denominator(tag, notin_vocab)/dict_tag[tag]
            emission_dict2[(tag,'unk')] = prob


# In[1042]:

# print('emm done')
# emission_dict2 == emission_dict


# In[ ]:


## build emission dictionary: {(s,x): e(x|s)}
## when the word is unknown, calculate e(s|'unk')
# emission_dict = {}
# for item in in_vocab:
#     word = item[1]
#     tag = item[2]
#     if (tag, word) in emission_dict:
#         continue
#     prob = calculate_count(word, tag, in_vocab)/calculate_denominator(tag, split)
#     emission_dict[(tag,word)] = prob


# In[206]:


## adding unknown word to emission dictionary
# dict_tag = {}
# for item in list(tags1):
#     val = calculate_denominator(item, split)
#     dict_tag[item] = val
# for item in notin_vocab:
#     if len(item) > 1:
#         tag = item[2]
#         word = item[1]
#         if (tag, 'unk') in emission_dict:
#             continue
#         else:
#             prob = calculate_denominator(tag, notin_vocab)/dict_tag[tag]
#             emission_dict[(tag,'unk')] = prob


# In[ ]:





# In[ ]:





# In[1046]:


## turn tuple keys of dictionaries into keys and convert them to json form
dicts_tran = {" ".join(key): value for key, value in transition_dict.items()}
dicts_emm = {" ".join(key): value for key, value in emission_dict2.items()}


# In[360]:


a = {'transition': dicts_tran, 'emission': dicts_emm}
import json
json_file = json.dumps(a)


# In[361]:


## output json file
with open('hmm.json', 'w') as f:
    json.dump(json_file, f)


# In[274]:


# f = open ('hmm.json', "r")
  
# # Reading from file
# data = json.loads(f.read())


# In[362]:


## numbers of transition and emission parameters in my HMM
num_tran, num_emm = 0, 0
for key, value in transition_dict.items():
    if value != 0:
        num_tran += 1
for key,value in emission_dict2.items():
    if value != 0:
        num_emm += 1
print('number of transition parameters in HMM is ', num_tran)
print('number of emission parameters in HMM is ', num_emm)


# In[ ]:





# ## Task 3: Greedy Decoding withHMM

# In[363]:


# f = open ('hmm.json', "r")
  
# # Reading from file
# data = json.loads(f.read())
# tran = json.loads(data)['transition']
# emm = json.loads(data)['emission']


# In[1047]:


# dicts_emm == emm


# In[1048]:


# dicts_tran == tran


# In[1049]:


emm = dicts_emm
tran = dicts_tran


# In[365]:


f_dev = open(input_dev, "r")
text_dev = [line.strip() for line in f_dev.readlines()]
split_dev = [item.split('\t') for item in text_dev]


# In[ ]:





# In[ ]:


## greedy algorithm

def find_max_tag_emm(x):
    values, keys = [], []
    for key, value in emm.items():
        if key.split(' ')[1] == x:
            values.append(value)
            keys.append(key.split(' ')[0])
    if len(values) == 0:
        for key, value in emm.items():
            if key.split(' ')[1] == 'unk':
                values.append(value)
                keys.append(key.split(' ')[0])
#     return keys[np.argmax(values)].split(' ')[0]
    return keys,values

predicted_tags, previous = [],''
i = 0
df_to_dict = df_tran.to_dict(orient='index')
for item in split_dev:
    if len(item) < 2:
        continue
    word = item[1]    
    if item[0] == '1':
        previous = 'start'
    tran_find = df_to_dict[previous]
    emm_find = find_max_tag_emm(word)
    tags = emm_find[0]
    pred = tags[np.argmax(np.array([tran_find[x] for x in tags]) * emm_find[1])]
    predicted_tags.append(pred)
    previous = pred
    i += 1
#     print(i)


# In[470]:


actual_tags = [item[2] for item in split_dev if len(item) > 1]
acc = 0
for i in range(len(actual_tags)):
    if predicted_tags[i] == actual_tags[i]:
        acc += 1
accuracy = acc/len(actual_tags)
# accuracy


# In[875]:


print('The accuracy on the dev data(greedy decoding) is ', accuracy)


# In[471]:


## use test data

f_test = open(input_test, "r")
text_test = [line.strip() for line in f_test.readlines()]
split_test = [item.split('\t') for item in text_test]


# In[473]:


# len(split_test)


# In[ ]:


## greedy algorithm on test data

predicted_tags, previous = [],''

for item in split_test:
    if len(item) < 2:
        predicted_tags.append('')
        continue
    word = item[1]    
    if item[0] == '1':
        previous = 'start'
    tran_find = df_to_dict[previous]
    emm_find = find_max_tag_emm(word)
    tags = emm_find[0]
    pred = tags[np.argmax(np.array([tran_find[x] for x in tags]) * emm_find[1])]
    predicted_tags.append(pred)
    previous = pred
    
#     print(i)
   


# In[503]:


##combine tags with text in test data and output as 'greedy.out'
output_test_pred = []
for i in range(len(text_test)):
    if text_test[i] != '':
        output_test_pred.append(text_test[i] + '\t'+ predicted_tags[i])  
    else:
        output_test_pred.append(text_test[i])


# In[508]:


file_test_pred = open('greedy.out','w')
for item in output_test_pred:
	file_test_pred.write(item+"\n")
file_test_pred.close()


# In[ ]:




# print('viterbi')
# ## Task 4: Viterbi Decoding with HMM

# In[635]:


def find_emm_prob(tag, word):
    key = tag + ' ' + word
    if key in emm:
        return emm[key]
    elif (tag+' ' + 'unk' in emm) and word not in vocab_lists:
        return emm[tag+' '+ 'unk']
    return 0


# In[ ]:


words_em = []
for key, value in emission_dict2.items():
    words_em.append(key[1])
df_emm2= pd.DataFrame(columns=list(set(words_em)),
                  index=list(set(tags1)))
for key, value in emission_dict2.items():
    df_emm2.at[key[0], key[1]] = value
df_emm2.fillna(0, inplace = True)
# df_emm2

df_dict_emm = df_emm2.to_dict()
df_dict_tran = df_tran.to_dict()
# In[751]:


## source: https://towardsdatascience.com/implementing-part-of-
## speech-tagging-for-english-words-using-viterbi-algorithm-from-scratch-9ded56b29133
# def viterbi_alog(sentence, tags):
#     tags = set(list(tags))
#     path = {}
#     for t in tags:
#         if sentence[0] not in df_emm2.columns:
#             path[t, 0] = df_tran.loc['start', t] * df_emm2.loc[t, 'unk']
#         else:
#             path[t, 0] = df_tran.loc['start', t] * df_emm2.loc[t, sentence[0]]

#     for i in range(1, len(sentence)):
#         if sentence[i] not in df_emm2.columns:
#             obs = 'unk'
#         else:
#             obs = sentence[i]
#         for t in tags:
#             v1 = [(path[k, i-1] * df_tran.loc[k, t] * df_emm2.loc[t, obs], k) for k in tags]
#             k = sorted(v1)[-1][1]
#             path[t, i] = path[k, i-1]* df_tran.loc[k,t] * df_emm2.loc[t, obs]

#     best = []
#     for i in range(len(sentence) - 1, -1, -1):
#         k = sorted([(path[k, i], k) for k in tags])[-1][1]
#         best.append((sentence[i], k))
#     best.reverse()

#     return [str(item[0]) + " " + str(item[1]) for item in best]

def viterbi_alog(sentence, tags):
    tags = set(list(tags))
    path = {}
    for t in tags:
        if sentence[0] not in df_emm2.columns:
            path[t, 0] = df_dict_tran[t]['start'] * df_dict_emm['unk'][t]
#             df_tran.loc['start', t] * df_emm2.loc[t, 'unk']
        else:
            path[t, 0] = df_dict_tran[t]['start'] * df_dict_emm[sentence[0]][t]
#             df_tran.loc['start', t] * df_emm2.loc[t, sentence[0]]

    for i in range(1, len(sentence)):
        if sentence[i] not in list(set(words_em)):
            obs = 'unk'
        else:
            obs = sentence[i]
        for t in tags:
            v1 = [(path[k, i-1] * df_dict_tran[t][k] * df_dict_emm[obs][t], k) for k in tags]
#             [(path[k, i-1] * df_tran.loc[k, t] * df_emm2.loc[t, obs], k) for k in tags]
            k = sorted(v1)[-1][1]
            path[t, i] = path[k, i-1]* df_dict_tran[t][k] * df_dict_emm[obs][t]

    best = []
    for i in range(len(sentence) - 1, -1, -1):
        k = sorted([(path[k, i], k) for k in tags])[-1][1]
        best.append((sentence[i], k))
    best.reverse()

    return [str(item[0]) + " " + str(item[1]) for item in best]

# In[855]:


dev_copy = split_dev.copy()
dev_copy.append([''])


# In[856]:


split_in_sentence = []
sentence = []
for item in dev_copy:
    if len(item) < 2:
        split_in_sentence.append(sentence)
        sentence = []
    else:
        sentence.append(item[1])
    


# In[ ]:


predicted_tags_viterbi = []

for item in split_in_sentence:
    predicted_tags_viterbi.extend(viterbi_alog(item, tags1))
    predicted_tags_viterbi.append('')

    # print(z)



# In[871]:


predicted_t_dev = []
for item in predicted_tags_viterbi:
    a = item.split(' ')
    if len(a) > 1:
        predicted_t_dev.append(a[1])


# In[874]:


# actual_tags = [item[2] for item in split_dev if len(item) > 1]
predicted_tags_viterbi2 = [item for item in predicted_t_dev if item != '']
acc_viterbi = 0
for i in range(len(actual_tags)):
    if predicted_tags_viterbi2[i] == '':
        continue
    if predicted_tags_viterbi2[i] == actual_tags[i]:
        acc_viterbi += 1
accuracy_viterbi = acc_viterbi/len(actual_tags)
print('The accuracy on the dev data(Viterbi decoding) is ', accuracy_viterbi)


# In[ ]:





# In[ ]:


## output predcitions for test data


# In[877]:


test_copy = split_test.copy()
test_copy.append([''])

split_in_sentence2 = []
sentence2 = []
for item in test_copy:
    if len(item) < 2:
        split_in_sentence2.append(sentence2)
        sentence2 = []
    else:
        sentence2.append(item[1])
   


# In[ ]:



predicted_tags_viterbi_test = []

for item in split_in_sentence2:
    predicted_tags_viterbi_test.extend(viterbi_alog(item, tags1))
    predicted_tags_viterbi_test.append('')





# In[891]:


predicted_t_test = []
for item in predicted_tags_viterbi_test:
    a = item.split(' ')
    if len(a) > 1:
        predicted_t_test.append(a[1])


# In[897]:


predicted_t_test = []
for item in predicted_tags_viterbi_test:
    a = item.split(' ')
    if len(a) > 1:
        predicted_t_test.append(a[1])
    else:
        predicted_t_test.append(item)


# In[902]:


##combine tags with text in test data and output as 'greedy.out'
output_test_pred_viterbi = []
for i in range(len(text_test)):
    if text_test[i] != '':
        output_test_pred_viterbi.append(text_test[i] + '\t'+ predicted_t_test[i])  
    else:
        output_test_pred_viterbi.append(text_test[i])


# In[907]:


file_test_pred_viterbi = open('viterbi.out','w')
for item in output_test_pred_viterbi:
	file_test_pred_viterbi.write(item+"\n")
file_test_pred_viterbi.close()


# In[ ]:




