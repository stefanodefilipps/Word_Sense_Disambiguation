import math
import tensorflow as tf
import tensorflow.keras as K
import tensorflow_hub as hub
from lxml import etree
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from gensim.models import FastText
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
from nltk.corpus import wordnet as wn
from collections import OrderedDict

def create_dataset_multi_learning(path_dataset,path_label,path_lex_label,O,O_lex):
  '''
     This function creates the correct dataset for the multi learning network architecture.
     :param path_dataset: the path to the dataset
     :param path_label: the path to the file containing the labels for the first task
     :param path_lex_label: the path to the file containing the labels for the second task
     :param O: a dictionary containing the mapping from the output vocabulary of the first task to unique integers
     :param O_lex: a dictionary containing the mapping from the output vocabulary of the second task to unique integers
     :returns X_training: a list of list containing at the i-th row the words of the i-th sentence 
     :returns Y_training: a list of list containing at the i-th row the corresponding integer first task labels of the words of the i-th sentence
     :returns Y_lex_training: a list of list containing at the i-th row the corresponding integer second task labels of the words of the i-th sentence 
  '''
  X_training = []
  Y_training = []
  Y_lex_training = []
  dataset = open(path_dataset)
  labels_ = open(path_label)
  labels_lex = open(path_lex_label)
  for line_d,line_l,line_lex in zip(dataset,labels_,labels_lex):
    l_d = line_d.strip("\n").split(" ")
    l_l = line_l.strip("\n").split(" ")
    l_lex = line_lex.strip("\n").split(" ")
    x_t = []
    y_t = []
    y_lex_t = []
    for w_d,w_l,w_lex in zip(l_d,l_l,l_lex):
      x_t.append(w_d)
      y_t.append(O[w_l])
      y_lex_t.append(O_lex[w_lex])
    X_training.append(x_t)
    Y_training.append(y_t)
    Y_lex_training.append(y_lex_t)
  return X_training,Y_training,Y_lex_training

  def create_dataset_basic(path_dataset,path_label,O):
  '''
     This function creates the correct dataset for the basic BSLT or BSLT+attention.
     :param path_dataset: the path to the dataset
     :param path_label: the path to the file containing the labels for the task
     :param O: a dictionary containing the mapping from the output vocabulary of the task to unique integers
     :returns X_training: a list of list containing at the i-th row the words of the i-th sentence 
     :returns Y_training: a list of list containing at the i-th row the corresponding integer task labels of the words of the i-th sentence
  '''
  X_training = []
  Y_training = []
  dataset = open(path_dataset)
  labels_ = open(path_label)
  for line_d,line_l in zip(dataset,labels_):
    l_d = line_d.strip("\n").split(" ")
    l_l = line_l.strip("\n").split(" ")
    x_t = []
    y_t = []
    for w_d,w_l in zip(l_d,l_l):
      x_t.append(w_d)
      y_t.append(O[w_l])
    X_training.append(x_t)
    Y_training.append(y_t)
  return X_training,Y_training

  def create_dataset_multi_learning_lex_boosting(path_dataset,path_label,path_lex_label,O,O_lex,w2b,b2lex,lex2idx,pos_tag_path):
  '''
     This function creates the correct dataset for the multi learning network architecture which employs a kind of hierarchical structure.
     :param path_dataset: the path to the dataset
     :param path_label: the path to the file containing the labels for the first task
     :param path_lex_label: the path to the file containing the labels for the second task
     :param O: a dictionary containing the mapping from the output vocabulary of the first task to unique integers
     :param O_lex: a dictionary containing the mapping from the output vocabulary of the second task to unique integers
     :param w2b: a dictionary containing the mapping from wordnet id to babelnet id
     :param b2lex: a dictionary containing the mapping from babelnet id to lexicographic domain
     :param lex2idx: a dictionary containing the mapping from the possible lexicographic domains to unique integers
     :param pos_tag_path: a path to thr file containing the pos tag associated to a word
     :returns X_training: a list of list containing at the i-th row the words of the i-th sentence 
     :returns Y_training: a list of list containing at the i-th row the corresponding integer first task labels of the words of the i-th sentence
     :returns X_lex_cod: a list of list containing at the i-th row the binary encoding of the most frequent lexicographic domain of the words of the i-th sentence 
     :returns Y_lex_training: a list of list containing at the i-th row the corresponding integer second task labels of the words of the i-th sentence 
  '''
  X_training = []
  Y_training = []
  X_lex_cod = []
  Y_lex_training = []
  dataset = open(path_dataset)
  labels_ = open(path_label)
  pos_tag = open(pos_tag_path)
  labels_lex = open(path_lex_label)
  for line_d,line_l,line_p,line_lex in zip(dataset,labels_,pos_tag,labels_lex):
    l_d = line_d.strip("\n").split(" ")
    l_l = line_l.strip("\n").split(" ")
    l_p = line_p.strip("\n").split(" ")
    l_lex = line_lex.strip("\n").split(" ")
    x_t = []
    y_t = []
    y_lex_t = []
    lex_cod = []
    for w_d,w_l,p,w_lex in zip(l_d,l_l,l_p,l_lex):
      x_t.append(w_d)
      y_t.append(O[w_l])
      y_lex_t.append(O_lex[w_lex])
      # I need to get the most frequent lexicographic domain given the target word w_d. Therefore I can use wordnet and given the word and its pos tag
      # I can take all the synsets associated to that couple and then I take only the first synset since wordnet results are ordered by frequency.
      synset = []
      if wn.synsets(w_d):
        if p not in wn_pos_dict or not wn.synsets(w_d, pos=wn_pos_dict[p]):
           # if the pos tga is not one of the ones allowed in wordnet or there is no synset associated to the couple word pos tag
           # then i simply search the synset associated to the word and take the most frequent one.
           synset =  wn.synsets(w_d)[0]
        else:
          # otherwise i take the most frequent synset associated to the couple word pos
          synset =  wn.synsets(w_d, pos=wn_pos_dict[p])[0]
        # Once I get the wordnet synset I can get its id from which I can get the babelnet id and finally the lexicographic domain of the word
        # Now I can create the correct binary encoding setting only one element to 1
        w_id = "wn:" + str(synset.offset()).zfill( 8) + synset.pos() 
        b_id = w2b[w_id]
        lex_domain = b2lex[b_id]
        lex_one_hot = np.zeros(45)
        lex_one_hot[lex2idx[lex_domain]] = 1
        lex_cod.append(lex_one_hot)
      else:
        #if the current traget word is not present in wordnet then I simply pass an encoding of all 0s
        lex_cod.append(np.zeros(45))
    X_training.append(x_t)
    Y_training.append(y_t)
    X_lex_cod.append(lex_cod)
    Y_lex_training.append(y_lex_t)
  return X_training,Y_training, X_lex_cod, Y_lex_training

def create_dataset_lex_boosting_window(path_dataset,path_label,O,w2b,b2lex,lex2idx,pos_tag_path):
  '''
     This function creates the correct dataset for the architecture that makes use of lexicographic domain within a certain window
     of the target word to boost fine grain predictions.
     :param path_dataset: the path to the dataset
     :param path_label: the path to the file containing the labels for the task
     :param O: a dictionary containing the mapping from the output vocabulary of the task to unique integers
     :param w2b: a dictionary containing the mapping from wordnet id to babelnet id
     :param b2lex: a dictionary containing the mapping from babelnet id to lexicographic domain
     :param lex2idx: a dictionary containing the mapping from the possible lexicographic domains to unique integers
     :param pos_tag_path: a path to thr file containing the pos tag associated to a word
     :returns X_training: a list of list containing at the i-th row the words of the i-th sentence 
     :returns Y_training: a list of list containing at the i-th row the corresponding integer first task labels of the words of the i-th sentence
     :returns X_lex_cod: a list of list containing at the i-th row the binary encoding of the most frequent lexicographic domains of the words of the i-th sentence 
  '''
  X_training = []
  Y_training = []
  X_lex_cod = []
  Y_lex_training = []
  dataset = open(path_dataset)
  labels_ = open(path_label)
  pos_tag = open(pos_tag_path)
  for line_d,line_l,line_p in zip(dataset,labels_,pos_tag):
    l_d = line_d.strip("\n").split(" ")
    l_l = line_l.strip("\n").split(" ")
    l_p = line_p.strip("\n").split(" ")
    x_t = []
    y_t = []
    lex_cod = []
    # this is the sliding window which contains the encoding of the target word, of its preceding word and of its successor
    # initially is set to all encoding of 0s
    window_coding = {"prev": np.zeros(45), "target": np.zeros(45), "next_": np.zeros(45)}
    for idx in range(len(l_d)):
      w_d = l_d[idx]
      w_l = l_l[idx]
      x_t.append(w_d)
      y_t.append(O[w_l])
      if idx == 0:
        # First i focus on the case where my target word is the first word in the sentence
        synset = []
        if wn.synsets(w_d):
          p = l_p[idx]
          # for the next few lines the considerations made for the previous function are also valid
          if p not in wn_pos_dict or not wn.synsets(w_d, pos=wn_pos_dict[p]):
            synset =  wn.synsets(w_d)[0]
          else:
            synset =  wn.synsets(w_d, pos=wn_pos_dict[p])[0]
          w_id = "wn:" + str(synset.offset()).zfill( 8) + synset.pos() 
          b_id = w2b[w_id]
          lex_domain = b2lex[b_id]
          lex_one_hot = np.zeros(45)
          lex_one_hot[lex2idx[lex_domain]] = 1
          # Since it is the first word i Change only the encoding of the second element of the window while the first one is kept to all 0s since it is un
          # undefined case
          window_coding["target"] = lex_one_hot
        else:
          window_coding["target"] = np.zeros(45)
        # Now instead I have to create the encoding of the successor and I have to check that the current word is not also the last word of the sentence
        # if it is the last word then the last element of the sliding windows is kept with the encoding of all 0s since it is un undefined case as well
        if idx + 1 < len(l_d) - 1:
          synset = []
          w_next = l_d[idx+1]
          p_next = l_p[idx+1]
          if wn.synsets(w_next):
            if p_next not in wn_pos_dict or not wn.synsets(w_next, pos=wn_pos_dict[p_next]):
              synset =  wn.synsets(w_next)[0]
            else:
              synset =  wn.synsets(w_next, pos=wn_pos_dict[p_next])[0]
            w_id = "wn:" + str(synset.offset()).zfill( 8) + synset.pos() 
            b_id = w2b[w_id]
            lex_domain = b2lex[b_id]
            lex_one_hot = np.zeros(45)
            lex_one_hot[lex2idx[lex_domain]] = 1
            # Now I change the last element of the sliding window
            window_coding["next_"] = lex_one_hot
          else:
            window_coding["next_"] = np.zeros(45)
      elif idx == len(l_d) - 1:
        # Second particular case happens when the current target word is the last word in the sentence
        # if it is the last word then the last element of the sliding windows is kept with the encoding of all 0s since it is un undefined case as well
        window_coding["next_"] = np.zeros(45)
      else:
        #Now i can focus on the general case
        if idx + 1 < len(l_d) - 1:
          synset = []
          w_next = l_d[idx+1]
          p_next = l_p[idx+1]
          if wn.synsets(w_next):
            if p_next not in wn_pos_dict or not wn.synsets(w_next, pos=wn_pos_dict[p_next]):
              synset =  wn.synsets(w_next)[0]
            else:
              synset =  wn.synsets(w_next, pos=wn_pos_dict[p_next])[0]
            w_id = "wn:" + str(synset.offset()).zfill( 8) + synset.pos() 
            b_id = w2b[w_id]
            lex_domain = b2lex[b_id]
            lex_one_hot = np.zeros(45)
            lex_one_hot[lex2idx[lex_domain]] = 1
            window_coding["next_"] = lex_one_hot
          else:
            window_coding["next_"] = np.zeros(45)
      #Now i have the correct window coding for the target word and i simply need to concatenate
      concatenation = np.concatenate((window_coding["prev"],window_coding["target"],window_coding["next_"]),axis = -1)
      #Now i have to slide the window_coding for the next target word
      window_coding["prev"] = window_coding["target"]
      window_coding["target"] = window_coding["next_"]
      window_coding["next_"] = np.zeros(45)
      lex_cod.append(concatenation)
    X_training.append(x_t)
    Y_training.append(y_t)
    X_lex_cod.append(lex_cod)
  return X_training,Y_training, X_lex_cod,


def create_dataset_subsentences(path_dataset,path_label,O,time_step):
  '''
     This function creates the correct dataset for the architecture that splits sentences longer than timestep into subsentences and then put all of them into
     the dataset.
     :param path_dataset: the path to the dataset
     :param path_label: the path to the file containing the labels for the task
     :param O: a dictionary containing the mapping from the output vocabulary of the task to unique integers
     :param timestep: the maximu length for a sequence
     :returns X_training: a list of list containing at the i-th row the words of the i-th sentence 
     :returns Y_training: a list of list containing at the i-th row the corresponding integer first task labels of the words of the i-th sentence
  '''
  X_training = []
  Y_training = []
  dataset = open(path_dataset)
  labels_ = open(path_label)
  for line_d,line_l in zip(dataset,labels_):
    l_d = line_d.strip("\n").split(" ")
    l_l = line_l.strip("\n").split(" ")
    # I count how many sub sentences will be created from the current sentence 
    number_substrings = math.ceil(len(l_d)/time_step)
    x_substrings = []
    y_substrings = []
    for i in range(number_substrings):
      #for each subsentence I will create a new entry in the dataset
      x_sub = []
      y_sub = []
      for j in range(len(l_d)):
        if j % time_step == 0 and len(x_sub) != 0:
          x_substrings.append(x_sub)
          y_substrings.append(y_sub)
          x_sub = []
          y_sub = []
        x_sub.append(l_d[j])
        y_sub.append(O[l_l[j]])
      x_substrings.append(x_sub)
      y_substrings.append(y_sub)
    for x,y in zip(x_substrings,y_substrings):
      X_training.append(x)
      Y_training.append(y)
  return X_training,Y_training

def create_dataset_synset_boosting_average(path_dataset,path_label,O,w2b,synset_embeddings,pos_tag_path):
  '''
     This function creates the correct dataset for the architecture that makes use of the average of the synset embeddings associated to a word to
     boost fine grain predictions.
     :param path_dataset: the path to the dataset
     :param path_label: the path to the file containing the labels for the task
     :param O: a dictionary containing the mapping from the output vocabulary of the task to unique integers
     :param w2b: a dictionary containing the mapping from wordnet id to babelnet id
     :param synset_embeddings: a dictionary containing the synset embedding
     :param pos_tag_path: a path to thr file containing the pos tag associated to a word
     :returns X_training: a list of list containing at the i-th row the words of the i-th sentence 
     :returns Y_training: a list of list containing at the i-th row the corresponding integer first task labels of the words of the i-th sentence
     :returns X_syn_emb: a list of list containing at the i-th row the average of the embeddings associated to the words of the i-th sentence 
  '''
  X_training = []
  Y_training = []
  X_syn_emb = []
  dataset = open(path_dataset)
  labels_ = open(path_label)
  pos_tag = open(pos_tag_path)
  for line_d,line_l,line_p in zip(dataset,labels_,pos_tag):
    l_d = line_d.strip("\n").split(" ")
    l_l = line_l.strip("\n").split(" ")
    l_p = line_p.strip("\n").split(" ")
    x_t = []
    y_t = []
    avg_emb = []
    for w_d,w_l,p in zip(l_d,l_l,l_p):
      x_t.append(w_d)
      y_t.append(O[w_l])
      synsets = []
      # Here i take the synsets associated to the target word
      if p not in wn_pos_dict:
        synsets =  wn.synsets(w_d, pos=wn.NOUN)
      else:
        synsets =  wn.synsets(w_d, pos=wn_pos_dict[p])
      s_emb = []
      for s in synsets:
        #for every synset found I take its embedding if present in the dictionary otherwise i take the <unk> embedding
        w_id = "wn:" + str(s.offset()).zfill( 8) + s.pos() 
        b_id = w2b[w_id]
        lemma_syn = w_d+"_"+b_id
        if lemma_syn in synset_embeddings:
          s_emb.append(synset_embeddings[lemma_syn])
        else:
          s_emb.append(synset_embeddings["<unk>"])
      if not synsets:
        # if no synset was found then i simply pass the <unk> embedding
        s_emb.append(synset_embeddings["<unk>"])
      s_emb = np.asarray(s_emb)
      # Now i take the average of all the embeddings extrapolated
      avg_emb.append(np.mean(s_emb,axis = 0))
    X_training.append(x_t)
    Y_training.append(y_t)
    X_syn_emb.append(avg_emb)
  return X_training,Y_training, X_syn_emb

def create_dataset_synset_boosting_most_frequent(path_dataset,path_label,O,w2b,synset_embeddings,pos_tag_path):
  '''
     This function creates the correct dataset for the architecture that makes use of the most frequen synset embedding associated to a word to
     boost fine grain predictions. All the considerations made for the previous function can be applied to this function
     :param path_dataset: the path to the dataset
     :param path_label: the path to the file containing the labels for the task
     :param O: a dictionary containing the mapping from the output vocabulary of the task to unique integers
     :param w2b: a dictionary containing the mapping from wordnet id to babelnet id
     :param synset_embeddings: a dictionary containing the synset embedding
     :param pos_tag_path: a path to thr file containing the pos tag associated to a word
     :returns X_training: a list of list containing at the i-th row the words of the i-th sentence 
     :returns Y_training: a list of list containing at the i-th row the corresponding integer first task labels of the words of the i-th sentence
     :returns X_syn_emb: a list of list containing at the i-th row the most frequent synet embeddings associated to the words of the i-th sentence 
     '''
  X_training = []
  Y_training = []
  X_syn_emb = []
  dataset = open(path_dataset)
  labels_ = open(path_label)
  pos_tag = open(pos_tag_path)
  for line_d,line_l,line_p in zip(dataset,labels_,pos_tag):
    l_d = line_d.strip("\n").split(" ")
    l_l = line_l.strip("\n").split(" ")
    l_p = line_p.strip("\n").split(" ")
    x_t = []
    y_t = []
    s_emb = []
    for w_d,w_l,p in zip(l_d,l_l,l_p):
      x_t.append(w_d)
      y_t.append(O[w_l])
      synset = []
      if wn.synsets(w_d):
        if p not in wn_pos_dict or not wn.synsets(w_d, pos=wn_pos_dict[p]):
           synset =  wn.synsets(w_d)[0]
        else:
          synset =  wn.synsets(w_d, pos=wn_pos_dict[p])[0]
        w_id = "wn:" + str(synset.offset()).zfill( 8) + synset.pos() 
        b_id = w2b[w_id]
        lemma_syn = w_d+"_"+b_id
        if lemma_syn in synset_embeddings:
          s_emb.append(synset_embeddings[lemma_syn])
        else:
          s_emb.append(synset_embeddings["<unk>"])
      else:
        # The difference is that if no synset is found then I don't use the <unk> embedding but an embedding of all 0s
        s_emb.append(np.zeros(500))
    X_training.append(x_t)
    Y_training.append(y_t)
    X_syn_emb.append(s_emb)
  return X_training,Y_training, X_syn_emb