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

wn_pos_dict = {"VERB":wn.VERB,"ADJ":wn.ADJ,"ADV":wn.ADV,"NOUN":wn.NOUN}

def batch_generator_basic(X, Y, batch_size, shuffle=True):
    if not shuffle:
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            x = X[start:end]
            yield X[start:end], Y[start:end]
    else:
        perm = np.random.permutation(len(X))
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            yield X[perm[start:end]], Y[perm[start:end]]

def batch_generator_hierarchical_multi_learning_lex_boosting(X, Y, Y_lex, X_lex_cod, batch_size, shuffle=True):
    if not shuffle:
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            x = X[start:end]
            yield X[start:end], Y[start:end], Y_lex[start:end], X_lex_cod[start:end]
    else:
        perm = np.random.permutation(len(X))
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            yield X[perm[start:end]], Y[perm[start:end]], Y_lex[perm[start:end]], X_lex_cod[perm[start:end]]

def batch_generator_multi_learning(X, Y, Y_lex, batch_size, shuffle=True):
    if not shuffle:
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            x = X[start:end]
            yield X[start:end], Y[start:end], Y_lex[start:end]
    else:
        perm = np.random.permutation(len(X))
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            yield X[perm[start:end]], Y[perm[start:end]], Y_lex[perm[start:end]]

def add_summary(writer, name, value, global_step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
    writer.add_summary(summary, global_step=global_step)
    
def max_acc(h):
  '''
      This function is used in order to find the best model accuracy in the past data for the developement set in order to know if to save the model after
      the ending of an epoch based on the best performace so far,

      :param h: a list containing accuracy metrics
      :return m: the best value in the list 
  ''' 
  m = 0
  for v in h:
    if v[2] > m:
      m = v[2]
  return m
    
def max_seq(x,val,max_t):
  '''
      This function is used in order to find the timestep to use in a batch.

      :param x: the batch of (batch_size,seq_lenght_i,ch_emb+big_emb)
      :param val: a boolean flag. If false it means we are in the training set so i have to truncate all the sentences to a set leght in this case 100.
                  If true than we are evaluating the developement set so I return the length of the longest sentence in the batch.
      :return m: the value containing the timestep to use in the current batch.
  '''
  m = 0
  if not val:
    return max_t
  for r in x:
    if len(r) > m:
      m = len(r)
  return m

def create_seq_lenght(x,batch_size,timestep):
  '''
      This function finds the actual lenght of each sequence in the batch and it also creates a mask to avoid the padding.

      :param x: the current batch.
      :param batch_size: the size of the current batch.
      :param timestep: the maixmum numer of timestep in the sequence for the current batch.
      :return seq_lenght, mask: seq_lenght has (batch_size,1) shape and contains integer.
                                mask has (batch_size,timestep,1) shape and contains only 1 or 0 based on the fact that we need to consider or not that element in the sequence
  '''
  seq_lenght = np.zeros(batch_size)
  mask = np.zeros((batch_size,timestep))
  for s_ind,s in enumerate(x):
    if len(x[s_ind]) > timestep:
           seq_lenght[s_ind] = timestep
    else:
           seq_lenght[s_ind] = len(x[s_ind])
    mask[s_ind][0:int((seq_lenght[s_ind]-1))] = 1
  return seq_lenght, mask
      
def create_sentences(x):
  '''
      This function is used to create a correct input to the elmo module.
  '''
  l = []
  for e in x:
    l.append(" ".join(e))
  return l

def create_sentences_2(x,timestep):
  """
  This function is used in order to have the correct format to pass to elmo architecture and extrapolate the word embeddings. So if a sentenec is shorter
      than col it is padded with ""
      :param x: current batch
      :param col: max length in the batch
      :return l: a list of list containing the eventually padded sentences
  """
  l = []
  for e in x:
    if len(e) > timestep:
      t = []
      for i in range(timestep):
        t.append(e[i])
      l.append(t)
      continue
    while len(e) < timestep:
      e.append("")
    l.append(e)
  return l

def get_sense_key(lemma,pos):
  '''
      This function is used to retrieve the wordnet sense key given the pai lemma pos
  '''
  l = wn.synsets(lemma, pos = wn_pos_dict[pos])[0].lemmas()
  for e in l:
    if e.name() == lemma:
      return e._key

def encoding_most_frequent_lex(lemma,pos,w2b,b2lex,lex2idx):
  """
    This function retrieves the most frequent lexographic domain given a couple lemma pos and create the correct binary encoding
    :param lemma: the lemma of the target word
    :param pos: the pos tag of the target word
    :param b2lex: a dictionary containing the mapping from the babelnet id to the lexicographic domains
    :param w2b: a dictionary containing the mapping from wordnet id to babelnet id
    :param lex2idx: a dictionary containing the mapping from the lexicographic domains to unique integers
    :return the binary encoding of the lexicographic domain
  """
  # For each target word I need to create the binary encoding for the most frequent lexicographic domain, needed for the fine grained predictions
  synset = []
  if wn.synsets(lemma):
      # if the pos tag of the target word is not one of the accepted ones from wordnet or there is no associated sysnet to the couple lemma pos
      # then i simply get the synsets associated to the lemma and get the first one
      if pos not in wn_pos_dict or not wn.synsets(lemma, pos=wn_pos_dict[pos]):
         synset =  wn.synsets(lemma)[0]
      else:
        synset =  wn.synsets(lemma, pos=wn_pos_dict[pos])[0]
      # from the synset I can retrieve the lex domain and create the correct binary encoding
      w_id = "wn:" + str(synset.offset()).zfill( 8) + synset.pos() 
      b_id = w2b[w_id]
      lex_domain = b2lex[b_id]
      lex_one_hot = np.zeros(45)
      lex_one_hot[lex2idx[lex_domain]] = 1
      return lex_one_hot
      lex_cod.append(lex_one_hot)
  else:
      return np.zeros(45)

def prediction_lex_predict(sentence,O,inverse_O,prob_lex,b2lex,w2b,lex2idx,annotations):
  '''
    This is the function used in the prediction of the network architecture that implements a sort of hierarchical modelling.
    It takes the lexicographic predictions and creates the lexicographic binary encodings that are used in the prediction of the 
    fine grained task.
    :param sentence: the target sentence to predict
    :param O: the output vocabulary for the lexicographic task
    :param inverse_O: the inverse output vocabulary for the lexigraphic task
    :param prob_lex: the matrices containing the network softmax outputs for the lexicographic task
    :param b2lex: a dictionary containing the mapping from the babelnet id to the lexicographic domains
    :param w2b: a dictionary containing the mapping from wordnet id to babelnet id
    :param lex2idx: a dictionary containing the mapping from the lexicographic domains to unique integers
    :annotations: a dictionary containing informations about the word that were annotated and for which I need to supply a prediction
    :return lex_codes: an array contaiing the encoding of the lexicographic domains predicted for the annotated words
  '''
  prob = prob_lex
  lex_codes = []
  s_split = sentence.split(" ")
  for idx,w in enumerate(s_split):
    # I need to scan the whole sentence but I need to tract separetely the case of annotated words and not annotated words
    # If annotated I need  to get the the most probable prediction from the list of candidates (as it happens for the fine grained predictions)
    # If the target word is not annotated then I simply need to get the prediction and if it is a lexicographic domain then I create the encoding
    # otherwise I need to retrieve the most frequent lexicographic domain and create the binary encoding
    if idx in annotations:
      # it means that it was an annotated word so for evaluation of the softmax i use the standard approach
      # The following logic is the same of the prediction of annotated words for the fine grained task
      if annotations[idx][2] != None:
        c = annotations[idx][2]
        prediction = c[np.argmax(prob[0][0][idx][c])]
        lex_domain = inverse_O[prediction]
        lex_code = np.zeros(45)
        lex_code[lex2idx[lex_domain]] = 1
        lex_codes.append(lex_code)
      else:
        lex_code = np.zeros(45)
        lex_code[lex2idx[annotations[idx][1]]] = 1
        lex_codes.append(lex_code)
    else:
      #it means the word was not annotated so i simply get the max over the softmax and build the correct encoding
      m = np.argmax(prob[0][0][idx])
      prediction = m
      prediction = inverse_O[prediction]
      if prediction in lex2idx:
        #it means that the prediction is a lexicographic domain
        temp = np.zeros(45)
        temp[lex2idx[prediction]] = 1
        lex_codes.append(temp)
      else:
        #it means that the prediction is a word of the input vocabulary so i try to extrapolate the most frequent lexographic domain
        #using wordnet, otherwise all 0s
        #all this informations are already stored in the X_lex_cod matrix created during the creation of the dataset
        synset = []
        if wn.synsets(w):
          synset =  wn.synsets(w)[0]
          w_id = "wn:" + str(synset.offset()).zfill( 8) + synset.pos() 
          b_id = w2b[w_id]
          lex_domain = b2lex[b_id]
          lex_one_hot = (np.zeros(45))
          lex_one_hot[lex2idx[lex_domain]] = 1
          lex_codes.append(lex_one_hot)
        else:
          lex_codes.append(np.zeros(45))
  return lex_codes

def prediction_lex(batch_x,s_len, m,inverse_O,O,predictions_lex,lex2idx,time_s,batch_x_lex_cod,sess):
  '''
      This function is used in the architecture that employes a some sort of hierarchical learning and it computes the binary encoding of the lexicographic predictions
      :param batch_x: the current batch
      :param s_len: the mex timestep
      :param m: the mask to ignore the padding
      :param inverse_O: a dictionary containing the mapping from integers to unique elements in the output vocabulary of the lexicographic task
      :param O: a dictionary containing the mapping from the output vocabulary of the lexicographic task to unique integers.
      :param predictions_lex: the tensorflow variable to be run in order thìo get the lexicographic prediction from the network
      :param lex2idx: a dictionary containing the mapping from the lexicographic domain to unique integers.
      :param batch_x_lex_cod: a numpy matrix containing the most frequent lexicographic domain of the batch words.
      :param sess: the currently executing tensorflow session
      :return: a numpy matrix containing the encoding of the lexicographic predictions
  '''
  #first i get the lexicographic predictions
  predictions = sess.run([predictions_lex],feed_dict = {inputs: batch_x, keep_prob: 1.0, seq_lenght: s_len, mask: m,ts: time_s})
  predictions = np.asarray(predictions)
  pred_cod = []
  for i in range(predictions.shape[1]):
    lex_cod = []
    for j in range(predictions.shape[2]):
      # Then from the prediction which is an integer I get the corresponding string which can be either a lexicographic domain or a word of the input vocabulary
      predict = predictions[0][i][j]
      predict = inverse_O[predict]
      if predict in lex2idx:
        #it means that the prediction is a lexicographic domain and I get the encoding
        temp = np.zeros(45)
        temp[lex2idx[predict]] = 1
        lex_cod.append(temp)
      else:
        #it means that the prediction is a word of the input vocabulary so i try to extrapolate the most frequent lexographic domain
        #using wordnet, otherwise all 0s
        #all this informations are already stored in the X_lex_cod matrix created during the creation of the dataset
        lex_cod.append(batch_x_lex_cod[i][j])
    pred_cod.append(lex_cod)
  return np.asarray(pred_cod)