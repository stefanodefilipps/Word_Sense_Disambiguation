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
from create_datasets import *
from models import *
from utils import *

def training_multi_learning(path_output_vocab,path_output_vocab2,inverse_vocab_path,inverse_vocab2_path,w2b_path,b2lex_path,lex2idx_path,dataset_path,gold1_path,gold2_path,resources_path):
'''
   This is the function that executes the training of a network exploiting multilearning
   :param path_output_vocab: path to the output dictionary for the first task
   :param path output_vocab2: path to the output dictionary for the second task
   :param inverse_vocab_path: path to the inverse dictionary of the first task
   :param inverse_vocab2_path: path to the inverse dictionray of the second task
   :param w2b_path: path to the dictionary containing the mapping the mapping from wordnet id to babelnet id
   :param b2lex_path: path to the dictionary containing the mapping the mapping from babelnet id to lexicographic domain
   :param lex2idx_path: path to the dictionary containing the mapping the mapping from lexicographic domain to unique integers
   :param dataset_path: path to the dataset
   :param gold1_path: path to the gold file for the first task
   :param gold2_path: path to the gold file for the second task
   :param resource_path: the path to the resource folder
'''
  o = open(path_output_vocab,"rb")
  O = pickle.load(o)
  o_lex = open(path_output_vocab2,"rb")
  O_lex = pickle.load(o_lex)
  inverse_o_lex = open(inverse_vocab2_path,"rb")
  inverse_O_lex = pickle.load(inverse_o_lex)
  w2b = open(w2b_path,"rb")
  W2B = pickle.load(w2b)
  b2lex = open(b2lex_path,"rb")
  B2LEX = pickle.load(b2lex)
  lex2idx = open(lex2idx_path,"rb")
  LEX2IDX = pickle.load(lex2idx)
  HIDDEN_SIZE = 512
  EMBEDDING_SIZE = 1024
  ATTENTION_SIZE = 512
  MAX_TIMESTEP = 30
  epochs = 30
  batch_size = 32
  inputs, labels, labels_lex, keep_prob, loss, loss_fine, loss_lex, train_op, acc, acc_op, acc_lex, acc_op_lex, predictions, predictions_lex, seq_lenght, mask, softmax_out, softmax_out_lex, ts = create_tensorflow_model(HIDDEN_SIZE,EMBEDDING_SIZE,len(O),len(O_lex),ATTENTION_SIZE,resources_path)
  X_training,Y_training,Y_lex_training= create_dataset_multi_learning(
                                        dataset_path,
                                        gold1_path,
                                        gold2_path,
                                        O,
                                        O_lex)
  # Add ops to save and restore all the variables.
  saver = tf.train.Saver()
  n_iterations = int(np.ceil(len(X_training)/batch_size))
  #n_dev_iterations = int(np.ceil(len(X_dev)/batch_size))
  print(n_iterations)
  #history_epoch contains the epoch loss and epoch accuracy at each epoch
  history_epoch= []
  #history_iterations contains the loss and accuracy in each iteration of each epoch
  history_iterations = []
  h_epoch = open('/content/drive/My Drive/homework3_NLP/h_epoch.obj', 'wb')
  pickle.dump(history_epoch, h_epoch)
  h_iterations = open('/content/drive/My Drive/homework3_NLP/h_iterations.obj', 'wb')
  pickle.dump(history_iterations,h_iterations)
  h_epoch.close
  h_iterations.close

  with tf.Session() as sess:
      train_writer = tf.summary.FileWriter('logging/tensorflow_model', sess.graph)
      print("\nStarting training...")
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      c = 0
      X_training = np.asarray(X_training)
      Y_training = np.asarray(Y_training)
      Y_lex_training = np.asarray(Y_lex_training)
      for epoch in range(epochs):
          #print("\nEpoch", epoch + 1)
          bar = tqdm(total=len(X_training))
          epoch_loss, epoch_acc = 0., 0.
          mb = 0
          print("======="*10)
          count = 0
          h_epoch = open('/content/drive/My Drive/homework3_NLP/h_epoch.obj', 'rb')
          history_epoch= pickle.load(h_epoch)
          h_iterations = open('/content/drive/My Drive/homework3_NLP/h_iterations.obj', 'rb')
          history_iterations = pickle.load(h_iterations)
          for batch_x, batch_y, batch_y_lex in batch_generator_multi_learning(X_training, Y_training, Y_lex_training, batch_size):
            sess.run(tf.local_variables_initializer())
            timestep = max_seq(batch_x,False,MAX_TIMESTEP)
            s,m = create_seq_lenght(batch_x,len(batch_x),timestep)
            batch_x = create_sentences_2(batch_x,timestep)
            batch_y = pad_sequences(batch_y, truncating='post', padding='post', maxlen=timestep)
            batch_y_lex = pad_sequences(batch_y_lex, truncating='post', padding='post', maxlen=timestep)
            count += batch_size
            mb += 1
            _, loss_val, loss_val_fine, loss_val_lex, _, _ = sess.run([train_op, loss, loss_fine, loss_lex, acc_op, acc_op_lex], 
                                                  feed_dict={inputs: batch_x, labels: batch_y,labels_lex: batch_y_lex, keep_prob: 0.6, seq_lenght: s, mask: m,ts:timestep })
            # Accumulate loss and acc as we scan through the dataset
            acc_val = sess.run(acc)
            acc_val_lex = sess.run(acc_lex)
            bar.update(batch_size)
            postfix = OrderedDict(epoch=f'{(epoch+1):.4f}',loss=f'{loss_val:.4f}',loss_fine=f'{loss_val_fine:.4f}',loss_lex=f'{loss_val_lex:.4f}', accuracy_fine=f'{acc_val:.4f}',accuracy_lex=f'{acc_val_lex:.4f}')
            bar.set_postfix(postfix)
            epoch_loss += loss_val
            epoch_acc += acc_val
            history_iterations.append([c,loss_val,acc_val])
            c+=1
          bar.close()
          epoch_loss /= n_iterations
          epoch_acc /= n_iterations
          history_epoch.append([epoch,epoch_loss,epoch_acc])
          add_summary(train_writer, "epoch_loss", epoch_loss, epoch)
          add_summary(train_writer, "epoch_acc", epoch_acc, epoch)
          h_epoch.close
          h_epoch = open('/content/drive/My Drive/homework3_NLP/h_epoch.obj', 'wb')
          pickle.dump(history_epoch, h_epoch)
          h_epoch.close
          h_iterations.close
          h_iterations = open('/content/drive/My Drive/homework3_NLP/h_iterations.obj', 'wb')
          pickle.dump(history_iterations,h_iterations)
          print("\n")
          print("Train Loss: {:.4f}\tTrain Accuracy: {:.4f}".format(epoch_loss, epoch_acc))
          print("======="*10)

      train_writer.close()


def training_attention_mechanism(path_output_vocab,inverse_vocab_path,dataset_path,gold1_path,badhanau = True,resources_path):
  '''
   This is the function that executes the training of a network exploiting attention
   :param path_output_vocab: path to the output dictionary for the task
   :param inverse_vocab_path: path to the inverse dictionary of the task
   :param dataset_path: path to the dataset
   :param gold1_path: path to the gold file for the task
   :param badhanau: if true then badhanau attention mechansim is used otherwise the attention mechanism from the Reganato article is used
'''
  o = open(path_output_vocab,"rb")
  O = pickle.load(o)
  HIDDEN_SIZE = 512
  EMBEDDING_SIZE = 1024
  ATTENTION_SIZE = 512
  MAX_TIMESTEP = 30
  epochs = 30
  batch_size = 32
  if badhanau:
    inputs, labels, keep_prob, loss, train_op, acc, acc_op, predictions, seq_lenght, mask, softmax_out, ts = create_tensorflow_model_Badhanau_attention(HIDDEN_SIZE,EMBEDDING_SIZE,len(O),ATTENTION_SIZE,resources_path)
  else:
    inputs, labels, keep_prob, loss, train_op, acc, acc_op, predictions, seq_lenght, mask, softmax_out, ts = create_tensorflow_model_Article_Attention(HIDDEN_SIZE,EMBEDDING_SIZE,len(O),ATTENTION_SIZE,resources_path)
  print(labels)
  X_training,Y_training= create_dataset_basic(
                      dataset_path,
                      gold1_path,
                      O,MAX_TIMESTEP)
  # Add ops to save and restore all the variables.
  saver = tf.train.Saver()
  n_iterations = int(np.ceil(len(X_training)/batch_size))
  #n_dev_iterations = int(np.ceil(len(X_dev)/batch_size))
  print(n_iterations)
  #history_epoch contains the epoch loss and epoch accuracy at each epoch
  history_epoch= []
  #history_iterations contains the loss and accuracy in each iteration of each epoch
  history_iterations = []
  h_epoch = open('/content/drive/My Drive/homework3_NLP/h_epoch.obj', 'wb')
  pickle.dump(history_epoch, h_epoch)
  h_iterations = open('/content/drive/My Drive/homework3_NLP/h_iterations.obj', 'wb')
  pickle.dump(history_iterations,h_iterations)
  h_epoch.close
  h_iterations.close

  with tf.Session() as sess:
      train_writer = tf.summary.FileWriter('logging/tensorflow_model', sess.graph)
      print("\nStarting training...")
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      c = 0
      X_training = np.asarray(X_training)
      Y_training = np.asarray(Y_training)
      for epoch in range(epochs):
          #print("\nEpoch", epoch + 1)
          bar = tqdm(total=len(X_training))
          epoch_loss, epoch_acc = 0., 0.
          mb = 0
          print("======="*10)
          count = 0
          h_epoch = open('/content/drive/My Drive/homework3_NLP/h_epoch.obj', 'rb')
          history_epoch= pickle.load(h_epoch)
          h_iterations = open('/content/drive/My Drive/homework3_NLP/h_iterations.obj', 'rb')
          history_iterations = pickle.load(h_iterations)
          for batch_x, batch_y in batch_generator_basic(X_training, Y_training, batch_size):
            timestep = max_seq(batch_x,False,MAX_TIMESTEP)
            s,m = create_seq_lenght(batch_x,len(batch_x),timestep)
            #batch_x = batch_x.tolist()
            batch_x = create_sentences_2(batch_x,timestep)
            #batch_x = pad_sequences(batch_x, truncating='pre', padding='post', maxlen=timestep)
            batch_y = pad_sequences(batch_y, truncating='post', padding='post', maxlen=timestep)
            count += batch_size
            mb += 1
            _, loss_val, _ = sess.run([train_op, loss, acc_op], 
                                                  feed_dict={inputs: batch_x, labels: batch_y, keep_prob: 0.6, seq_lenght: s, mask: m,ts:timestep })
            # Accumulate loss and acc as we scan through the dataset
            acc_val = sess.run(acc)
            bar.update(batch_size)
            postfix = OrderedDict(epoch=f'{(epoch+1):.4f}',loss=f'{loss_val:.4f}', accuracy=f'{acc_val:.4f}')
            bar.set_postfix(postfix)
            epoch_loss += loss_val
            epoch_acc += acc_val
            history_iterations.append([c,loss_val,acc_val])
            c+=1
          bar.close()
          epoch_loss /= n_iterations
          epoch_acc /= n_iterations
          history_epoch.append([epoch,epoch_loss,epoch_acc])
          add_summary(train_writer, "epoch_loss", epoch_loss, epoch)
          add_summary(train_writer, "epoch_acc", epoch_acc, epoch)
          h_epoch.close
          h_epoch = open('/content/drive/My Drive/homework3_NLP/h_epoch.obj', 'wb')
          pickle.dump(history_epoch, h_epoch)
          h_epoch.close
          h_iterations.close
          h_iterations = open('/content/drive/My Drive/homework3_NLP/h_iterations.obj', 'wb')
          pickle.dump(history_iterations,h_iterations)
          print("\n")
          print("Train Loss: {:.4f}\tTrain Accuracy: {:.4f}".format(epoch_loss, epoch_acc))
          print("======="*10)
          
      train_writer.close()


def training_hierarchical_multi_learning_lex_boosting(path_output_vocab,path_output_vocab2,inverse_vocab_path,inverse_vocab2_path,w2b_path,b2lex_path,lex2idx_path,dataset_path,gold1_path,gold2_path,pos_tag_path,resources_path):
'''
   This is the function that executes the training of a network exploiting a kind of hierarchical modelling
   :param path_output_vocab: path to the output dictionary for the first task
   :param path output_vocab2: path to the output dictionary for the second task
   :param inverse_vocab_path: path to the inverse dictionary of the first task
   :param inverse_vocab2_path: path to the inverse dictionray of the second task
   :param w2b_path: path to the dictionary containing the mapping the mapping from wordnet id to babelnet id
   :param b2lex_path: path to the dictionary containing the mapping the mapping from babelnet id to lexicographic domain
   :param lex2idx_path: path to the dictionary containing the mapping the mapping from lexicographic domain to unique integers
   :param dataset_path: path to the dataset
   :param gold1_path: path to the gold file for the first task
   :param gold2_path: path to the gold file for the second task
   :param pos_tag_path: path to the file containing the pos tag for each word in the sentence
'''

  o = open(path_output_vocab,"rb")
  O = pickle.load(o)
  o_lex = open(path_output_vocab2,"rb")
  O_lex = pickle.load(o_lex)
  inverse_o_lex = open(inverse_vocab2_path,"rb")
  inverse_O_lex = pickle.load(inverse_o_lex)
  w2b = open(w2b_path,"rb")
  W2B = pickle.load(w2b)
  b2lex = open(b2lex_path,"rb")
  B2LEX = pickle.load(b2lex)
  lex2idx = open(lex2idx_path,"rb")
  LEX2IDX = pickle.load(lex2idx)
  HIDDEN_SIZE = 512
  EMBEDDING_SIZE = 1024
  ATTENTION_SIZE = 512
  MAX_TIMESTEP = 30
  epochs = 30
  batch_size = 32
  inputs, labels, labels_lex, keep_prob, loss_fine, loss_lex, train_op_fine,train_op_lex,acc, acc_op, acc_lex, acc_op_lex, predictions, seq_lenght, mask, softmax_out, softmax_out_lex, predictions_lex, ts, lex_codes, prediction_lex_codes = create_tensorflow_model(HIDDEN_SIZE,EMBEDDING_SIZE,len(O),len(O_lex),ATTENTION_SIZE,resources_path)
  X_training,Y_training,X_lex_cod,Y_lex_training= create_dataset_multi_learning_lex_boosting(
                      dataset_path,
                      gold1_path,
                      gold2_path,
                      O,O_lex,W2B,B2LEX,LEX2IDX,
                      pos_tag_path)
  # Add ops to save and restore all the variables.
  saver = tf.train.Saver()
  n_iterations = int(np.ceil(len(X_training)/batch_size))
  #n_dev_iterations = int(np.ceil(len(X_dev)/batch_size))
  print(n_iterations)
  #history_epoch contains the epoch loss and epoch accuracy at each epoch
  history_epoch= []
  #history_iterations contains the loss and accuracy in each iteration of each epoch
  history_iterations = []
  h_epoch = open('/content/drive/My Drive/homework3_NLP/h_epoch.obj', 'wb')
  pickle.dump(history_epoch, h_epoch)
  h_iterations = open('/content/drive/My Drive/homework3_NLP/h_iterations.obj', 'wb')
  pickle.dump(history_iterations,h_iterations)
  h_epoch.close
  h_iterations.close

  with tf.Session() as sess:
      train_writer = tf.summary.FileWriter('logging/tensorflow_model', sess.graph)
      print("\nStarting training...")
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      c = 0
      X_training = np.asarray(X_training)
      Y_training = np.asarray(Y_training)
      Y_lex_training = np.asarray(Y_lex_training)
      X_lex_cod = np.asarray(X_lex_cod)
      for epoch in range(epochs):
          #print("\nEpoch", epoch + 1)
          bar = tqdm(total=len(X_training))
          epoch_loss, epoch_acc = 0., 0.
          mb = 0
          print("======="*10)
          count = 0
          h_epoch = open('/content/drive/My Drive/homework3_NLP/h_epoch.obj', 'rb')
          history_epoch= pickle.load(h_epoch)
          h_iterations = open('/content/drive/My Drive/homework3_NLP/h_iterations.obj', 'rb')
          history_iterations = pickle.load(h_iterations)
          for batch_x, batch_y, batch_y_lex, batch_x_cod in batch_generator_hierarchical_multi_learning_lex_boosting(X_training, Y_training, Y_lex_training,X_lex_cod, batch_size):
            sess.run(tf.local_variables_initializer())
            timestep = max_seq(batch_x,False,MAX_TIMESTEP)
            s,m = create_seq_lenght(batch_x,len(batch_x),timestep)
            batch_x = create_sentences_2(batch_x,timestep)
            batch_y = pad_sequences(batch_y, truncating='post', padding='post', maxlen=timestep)
            batch_y_lex = pad_sequences(batch_y_lex, truncating='post', padding='post', maxlen=timestep)
            batch_x_cod = pad_sequences(batch_x_cod, truncating='post', padding='post', maxlen=timestep)
            count += batch_size
            mb += 1
            # I first train the lexicographic task
            _, loss_val_lex, _ = sess.run([train_op_lex, loss_lex, acc_op_lex], 
                                                  feed_dict={inputs: batch_x, labels_lex: batch_y_lex, keep_prob: 0.6, seq_lenght: s, mask: m,ts:timestep })
            acc_val_lex = sess.run(acc_lex)
            # Then i get the lexicographic predictions and their relative encoding
            predict_lex = prediction_lex(batch_x,s, m,inverse_O_lex,O_lex,predictions_lex,LEX2IDX,timestep,batch_x_cod,sess)
            #Then i need to train the fine grained task passing the encoding of the predictions just retrieved
            _, loss_val_fine, _ = sess.run([train_op_fine, loss_fine, acc_op], 
                                                  feed_dict={inputs: batch_x, lex_codes: batch_x_cod, prediction_lex_codes: predict_lex, labels: batch_y, keep_prob: 0.6, seq_lenght: s, mask: m,ts:timestep })
            
            # Accumulate loss and acc as we scan through the dataset
            acc_val_fine = sess.run(acc)
            bar.update(batch_size)
            postfix = OrderedDict(epoch=f'{(epoch+1):.4f}',loss_fine=f'{loss_val_fine:.4f}',loss_lex=f'{loss_val_lex:.4f}', accuracy_fine=f'{acc_val_fine:.4f}',accuracy_lex=f'{acc_val_lex:.4f}')
            bar.set_postfix(postfix)
            #print(bar)
            epoch_loss += loss_val_fine
            epoch_acc += acc_val_fine
            history_iterations.append([c,loss_val_fine,acc_val_fine])
            c+=1
          bar.close()
          epoch_loss /= n_iterations
          epoch_acc /= n_iterations
          history_epoch.append([epoch,epoch_loss,epoch_acc])
          add_summary(train_writer, "epoch_loss", epoch_loss, epoch)
          add_summary(train_writer, "epoch_acc", epoch_acc, epoch)
          h_epoch.close
          h_epoch = open('/content/drive/My Drive/homework3_NLP/h_epoch.obj', 'wb')
          pickle.dump(history_epoch, h_epoch)
          h_epoch.close
          h_iterations.close
          h_iterations = open('/content/drive/My Drive/homework3_NLP/h_iterations.obj', 'wb')
          pickle.dump(history_iterations,h_iterations)
          print("\n")
          print("Train Loss: {:.4f}\tTrain Accuracy: {:.4f}".format(epoch_loss, epoch_acc))
          print("======="*10)     
                                  
      train_writer.close()