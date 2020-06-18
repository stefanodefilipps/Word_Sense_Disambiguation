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

def ELMoEmbedding(x,seq_len,elmo):
    return elmo(inputs = {"tokens":x,"sequence_len":seq_len}, signature="tokens", as_dict=True)["elmo"]

def create_tensorflow_model(hidden_size,embedding_size,output_size,output_lex_size,attention_size,resource_path):

    '''
        This is the model where multilearning and attention mechanism are employed. It takes 2 tasks and compute the wo losses and the optimizer
        trains a loss which is the average of the 2 losses.
        :param hidden_size: number of units of the bidirectional LSTM
        :param embedding_size: the size of the input word embedding
        :param output_size: the size of output vocabulary of first task
        :param output_lex_size: the size of the output vocabulary of the second task
        :param resource_path: the path to the resource folder
        :returns: all the placeholders needed for the training and prediction
    '''
    elmo = hub.Module(resource_path, trainable=True)
    print("Creating TENSORFLOW model")
     
    # Inputs have (batch_size, timesteps) shape.
    inputs = tf.placeholder(tf.string, shape=[None, None])
    # Labels have (batch_size,) shape.
    labels = tf.placeholder(tf.int64, shape=[None, None])
    # Labels have (batch_size,) shape.
    labels_lex = tf.placeholder(tf.int64, shape=[None, None])
    # Keep_prob is a scalar.
    keep_prob = tf.placeholder(tf.float32, shape=[])
    
    timestep = tf.placeholder(tf.float32, shape=[])
    
    #this is the mask for every batch and computed differently in every batch
    # Calculate sequence lengths to mask out the paddings later on.
    
    seq_lenght = tf.placeholder(tf.int32, shape = [None,])
      
    mask = tf.placeholder(tf.float32, shape = [None,None])

    #Here I compute the embedding of the words in input
    
    embeddings = ELMoEmbedding(inputs,seq_lenght,elmo)
    

    with tf.variable_scope("BiLSTM"):
        cell_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size,dropout_keep_prob = keep_prob)

        cell_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size, dropout_keep_prob = keep_prob)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                          cell_bw, 
                                                          embeddings,
                                                          sequence_length = seq_lenght,
                                                          dtype=tf.float32)
        #this time I only need the output of the second layer of the BLSTM
        outputs = tf.concat(outputs,2)
        state = states[1][0]
        print("STATES")
        print(states[1])
        
    with tf.variable_scope("attention"):
        # state shape == (batch_size, hidden size)
        # state_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        state_with_time_axis = tf.expand_dims(state, 1)
        W1 = tf.layers.dense(outputs, attention_size, activation=None)
        W2 = tf.layers.dense(state_with_time_axis, attention_size, activation=None)
        tanh_input = W1+W2
        V_input = tf.nn.tanh(tanh_input)
        # score shape == (batch_size, time_step, 1) 
        # the shape of the tensor before applying score is (batch_size, max_length, attention_size)
        score = tf.layers.dense(V_input,1,activation = None)
        # attention_weights shape == (batch_size, max_length, 1)
        # Softmax by default is applied on the last axis but here we want to apply it on the 1st axis, since the shape of score 
        # is (batch_size, max_length, hidden_size). Max_length is the length of our input. Since we are trying to assign a weight to each input, 
        # softmax should be applied on that axis.'''
        attention_weights = tf.nn.softmax(score, axis=1)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * outputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        context_vector = tf.expand_dims(context_vector,axis = 1) #add a dimension
        context_vector = tf.tile(context_vector,multiples = [1,timestep,1]) #duplicate in this dimension
        outputs = tf.concat([outputs,context_vector],axis=-1) #concatentae on innermost dimension
        

    with tf.variable_scope("dense"):
        # TensorFlow performs the sigmoid in the loss function for efficiency,
        # so don't use any activation on last dense.
        #dropout = tf.layers.dropout(outputs,1-keep_prob)
        logits = tf.layers.dense(outputs, output_size, activation=None)
        logits_lex = tf.layers.dense(outputs, output_lex_size, activation=None)

    with tf.variable_scope("loss"):
        # Use the contrib sequence loss and average over the batches
        #instead of tf.ones i need to define a correct mask in order to eliminate the padding in the loss computation 
        #maybe using tf.sequence_mask
        loss_fine = tf.contrib.seq2seq.sequence_loss(
                    logits,
                    labels,
                    mask,
                    average_across_timesteps=True,
                    average_across_batch=True)
        
        loss_lex = tf.contrib.seq2seq.sequence_loss(
                    logits_lex,
                    labels_lex,
                    mask,
                    average_across_timesteps=True,
                    average_across_batch=True)
        
        loss = 0.5*(loss_fine + loss_lex)
        

    with tf.variable_scope("train"):
        optimize = tf.train.AdamOptimizer(0.02).minimize(loss)
        #actually during evalutaion i will need to have the result directly of the softmax and then i need to manipulate it
        softmax_out = tf.nn.softmax(logits)
        softmax_out_lex = tf.nn.softmax(logits_lex)
        predictions_lex = tf.cast(tf.argmax(softmax_out_lex, axis=-1), tf.int64)
        #Remeber that this time the prediction during evaluation must be done in a peculiar way
        predictions = tf.cast(tf.argmax(softmax_out, axis=-1), tf.int64)
        acc,acc_op = tf.metrics.accuracy(labels,predictions,weights = mask)
        acc_lex,acc_op_lex = tf.metrics.accuracy(labels_lex,predictions_lex,weights = mask)
        

    return inputs, labels, labels_lex, keep_prob, loss, loss_fine, loss_lex, optimize, acc, acc_op, acc_lex, acc_op_lex, predictions, predictions_lex, seq_lenght, mask, softmax_out, softmax_out_lex, timestep


def create_tensorflow_model_Badhanau_attention(hidden_size,embedding_size,output_size,attention_size,resource_path):

    '''
        This is the model where attention mechanism is employed. 
        :param hidden_size: number of units of the bidirectional LSTM
        :param embedding_size: the size of the input word embedding
        :param output_size: the size of output vocabulary of first task
        :param resource_path: the path to the resource folder
        :returns: all the placeholders needed for the training and prediction
    '''
    elmo = hub.Module(resource_path, trainable=True)
    print("Creating TENSORFLOW model")
     
    # Inputs have (batch_size, timesteps) shape.
    inputs = tf.placeholder(tf.string, shape=[None, None])
    # Labels have (batch_size,) shape.
    labels = tf.placeholder(tf.int64, shape=[None, None])
    # Keep_prob is a scalar.
    keep_prob = tf.placeholder(tf.float32, shape=[])
    
    timestep = tf.placeholder(tf.float32, shape=[])
    
    #this is the mask for every batch and computed differently in every batch
    # Calculate sequence lengths to mask out the paddings later on.
    
    seq_lenght = tf.placeholder(tf.int32, shape = [None,])
      
    mask = tf.placeholder(tf.float32, shape = [None,None])
    
    embeddings = ELMoEmbedding(inputs,seq_lenght,elmo)
    

    with tf.variable_scope("BiLSTM"):
        cell_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size,dropout_keep_prob = keep_prob)

        cell_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size, dropout_keep_prob = keep_prob)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                          cell_bw, 
                                                          embeddings,
                                                          sequence_length = seq_lenght,
                                                          dtype=tf.float32)
        #this time I only need the output of the second layer of the BLSTM
        outputs = tf.concat(outputs,2)
        state = states[1][0]
        print("STATES")
        print(states[1])
        
    with tf.variable_scope("attention"):
        # state shape == (batch_size, hidden size)
        # state_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        state_with_time_axis = tf.expand_dims(state, 1)
        W1 = tf.layers.dense(outputs, attention_size, activation=None)
        W2 = tf.layers.dense(state_with_time_axis, attention_size, activation=None)
        tanh_input = W1+W2
        V_input = tf.nn.tanh(tanh_input)
        # score shape == (batch_size, time_step, 1) 
        # the shape of the tensor before applying score is (batch_size, max_length, attention_size)
        score = tf.layers.dense(V_input,1,activation = None)
        # attention_weights shape == (batch_size, max_length, 1)
        # Softmax by default is applied on the last axis but here we want to apply it on the 1st axis, since the shape of score 
        # is (batch_size, max_length, hidden_size). Max_length is the length of our input. Since we are trying to assign a weight to each input, 
        # softmax should be applied on that axis.'''
        attention_weights = tf.nn.softmax(score, axis=1)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * outputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        context_vector = tf.expand_dims(context_vector,axis = 1) #add a dimension
        context_vector = tf.tile(context_vector,multiples = [1,timestep,1]) #duplicate in this dimension
        outputs = tf.concat([outputs,context_vector],axis=-1) #concatentae on innermost dimension
        

    with tf.variable_scope("dense"):
        # TensorFlow performs the sigmoid in the loss function for efficiency,
        # so don't use any activation on last dense.
        #dropout = tf.layers.dropout(outputs,1-keep_prob)
        logits = tf.layers.dense(outputs, output_size, activation=None)

    with tf.variable_scope("loss"):
        # Use the contrib sequence loss and average over the batches
        #instead of tf.ones i need to define a correct mask in order to eliminate the padding in the loss computation 
        #maybe using tf.sequence_mask
        loss = tf.contrib.seq2seq.sequence_loss(
                    logits,
                    labels,
                    mask,
                    average_across_timesteps=True,
                    average_across_batch=True)
        

    with tf.variable_scope("train"):
        optimize = tf.train.AdamOptimizer(0.02).minimize(loss)
        #actually during evalutaion i will need to have the result directly of the softmax and then i need to manipulate it
        softmax_out = tf.nn.softmax(logits)
        #Remeber that this time the prediction during evaluation must be done in a peculiar way
        predictions = tf.cast(tf.argmax(softmax_out, axis=-1), tf.int64)
        acc,acc_op = tf.metrics.accuracy(labels,predictions,weights = mask)
        

    return inputs, labels, keep_prob, loss, optimize,acc, acc_op, predictions, seq_lenght, mask, softmax_out, timestep

def create_tensorflow_model_Article_Attention(hidden_size,embedding_size,output_size,attention_size,resource_path):

    '''
        This is the model where multilearning and attention mechanism from the Reganato article is employed.
        :param hidden_size: number of units of the bidirectional LSTM
        :param embedding_size: the size of the input word embedding
        :param output_size: the size of output vocabulary of first task
        :param resource_path: the path to the resource folder
        :returns: all the placeholders needed for the training and prediction
    '''
    elmo = hub.Module(resource_path, trainable=True)
    print("Creating TENSORFLOW model")
     
    # Inputs have (batch_size, timesteps) shape.
    inputs = tf.placeholder(tf.string, shape=[None, None])
    # Labels have (batch_size,) shape.
    labels = tf.placeholder(tf.int64, shape=[None, None])
    # Keep_prob is a scalar.
    keep_prob = tf.placeholder(tf.float32, shape=[])
    
    timestep = tf.placeholder(tf.float32, shape=[])
    
    # Calculate sequence lengths to mask out the paddings later on.
    
    keep_prob2 = tf.placeholder(tf.float32, shape=[])
    
    #this is the mask for every batch and computed differently in every batch
    # Calculate sequence lengths to mask out the paddings later on.
    
    seq_lenght = tf.placeholder(tf.int32, shape = [None,])
      
    mask = tf.placeholder(tf.float32, shape = [None,None])
    
    embeddings = ELMoEmbedding(inputs,seq_lenght,elmo)
    

    with tf.variable_scope("BiLSTM"):
        cell_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size,dropout_keep_prob = keep_prob)

        cell_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size, dropout_keep_prob = keep_prob)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                          cell_bw, 
                                                          embeddings,
                                                          sequence_length = seq_lenght,
                                                          dtype=tf.float32)
        #this time I only need the output of the second layer of the BLSTM
        outputs = outputs[1]
        print("STATES")
        print(states[1])
        
    with tf.variable_scope("attention"):
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(outputs, w_omega, axes=1) + b_omega)
        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape
        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        c = tf.reduce_sum(outputs * tf.expand_dims(alphas, -1), 1)
        c_exp = tf.expand_dims(c,axis = 1) #add a dimension
        c_exp = tf.tile(c_exp,multiples = [1,timestep,1]) #duplicate in this dimension
        outputs = tf.concat([outputs,c_exp],axis=-1) #concatentae on innermost dimension
        

    with tf.variable_scope("dense"):
        # TensorFlow performs the sigmoid in the loss function for efficiency,
        # so don't use any activation on last dense.
        #dropout = tf.layers.dropout(outputs,1-keep_prob)
        logits = tf.layers.dense(outputs, output_size, activation=None)

    with tf.variable_scope("loss"):
        # Use the contrib sequence loss and average over the batches
        #instead of tf.ones i need to define a correct mask in order to eliminate the padding in the loss computation 
        #maybe using tf.sequence_mask
        loss = tf.contrib.seq2seq.sequence_loss(
                    logits,
                    labels,
                    mask,
                    average_across_timesteps=True,
                    average_across_batch=True)

    with tf.variable_scope("train"):
        optimize = tf.train.AdamOptimizer(0.02).minimize(loss)
        #actually during evalutaion i will need to have the result directly of the softmax and then i need to manipulate it
        softmax_out = tf.nn.softmax(logits)
        #Remeber that this time the prediction during evaluation must be done in a peculiar way
        predictions = tf.cast(tf.argmax(softmax_out, axis=-1), tf.int64)
        acc,acc_op = tf.metrics.accuracy(labels,predictions,weights = mask)
        

    return inputs, labels, keep_prob, loss, optimize,acc, acc_op, predictions, seq_lenght, mask, keep_prob2, softmax_out, timestep

def create_tensorflow_model_hierarchical_multi_learning(hidden_size,embedding_size,output_size,output_lex_size,attention_size,resource_path):

    '''
        This is the model where a kind of hierarchical multilearning and attention mechanism are employed. It takes 2 tasks and compute 2 different losses.
        At each training step the second task is optimized first and its predictions are concatenated to the context vector and other additional
        information as final input for the layer of the first task and finally it is optimized.
        :param hidden_size: number of units of the bidirectional LSTM
        :param embedding_size: the size of the input word embedding
        :param output_size: the size of output vocabulary of first task
        :param output_lex_size: the size of the output vocabulary of the second task
        :param resource_path: the path to the resource folder
        :returns: all the placeholders needed for the training and prediction
    '''
    elmo = hub.Module(resource_path, trainable=True)
    print("Creating TENSORFLOW model")
     
    # Inputs have (batch_size, timesteps) shape.
    inputs = tf.placeholder(tf.string, shape=[None, None])
    # Labels have (batch_size,) shape.
    labels = tf.placeholder(tf.int64, shape=[None, None])
    # Labels of the second task
    labels_lex = tf.placeholder(tf.int64, shape=[None, None])
    # These are the most frequent lexographic domain of the input word which are concatenate to the input of the final layer of the first task
    # It has shape (batch_size,timesteps,45)
    lex_codes = tf.placeholder(tf.float32,shape = [None,None,45])
    # These are the lexographic domain predictions of the input word which are concatenate along with the most frequent lexographic domain to the 
    # input of the final layer of the first task
    # It has shape (batch_size,timesteps,45)
    prediction_lex_codes = tf.placeholder(tf.float32,shape = [None,None,45])
    # Keep_prob is a scalar.
    keep_prob = tf.placeholder(tf.float32, shape=[])
    
    timestep = tf.placeholder(tf.float32, shape=[])
    
    #this is the mask for every batch and computed differently in every batch
    # Calculate sequence lengths to mask out the paddings later on.
    
    seq_lenght = tf.placeholder(tf.int32, shape = [None,])
      
    mask = tf.placeholder(tf.float32, shape = [None,None])
    
    embeddings = ELMoEmbedding(inputs,seq_lenght,elmo)
    

    with tf.variable_scope("BiLSTM"):
        cell_fw = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size,dropout_keep_prob = keep_prob)

        cell_bw = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size, dropout_keep_prob = keep_prob)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                          cell_bw, 
                                                          embeddings,
                                                          sequence_length = seq_lenght,
                                                          dtype=tf.float32)
        #this time I only need the output of the second layer of the BLSTM
        outputs = tf.concat(outputs,2)
        state = states[1][0]
        print("STATES")
        print(states[1])
        
    with tf.variable_scope("attention"):
        # state shape == (batch_size, hidden size)
        # state_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        state_with_time_axis = tf.expand_dims(state, 1)
        W1 = tf.layers.dense(outputs, attention_size, activation=None)
        W2 = tf.layers.dense(state_with_time_axis, attention_size, activation=None)
        tanh_input = W1+W2
        V_input = tf.nn.tanh(tanh_input)
        # score shape == (batch_size, time_step, 1) 
        # the shape of the tensor before applying score is (batch_size, max_length, attention_size)
        score = tf.layers.dense(V_input,1,activation = None)
        # attention_weights shape == (batch_size, max_length, 1)
        # Softmax by default is applied on the last axis but here we want to apply it on the 1st axis, since the shape of score 
        # is (batch_size, max_length, hidden_size). Max_length is the length of our input. Since we are trying to assign a weight to each input, 
        # softmax should be applied on that axis.'''
        attention_weights = tf.nn.softmax(score, axis=1)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * outputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        context_vector = tf.expand_dims(context_vector,axis = 1) #add a dimension
        context_vector = tf.tile(context_vector,multiples = [1,timestep,1]) #duplicate in this dimension
        outputs = tf.concat([outputs,context_vector],axis=-1) #concatentae on innermost dimension
        # the input to the final layer of the first fine grained task takes additional information with respect the input of the final layer of
        # the lexicographic task which are the most frequent lexicographic domains of the input words and the lexicographic predictions
        outputs_fine = tf.concat([outputs,lex_codes,prediction_lex_codes],axis=-1)
        

    with tf.variable_scope("dense"):
        # TensorFlow performs the sigmoid in the loss function for efficiency,
        # so don't use any activation on last dense.
        # dropout = tf.layers.dropout(outputs,1-keep_prob)
        logits = tf.layers.dense(outputs_fine, output_size, activation=None)
        logits_lex = tf.layers.dense(outputs, output_lex_size, activation=None)

    with tf.variable_scope("loss"):
        # Use the contrib sequence loss and average over the batches
        #instead of tf.ones i need to define a correct mask in order to eliminate the padding in the loss computation 
        #maybe using tf.sequence_mask
        loss_fine = tf.contrib.seq2seq.sequence_loss(
                    logits,
                    labels,
                    mask,
                    average_across_timesteps=True,
                    average_across_batch=True)
        
        loss_lex = tf.contrib.seq2seq.sequence_loss(
                    logits_lex,
                    labels_lex,
                    mask,
                    average_across_timesteps=True,
                    average_across_batch=True)
        
        loss = 0.5*(loss_fine + loss_lex)
        

    with tf.variable_scope("train"):
        #Now I have 2 optimizer becasuse the optimization of the 2 losses doesn't happen at the same time
        optimize_fine = tf.train.AdamOptimizer(0.02).minimize(loss_fine)
        optimize_lex = tf.train.AdamOptimizer(0.02).minimize(loss_lex)
        #optimize = tf.train.AdamOptimizer(0.02).minimize(loss)
        #actually during evalutaion i will need to have the result directly of the softmax and then i need to manipulate it
        softmax_out = tf.nn.softmax(logits)
        softmax_out_lex = tf.nn.softmax(logits_lex)
        predictions_lex = tf.cast(tf.argmax(softmax_out_lex, axis=-1), tf.int64)
        #Remeber that this time the prediction during evaluation must be done in a peculiar way
        predictions = tf.cast(tf.argmax(softmax_out, axis=-1), tf.int64)
        acc,acc_op = tf.metrics.accuracy(labels,predictions,weights = mask)
        acc_lex,acc_op_lex = tf.metrics.accuracy(labels_lex,predictions_lex,weights = mask)
        

    return inputs, labels, labels_lex, keep_prob, loss_fine, loss_lex, optimize_fine, optimize_lex,acc, acc_op, acc_lex, acc_op_lex, predictions, seq_lenght, mask, softmax_out, softmax_out_lex, predictions_lex, timestep, lex_codes, prediction_lex_codes
