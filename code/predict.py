from models import create_tensorflow_model
from models import create_tensorflow_model_Badhanau_attention
from models import create_tensorflow_model_hierarchical_multi_learning
from lxml import etree
import nltk
from nltk.corpus import wordnet as wn
import tensorflow as tf
import tensorflow.keras as K
import tensorflow_hub as hub
import pickle
import numpy as np
from utils import prediction_lex_predict
from utils import encoding_most_frequent_lex

def predict_babelnet(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <BABELSynset>" format (e.g. "d000.s000.t000 bn:01234567n").
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    HIDDEN_SIZE = 512
    EMBEDDING_SIZE = 1024
    ATTENTION_SIZE = 512
    MAX_TIMESTEP = 30

    o = open(resources_path+"/output_synset_dictionary.pkl","rb")
    O = pickle.load(o)
    i_o = open(resources_path+"/inverse_output_synset_dictionary.pkl","rb")
    inv_o = pickle.load(i_o)
    o_lex = open(resources_path+"/output_lex_dictionary.pkl","rb")
    O_lex = pickle.load(o_lex)
    inverse_o_lex = open(resources_path+"/inverse_lex_output_dictionary.pkl","rb")
    inverse_O_lex = pickle.load(inverse_o_lex)
    W2B = open(resources_path+"/w_to_b.dict","rb")
    w2b = pickle.load(W2B)
    B2LEX = open(resources_path+"/b_to_lex.dict","rb")
    b2lex = pickle.load(B2LEX)
    LEX2IDX = open(resources_path+"/lex2idx.pkl","rb")
    lex2idx = pickle.load(LEX2IDX)

    wn_pos_dict = {"VERB":wn.VERB,"ADJ":wn.ADJ,"ADV":wn.ADV,"NOUN":wn.NOUN}

    inputs, labels, labels_lex, keep_prob, loss_fine, loss_lex, train_op_fine,train_op_lex,acc, acc_op, acc_lex, acc_op_lex, predictions, seq_lenght, mask, softmax_out, softmax_out_lex, predictions_lex, ts, lex_codes, prediction_lex_codes = create_tensorflow_model_hierarchical_multi_learning(HIDDEN_SIZE,EMBEDDING_SIZE,len(O),len(O_lex),ATTENTION_SIZE,resources_path)

    saver = tf.train.Saver()
    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:
      # Restore variables from disk.
        print("restoring")
        saver.restore(sess, resources_path+"/fine_grain_model/model_NLP.ckpt")
        context = etree.iterparse(input_path)
        sentence = ""
        candidates = []
        gold_prediction = open(output_path,"w")
        # annotations is a dictionary containing needed information for the target words that need to be predicted
        annotations = dict()
        annotations_lex = dict()
        idx = 0
        s_d_t = open(resources_path+"/lemma_seen_training.pkl","rb")
        seen_during_training = pickle.load(s_d_t)
        count = 0
        lex_cod = []
        for action, elem in context:
            if elem.tag == "sentence":
                #Now I first need to predict the lexicographic domains and get the one_hot_encoding
                prob_lex = sess.run([softmax_out_lex],feed_dict = {inputs: [sentence.split(" ")], keep_prob: 1.0, seq_lenght: [len(sentence.split(" "))], mask:np.ones((1,len(sentence.strip("\n").split(" ")))),ts: len(sentence.split(" "))})
                prob_lex = np.asarray(prob_lex)
                # get the one hot encoding
                predict_lex = prediction_lex_predict(sentence,O_lex,inverse_O_lex,prob_lex,b2lex,w2b,lex2idx,annotations_lex)
                # get the predictions for the fine grained task
                prob = sess.run([softmax_out],feed_dict = {inputs: [sentence.split(" ")],lex_codes: [lex_cod], prediction_lex_codes: [predict_lex], keep_prob: 1.0, seq_lenght: [len(sentence.split(" "))], mask:np.ones((1,len(sentence.strip("\n").split(" ")))),ts: len(sentence.split(" "))})
                prob = np.asarray(prob)
                for a in annotations:
                    # Now for each target word that needs to be predicted
                    if annotations[a][2] != None:
                        # This case happend when the word at position a in the sentnce has been seen during the training phase and therefore I could extrapolate
                        # a list of candidates
                        # c is an array containing the indices of the output vocabulary of the task which are candidates for the current word
                        c = annotations[a][2]
                        # now from the output of the network I first get a-th element of prob which is the softmax output of the a-th word in the sentence. From it I 
                        # only retains the values specified by the indices in the c vector(aka the candidates value for the target word) and finally i get the indices of
                        # the maximum of these values.
                        prediction = c[np.argmax(prob[0][0][a][c])]
                        #from the integer i get the wordent id and thereafter the synset
                        inverse = inv_o[prediction]
                        pos = inverse[len(inverse)-1]
                        offset = int(inverse[3:len(inverse)-1])
                        synset = wn.synset_from_pos_and_offset(pos,offset)
                        synset_id = "wn:" + str(synset.offset()).zfill( 8) + synset.pos()
                        babelnet_id = w2b[synset_id]
                        gold_prediction.write(annotations[a][0] + " " + babelnet_id + "\n")
                    else:
                        # it means that theword was not seen in the training phase and so I get the most frequent sense through wordnet
                        gold_prediction.write(annotations[a][0] + " " + annotations[a][1] + "\n")
                sentence = ""
                annotations = dict()
                annotations_lex = dict()
                idx = 0
                lex_cod = []
                print(count)
                count += 1
            if elem.tag == "instance":
                # For each target word I need to create the binary encoding for the most frequent lexicographic domain, needed for the fine grained predictions
                enc = encoding_most_frequent_lex(elem.get("lemma"),elem.get("pos"),w2b,b2lex,lex2idx)
                lex_cod.append(enc)
                if elem.get("lemma") in seen_during_training:
                    annotations[idx] = [elem.get("id"),None]
                    #l will contain the list of candidate indices
                    l = []
                    #first i see all the synset of the lemma pos couple
                    #then from the output dictionary for each entries that is a synset id I get the synset and i consider it as a candidate if and only if is present in the synsets of the
                    #annotation
                    synsets = wn.synsets(elem.get("lemma"),pos = wn_pos_dict[elem.get("pos")])
                    for k in O:
                        if "wn:" in k:
                            pos = k[len(k)-1]
                            offset = int(k[3:len(k)-1])
                            synset = wn.synset_from_pos_and_offset(pos,offset)
                            if synset in synsets:
                                l.append(O[k])
                    if not l:
                        # if I could not find any match then I fall back to most frequen sense backing strategy
                        MFC_synset = wn.synsets(elem.get("lemma"),pos = wn_pos_dict[elem.get("pos")])[0]
                        MFC_sense = ""
                        babelnet_id = ""
                        synset_id = "wn:" + str(MFC_synset.offset()).zfill( 8) + MFC_synset.pos()
                        babelnet_id = w2b[synset_id]
                        # so in annotation I already store the most frequent babelnet id
                        annotations[idx] = [elem.get("id"),babelnet_id,None]
                    else:
                        annotations[idx].append(l)
                        annotations[idx].append(elem.get("lemma"))
                else:
                    # This is the case where the word was not in the seen lemma and therefore I need to fall back to most frequent sense strategy
                    MFC_synset = wn.synsets(elem.get("lemma"),pos = wn_pos_dict[elem.get("pos")])[0]
                    MFC_sense = ""
                    babelnet_id = ""
                    synset_id = "wn:" + str(MFC_synset.offset()).zfill( 8) + MFC_synset.pos()
                    babelnet_id = w2b[synset_id]
                    # so in annotation I already store the most frequent babelnet id
                    annotations[idx] = [elem.get("id"),babelnet_id,None]
                #this part is needed to get the prediction on the lexicographic task
                if elem.get("lemma") in seen_during_training:
                    annotations_lex[idx] = [elem.get("id"),None]
                    #get the synset associated the lemma pos
                    synsets = wn.synsets(elem.get("lemma"),pos = wn_pos_dict[elem.get("pos")])
                    #for every synset I get the wordnet lex domain and thhese are the candidates for when i will have to find the maximum
                    lexs = []
                    for synset in synsets:
                      synset_id = "wn:" + str(synset.offset()).zfill( 8) + synset.pos()
                      babelnet_id = w2b[synset_id]
                      lex = b2lex[babelnet_id]
                      if lex in O_lex and O_lex[lex] not in lexs:
                        lexs.append(O_lex[lex])
                    annotations_lex[idx].append(lexs)
                else:
                    # else i need to fall back to the most frequent back off strategy
                    MFC_lex = ""
                    MFC_sense = wn.synsets(elem.get("lemma"),pos = wn_pos_dict[elem.get("pos")])[0]
                    synset_id = "wn:" + str(MFC_sense.offset()).zfill( 8) + MFC_sense.pos()
                    babelnet_id = w2b[synset_id]
                    lex = b2lex[babelnet_id]
                    MFC_lex = lex
                    # so in annotations_lex i store the most frequent lexicographic domain
                    annotations_lex[idx] = [elem.get("id"),MFC_lex,None] 
                if sentence == "":
                    sentence = sentence + elem.get("lemma")
                else:
                    sentence = sentence + " " + elem.get("lemma")
                idx+=1
            if elem.tag == "wf":
                # For each target word I need to create the binary encoding for the most frequent lexicographic domain, needed for the fine grained predictions
                # it is the same as the case of the instances lemma
                enc = encoding_most_frequent_lex(elem.get("lemma"),elem.get("pos"),w2b,b2lex,lex2idx)
                lex_cod.append(enc)
                if sentence == "":
                    sentence = sentence + elem.get("lemma")
                else:
                    sentence = sentence + " " + elem.get("lemma")
                idx+=1
        gold_prediction.close()

    pass


def predict_wordnet_domains(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <wordnetDomain>" format (e.g. "d000.s000.t000 sport").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    HIDDEN_SIZE = 512
    EMBEDDING_SIZE = 1024
    ATTENTION_SIZE = 512
    MAX_TIMESTEP = 30

    o = open(resources_path+"/coarsed_grain_domain_model/output_dictionary.pkl","rb")
    O = pickle.load(o)
    i_o = open(resources_path+"/coarsed_grain_domain_model/inverse_output_dictionary.pkl","rb")
    inv_o = pickle.load(i_o)
    W2B = open(resources_path+"/w_to_b.dict","rb")
    w2b = pickle.load(W2B)
    b_to_d = open(resources_path+"/coarsed_grain_domain_model/b_to_dom.dict","rb")
    b2d = pickle.load(b_to_d)

    wn_pos_dict = {"VERB":wn.VERB,"ADJ":wn.ADJ,"ADV":wn.ADV,"NOUN":wn.NOUN}

    inputs, labels, keep_prob, loss, train_op, acc, acc_op, predictions, seq_lenght, mask, softmax_out, ts = create_tensorflow_model_Badhanau_attention(HIDDEN_SIZE,EMBEDDING_SIZE,len(O),ATTENTION_SIZE,resources_path)
    print(labels)
    saver = tf.train.Saver()
    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:
      # Restore variables from disk.
        print("restoring")
        saver.restore(sess, resources_path+"/coarsed_grain_domain_model/model_NLP.ckpt")
        context = etree.iterparse(input_path)
        sentence = ""
        candidates = []
        gold_prediction = open(output_path,"w")
        # annotations is a dictionary containing needed information for the target words that need to be predicted
        annotations = dict()
        idx = 0
        s_d_t = open(resources_path+"/lemma_seen_training.pkl","rb")
        seen_during_training = pickle.load(s_d_t)
        count = 0
        for action, elem in context:
            # logic in the following lines are very similar to the preceding function
            if elem.tag == "sentence":
              prob = sess.run([softmax_out],feed_dict = {inputs: [sentence.split(" ")], keep_prob: 1.0, seq_lenght: [len(sentence.split(" "))], mask:np.ones((1,len(sentence.strip("\n").split(" ")))),ts: len(sentence.split(" "))})
              prob = np.asarray(prob)
              for a in annotations:
                if annotations[a][2] != None:
                  c = annotations[a][2]
                  prediction = c[np.argmax(prob[0][0][a][c])]
                  gold_prediction.write(annotations[a][0] + " " + inv_o[prediction] + "\n")
                else:
                  gold_prediction.write(annotations[a][0] + " " + annotations[a][1] + "\n")
              sentence = ""
              annotations = dict()
              idx = 0
              print(count)
              count+=1
            if elem.tag == "instance":
              if elem.get("lemma") in seen_during_training:
                annotations[idx] = [elem.get("id"),None]
                #get the synset associated the lemma pos
                synsets = wn.synsets(elem.get("lemma"),pos = wn_pos_dict[elem.get("pos")])
                #for every synset I get the wordnet domain
                domains = []
                for synset in synsets:
                  synset_id = "wn:" + str(synset.offset()).zfill( 8) + synset.pos()
                  babelnet_id = w2b[synset_id]
                  if babelnet_id in b2d:
                    domain = b2d[babelnet_id]
                    if domain in O and O[domain] not in domains:
                      domains.append(O[domain])
                  else:
                    if O["factotum"] not in domains:
                      domains.append(O["factotum"])
                annotations[idx].append(domains)
              else:
                MFC_domain = ""
                MFC_sense = wn.synsets(elem.get("lemma"),pos = wn_pos_dict[elem.get("pos")])[0]
                synset_id = "wn:" + str(MFC_sense.offset()).zfill( 8) + MFC_sense.pos()
                babelnet_id = w2b[synset_id]
                if babelnet_id in b2d:
                  domain = b2d[babelnet_id]
                  MFC_domain = domain
                else:
                  MFC_domain = "factotum"
                annotations[idx] = [elem.get("id"),MFC_domain,None]
              if sentence == "":
                sentence = sentence + elem.get("lemma")
              else:
                sentence = sentence + " " + elem.get("lemma")
              idx+=1
            if elem.tag == "wf":
              if sentence == "":
                sentence = sentence + elem.get("lemma")
              else:
                sentence = sentence + " " + elem.get("lemma")
              idx+=1
        gold_prediction.close()
    pass


def predict_lexicographer(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <lexicographerId>" format (e.g. "d000.s000.t000 noun.animal").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """


    HIDDEN_SIZE = 512
    EMBEDDING_SIZE = 1024
    ATTENTION_SIZE = 512
    MAX_TIMESTEP = 30

    o = open(resources_path+"/output_synset_dictionary.pkl","rb")
    O = pickle.load(o)
    i_o = open(resources_path+"/inverse_output_synset_dictionary.pkl","rb")
    inv_o = pickle.load(i_o)
    o_lex = open(resources_path+"/output_lex_dictionary.pkl","rb")
    O_lex = pickle.load(o_lex)
    inverse_o_lex = open(resources_path+"/inverse_lex_output_dictionary.pkl","rb")
    inverse_O_lex = pickle.load(inverse_o_lex)
    W2B = open(resources_path+"/w_to_b.dict","rb")
    w2b = pickle.load(W2B)
    B2LEX = open(resources_path+"/b_to_lex.dict","rb")
    b2lex = pickle.load(B2LEX)

    wn_pos_dict = {"VERB":wn.VERB,"ADJ":wn.ADJ,"ADV":wn.ADV,"NOUN":wn.NOUN}

    inputs, labels, labels_lex, keep_prob, loss, loss_fine, loss_lex, train_op, acc, acc_op, acc_lex, acc_op_lex, predictions, predictions_lex, seq_lenght, mask, softmax_out, softmax_out_lex, ts = create_tensorflow_model(HIDDEN_SIZE,EMBEDDING_SIZE,len(O),len(O_lex),ATTENTION_SIZE,resources_path)

    saver = tf.train.Saver()
    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:
      # Restore variables from disk.
        print("restoring")
        saver.restore(sess, resources_path+"/coarsed_grained_lex_model/model_NLP_lex.ckpt")
        context = etree.iterparse(input_path)
        sentence = ""
        candidates = []
        gold_prediction_lex = open(output_path,"w")
        annotations_lex = dict()
        idx = 0
        s_d_t = open(resources_path+"/lemma_seen_training.pkl","rb")
        seen_during_training = pickle.load(s_d_t)
        count = 0
        for action, elem in context:
            if elem.tag == "sentence":
              # in the following lines is implemented a logic very similar to previous prediction function
              prob_lex = sess.run([softmax_out_lex],feed_dict = {inputs: [sentence.split(" ")], keep_prob: 1.0, seq_lenght: [len(sentence.split(" "))], mask:np.ones((1,len(sentence.strip("\n").split(" ")))),ts: len(sentence.split(" "))})
              prob_lex = np.asarray(prob_lex)
              for a in annotations_lex:
                if annotations_lex[a][2] != None:
                  c = annotations_lex[a][2]
                  prediction = c[np.argmax(prob_lex[0][0][a][c])]
                  gold_prediction_lex.write(annotations_lex[a][0] + " " + inverse_O_lex[prediction] + "\n")
                else:
                  gold_prediction_lex.write(annotations_lex[a][0] + " " + annotations_lex[a][1] + "\n")
              sentence = ""
              annotations_lex = dict()
              idx = 0
              if count%100 == 0:
                print(count)
              count += 1
            if elem.tag == "instance":
              #this part is needed to get the prediction on the lexicographic task
              if elem.get("lemma") in seen_during_training:
                annotations_lex[idx] = [elem.get("id"),None]
                #get the synset associated the lemma pos
                synsets = wn.synsets(elem.get("lemma"),pos = wn_pos_dict[elem.get("pos")])
                #for every synset I get the wordnet lex domain
                lexs = []
                for synset in synsets:
                  synset_id = "wn:" + str(synset.offset()).zfill( 8) + synset.pos()
                  babelnet_id = w2b[synset_id]
                  lex = b2lex[babelnet_id]
                  if lex in O_lex and O_lex[lex] not in lexs:
                    lexs.append(O_lex[lex])
                annotations_lex[idx].append(lexs)
              else:
                MFC_lex = ""
                MFC_sense = wn.synsets(elem.get("lemma"),pos = wn_pos_dict[elem.get("pos")])[0]
                synset_id = "wn:" + str(MFC_sense.offset()).zfill( 8) + MFC_sense.pos()
                babelnet_id = w2b[synset_id]
                lex = b2lex[babelnet_id]
                MFC_lex = lex
                annotations_lex[idx] = [elem.get("id"),MFC_lex,None]  
              if sentence == "":
                sentence = sentence + elem.get("lemma")
              else:
                sentence = sentence + " " + elem.get("lemma")
              idx+=1
            if elem.tag == "wf":
              if sentence == "":
                sentence = sentence + elem.get("lemma")
              else:
                sentence = sentence + " " + elem.get("lemma")
              idx+=1
        gold_prediction_lex.close()
    pass
