import tensorflow as tf
import numpy as np
import helper as h


class Chatbot():
    def __init__(self, embed_dim, vocab_size, lstm_size, batch_size, input_sequence_length, target_sequence_length, learning_rate =0.0001, keep_prob = 0.5, num_layers = 1, policy_gradients = False, Training = True):
        self.embed_dim = embed_dim
        self.lstm_size = lstm_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.input_sequence_length = tf.fill([self.batch_size],input_sequence_length+1)
        self.target_sequence_length = tf.fill([self.batch_size],target_sequence_length+1)
        self.output_sequence_length = target_sequence_length +1
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.num_layers = num_layers
        self.policy_gradients = policy_gradients
        self.Training = Training
        
    def build_model(self):
        if self.policy_gradients:
            word_vectors, caption, caption_mask, rewards = h.model_inputs(self.embed_dim, True)
            place_holders = {'word_vectors': word_vectors,
                'caption': caption,
                'caption_mask': caption_mask, "rewards": rewards
                             }
        else:
            word_vectors, caption, caption_mask = h.model_inputs(self.embed_dim)
            
            place_holders = {'word_vectors': word_vectors,
                'caption': caption,
                'caption_mask': caption_mask}
        enc_output, enc_state = h.encoding_layer(word_vectors, self.lstm_size, self.num_layers,
                                         self.keep_prob, self.vocab_size)
        #dec_inp = h.bos_inclusion(caption, self.batch_size)
        dec_inp = caption
        
        if not self.Training:
            print("Test mode")
            inference_out = h.decoding_layer(dec_inp, enc_state,self.target_sequence_length, 
                                                    self.output_sequence_length,
                                                    self.lstm_size, self.num_layers,
                                                    self.vocab_size, self.batch_size,
                                                  self.keep_prob, self.embed_dim, False)
            logits = tf.identity(inference_out.rnn_output, name = "train_logits")
            predictions = tf.identity(inference_out.sample_id, name = "predictions")
            return place_holders, predictions, logits
        
        train_out, inference_out = h.decoding_layer(dec_inp, enc_state,self.target_sequence_length, 
                                                    self.output_sequence_length,
                                                    self.lstm_size, self.num_layers,
                                                    self.vocab_size, self.batch_size,
                                                  self.keep_prob, self.embed_dim)
        
        
        
        
        training_logits = tf.identity(train_out.rnn_output, name = "train_logits")
        prediction_logits = tf.identity(inference_out.sample_id, name = "predictions")
        cross_entropy = tf.contrib.seq2seq.sequence_loss(training_logits, caption, caption_mask)
        losses = {"entropy": cross_entropy}
        
        
        if self.policy_gradients:
            pg_loss = tf.contrib.seq2seq.sequence_loss(training_logits, caption, caption_mask*rewards)
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(pg_loss)
            losses.update({"pg":pg_loss}) 
        else:
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)
                
        return optimizer, place_holders,prediction_logits,training_logits, losses

