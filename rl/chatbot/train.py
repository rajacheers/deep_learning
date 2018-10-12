from data_reader import Data_Reader
import data_parser
from gensim.models import KeyedVectors
import helper as h
from seq_model import Chatbot
import tensorflow as tf
import numpy as np

generic_responses = [
    "I don't know what you're talking about.",
    "I don't know.",
    "You don't know.",
    "You know what I mean.",
    "I know what you mean.",
    "You know what I'm saying.",
    "You don't know anything."
]

checkpoint = True
forward_model_path = 'model/forward'
reversed_model_path = 'model/reversed'
rl_model_path = "model/rl"
model_name = 'seq2seq'
# batch_size = 2
word_count_threshold = 20
reversed_word_count_threshold = 6
# max_turns = 10
dim_wordvec = 300
dim_hidden = 1000
input_sequence_length = 22
output_sequence_length = 22
learning_rate = 0.0001
epochs = 1
batch_size = 200
forward_ = "forward"
reverse_ = "reverse"
forward_epochs = 50
reverse_epochs = 50
display_interval = 100


def train(type_, epochs=epochs, checkpoint=False):
    tf.reset_default_graph()
    if type_ == "forward":
        path = "model/forward/seq2seq"
        dr = Data_Reader(reverse=False)
    else:
        dr = Data_Reader(reverse=True)
        path = "model/reverse/seq2seq"

    word_to_index, index_to_word, _ = data_parser.preProBuildWordVocab(word_count_threshold=word_count_threshold)
    word_vector = KeyedVectors.load_word2vec_format('model/word_vector.bin', binary=True)
    model = Chatbot(dim_wordvec, len(word_to_index), dim_hidden, batch_size,
                    input_sequence_length, output_sequence_length, learning_rate)
    optimizer, place_holders, predictions, logits, losses = model.build_model()
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    if checkpoint:
        saver.restore(sess, path)
        print("checkpoint restored at path: {}".format(path))
    else:
        tf.global_variables_initializer().run()

    for epoch in range(epochs):
        n_batch = dr.get_batch_num(batch_size=batch_size)
        for batch in range(n_batch):

            batch_input, batch_target = dr.generate_training_batch(batch_size)
            inputs_ = h.make_batch_input(batch_input, input_sequence_length, dim_wordvec, word_vector)

            targets, masks = h.make_batch_target(batch_target, word_to_index, output_sequence_length)
            feed_dict = {
                place_holders['word_vectors']: inputs_,
                place_holders['caption']: targets,
                place_holders['caption_mask']: masks
            }

            _, loss_val, preds = sess.run([optimizer, losses["entropy"], predictions],
                                          feed_dict=feed_dict)

            if batch % display_interval == 0:
                print(preds.shape)
                print("Epoch: {}, batch: {}, loss: {}".format(epoch, batch, loss_val))
                print("===========================================================")

        saver.save(sess, path)

        print("Model saved at {}".format(path))
    print("Training done")

    sess.close()


def pg_train(epochs=epochs, checkpoint=False):
    tf.reset_default_graph()
    path = "model/reinforcement/seq2seq"
    word_to_index, index_to_word, _ = data_parser.preProBuildWordVocab(word_count_threshold=word_count_threshold)
    word_vector = KeyedVectors.load_word2vec_format('model/word_vector.bin', binary=True)
    generic_caption, generic_mask = h.generic_batch(generic_responses, batch_size, word_to_index,
                                                    output_sequence_length)

    dr = Data_Reader()
    forward_graph = tf.Graph()
    reverse_graph = tf.Graph()
    default_graph = tf.get_default_graph()

    with forward_graph.as_default():
        pg_model = Chatbot(dim_wordvec, len(word_to_index), dim_hidden, batch_size,
                           input_sequence_length, output_sequence_length, learning_rate, policy_gradients=True)
        optimizer, place_holders, predictions, logits, losses = pg_model.build_model()

        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        if checkpoint:
            saver.restore(sess, path)
            print("checkpoint restored at path: {}".format(path))
        else:
            tf.global_variables_initializer().run()
            saver.restore(sess, 'model/forward/seq2seq')
    # tf.global_variables_initializer().run()
    with reverse_graph.as_default():
        model = Chatbot(dim_wordvec, len(word_to_index), dim_hidden, batch_size,
                        input_sequence_length, output_sequence_length, learning_rate)
        _, rev_place_holders, _, _, reverse_loss = model.build_model()
        sess2 = tf.InteractiveSession()
        saver2 = tf.train.Saver()

        saver2.restore(sess2, "model/reverse/seq2seq")
        print("reverse model restored")

    dr = Data_Reader(load_list=True)

    for epoch in range(epochs):
        n_batch = dr.get_batch_num(batch_size=batch_size)
        for batch in range(n_batch):

            batch_input, batch_caption, prev_utterance = dr.generate_training_batch_with_former(batch_size)
            targets, masks = h.make_batch_target(batch_caption, word_to_index, output_sequence_length)
            inputs_ = h.make_batch_input(batch_input, input_sequence_length, dim_wordvec, word_vector)

            word_indices, probabilities = sess.run([predictions, logits],
                                                   feed_dict={place_holders['word_vectors']: inputs_

                                                       , place_holders["caption"]: targets})

            sentence = [h.index2sentence(generated_word, probability, index_to_word) for
                        generated_word, probability in zip(word_indices, probabilities)]

            word_list = [word.split() for word in sentence]

            generic_test_input = h.make_batch_input(word_list, input_sequence_length, dim_wordvec, word_vector)

            forward_coherence_target, forward_coherence_masks = h.make_batch_target(sentence,
                                                                                    word_to_index,
                                                                                    output_sequence_length)

            generic_loss = 0.0

            for response in generic_test_input:
                sentence_input = np.array([response] * batch_size)
                feed_dict = {place_holders['word_vectors']: sentence_input,
                             place_holders['caption']: generic_caption,
                             place_holders['caption_mask']: generic_mask,
                             }
                generic_loss_i = sess.run(losses["entropy"], feed_dict=feed_dict)
                generic_loss -= generic_loss_i / batch_size

            # print("generic loss work: {}".format(generic_loss))

            feed_dict = {place_holders['word_vectors']: inputs_,
                         place_holders['caption']: forward_coherence_target,
                         place_holders['caption_mask']: forward_coherence_masks,
                         }

            forward_entropy = sess.run(losses["entropy"], feed_dict=feed_dict)

            previous_utterance, previous_mask = h.make_batch_target(prev_utterance,
                                                                    word_to_index, output_sequence_length)

            feed_dict = {rev_place_holders['word_vectors']: generic_test_input,
                         rev_place_holders['caption']: previous_utterance,
                         rev_place_holders['caption_mask']: previous_mask,
                         }
            reverse_entropy = sess2.run(reverse_loss["entropy"], feed_dict=feed_dict)

            rewards = 1 / (1 + np.exp(-reverse_entropy - forward_entropy - generic_loss))

            feed_dict = {place_holders['word_vectors']: inputs_,
                         place_holders['caption']: targets,
                         place_holders['caption_mask']: masks,
                         place_holders['rewards']: rewards
                         }

            _, loss_pg, loss_ent = sess.run([optimizer, losses["pg"], losses["entropy"]], feed_dict=feed_dict)

            if batch % display_interval == 0:
                print("Epoch: {}, batch: {}, Entropy loss: {}, Policy gradient loss: {}".format(epoch, batch, loss_ent,
                                                                                                loss_pg))

                print("rewards: {}".format(rewards))
                print("===========================================================")
        saver.save(sess, path)
        print("Model saved at {}".format(path))
    print("Training done")


train(forward_, forward_epochs, False)
train(reverse_, reverse_epochs, False)
pg_train(100, False)