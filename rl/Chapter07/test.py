import data_parser
from gensim.models import KeyedVectors
from seq_model import Chatbot
import tensorflow as tf
import numpy as np
import helper as h

reinforcement_model_path = "model/reinforcement/seq2seq"
forward_model_path = "model/forward/seq2seq"
reverse_model_path = "model/reverse/seq2seq"
path_to_questions = 'results/sample_input.txt'
responses_path = 'results/sample_output_RL.txt'

word_count_threshold = 20
dim_wordvec = 300
dim_hidden = 1000

input_sequence_length = 25
target_sequence_length = 22

batch_size = 2


def test(model_path=forward_model_path):
    testing_data = open(path_to_questions, 'r').read().split('\n')
    word_vector = KeyedVectors.load_word2vec_format('model/word_vector.bin', binary=True)

    _, index_to_word, _ = data_parser.preProBuildWordVocab(word_count_threshold=word_count_threshold)

    model = Chatbot(dim_wordvec, len(index_to_word), dim_hidden, batch_size,
                            input_sequence_length, target_sequence_length, Training=False)

    place_holders, predictions, logits = model.build_model()

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()

    saver.restore(sess, model_path)

    with open(responses_path, 'w') as out:

        for idx, question in enumerate(testing_data):
            print('question =>', question)

            question = [h.refine(w) for w in question.lower().split()]
            question = [word_vector[w] if w in word_vector else np.zeros(dim_wordvec) for w in question]
            question.insert(0, np.random.normal(size=(dim_wordvec,)))  # insert random normal at the first step

            if len(question) > input_sequence_length:
                question = question[:input_sequence_length]
            else:
                for _ in range(input_sequence_length - len(question)):
                    question.append(np.zeros(dim_wordvec))

            question = np.array([question])

            feed_dict = {place_holders["word_vectors"]: np.concatenate([question] * 2, 0),
                         }

            word_indices, prob_logit = sess.run([predictions, logits], feed_dict=feed_dict)

            # print(word_indices[0].shape)
            generated_sentence = h.index2sentence(word_indices[0], prob_logit[0], index_to_word)

            print('generated_sentence =>', generated_sentence)
            out.write(generated_sentence + '\n')


test(reinforcement_model_path)
