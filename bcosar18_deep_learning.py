import numpy as np
import os


file_dir='C:\\Users\\Dell\\Desktop\\aclImdb'
train_dir= os.path.join(file_dir, 'train')
test_dir= os.path.join(file_dir, 'test')

import time
def naive_bayes(file_dir):
    train_dir= os.path.join(file_dir, 'train')
    test_dir= os.path.join(file_dir, 'test')
    labels = []
    texts = []


    punctuation = '''!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~'''

    for label in ['neg', 'pos']:
        directory = os.path.join(train_dir, label)
        for file_name in os.listdir(directory):
            if file_name.endswith('.txt'):  # Check if it is a .txt file
                with open(os.path.join(directory, file_name), encoding='utf-8') as f:

                    text = f.read().lower()
                    text = ''.join([char for char in text if char not in punctuation])
                    texts.append(text)

                if label == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
                    
    test_labels = []
    test_texts = []

    for label in ['neg', 'pos']:
        directory = os.path.join(test_dir, label)
        for file_name in os.listdir(directory):
            if file_name.endswith('.txt'):  # Check if it is a .txt file
                with open(os.path.join(directory, file_name), encoding='utf-8') as f:

                    text = f.read().lower()
                    text = ''.join([char for char in text if char not in punctuation])
                    test_texts.append(text)


                if label == 'neg':
                    test_labels.append(0)
                else:
                    test_labels.append(1)

    pos_dict=dict()
    neg_dict=dict()

    for idx, sentence in enumerate(texts):
        words=sentence.split()
        if labels[idx]==0:
            dictionary=neg_dict
        else:
            dictionary=pos_dict

        for word in words:
            if word not in dictionary:
                dictionary[word]=1
            else:
                dictionary[word]+=1
                
    total_docs = len(labels)
    num_pos = np.sum(labels) 
    num_neg = total_docs - num_pos 

    #prior probs
    p_pos = num_pos / total_docs
    p_neg = num_neg / total_docs

    #unique words, ie vocabulary of train set
    vocabulary = set(list(pos_dict.keys()) + list(neg_dict.keys()))
    vocab_size = len(vocabulary)

    def calculate_likelihood(word, dictionary, total_word_count):
        return (dictionary.get(word, 0) + 1) / (total_word_count + vocab_size)

    total_words_pos = sum(pos_dict.values())
    total_words_neg = sum(neg_dict.values())

    #prediction function
    def predict(sentence):
        words = sentence.split()

        log_prob_pos = np.log(p_pos)
        log_prob_neg = np.log(p_neg)

        for word in words:
            log_prob_pos += np.log(calculate_likelihood(word, pos_dict, total_words_pos))
            log_prob_neg += np.log(calculate_likelihood(word, neg_dict, total_words_neg))

        return 1 if log_prob_pos > log_prob_neg else 0

 

    def calculate_accuracy_with_time(test_texts, test_labels):
        correct_predictions = 0
        start_time = time.time()  

        for idx, test_sentence in enumerate(test_texts):
            predicted_class = predict(test_sentence)  
            actual_class = test_labels[idx]  

            if predicted_class == actual_class:
                correct_predictions += 1

        end_time = time.time()  
        elapsed_time = end_time - start_time  

        accuracy = correct_predictions / len(test_labels)
        return accuracy, elapsed_time


    accuracy, prediction_time = calculate_accuracy_with_time(test_texts, test_labels)
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")
    print(f"Time taken to predict on the test set: {prediction_time:.4f} seconds")



start_time = time.time()  
naive_bayes(file_dir)           
end_time = time.time() 
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")
    