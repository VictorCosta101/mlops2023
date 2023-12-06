
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.macros import ds_add
from airflow.utils.task_group import TaskGroup

import tensorflow as tf
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, TextVectorization, Dropout, GlobalAveragePooling1D, Bidirectional
from tensorflow.keras.regularizers import L1, L2, L1L2
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.data import Dataset

import transformers
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification

import pendulum
import requests
import logging
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

def download_data(url, file_name):
    logging.info(f"Downloading {url} to {file_name}")

    r = requests.get(url)

    with open(file_name, "wb") as f:
        f.write(r.content)

def clean_data():
    
    train = pd.read_csv("train.csv")
    train = train.drop(['id', 'keyword', 'location'], axis = 1)
    #test = pd.read_csv("test.csv")
    ##data = pd.concat([train, test], axis=0)
    ##data.to_csv("data_concatenated.csv", index=False)
    train.to_csv("train_cleaned.csv", index=False)
def remove_punctuation_and_number(text: str):

    return re.sub(r'[^a-zA-Z]', ' ', text)

def remove_stopwords(tokens: list, stop_words):

    return [token for token in tokens if token not in stop_words]

def lemmatize_word(tokens: list, lemmatizer: WordNetLemmatizer):

    return [lemmatizer.lemmatize(token, pos = 'v') for token in tokens]

def text_preprocessing():

    train = pd.read_csv("train_cleaned.csv")
    
    ##1. Convert to lower case
    train['text'] = train['text'].str.lower()
    
    ##2. Remove punctuations and numbers
    train['text'] = train['text'].apply(remove_punctuation_and_number)

    ##3. Tokenize
    train['text'] = train['text'].apply(word_tokenize)

    ##4.Remove stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    train['text'] = train['text'].apply(remove_stopwords, stop_words = stop_words)

    ##5. Lemmatize
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    lemmatizer = WordNetLemmatizer()
    train['text'] = train['text'].apply(lemmatize_word, lemmatizer = lemmatizer)

    ##6. Join the tokens into processed text
    train['text_final'] = train['text'].str.join(' ')

    train.to_csv("train_text_prepocessing.csv", index=False)

def create_nn_model(vectorizer_layer, embedding_layer):

    inputs = Input(shape = (1,), dtype = tf.string)

    x = vectorizer_layer(inputs)
    x = embedding_layer(x)
    x = Dense(64, activation = 'relu')(x)
    x = Dense(16, activation = 'relu')(x)
    outputs = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer = Adam(learning_rate = 0.01), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model
def evaluate_model(model, X_train, y_train, X_test, y_test, epochs = 10):

    model.fit(X_train, y_train, epochs = epochs)
    print()
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print()
    print(f'Test loss: {test_loss}\nTest accuracy: {test_acc}')

def prepare_data():
    # Lendo os dados do CSV
    train = pd.read_csv("/home/victor/Documentos/mlops/twitter_classifying/mlops2023/Classifying_Disaster-Related_Tweets_as_Real_or_Fake/train_text_prepocessing.csv")

    X = train['text_final']
    y = train['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
    max_tokens = 5000
    sequence_length = 128
    embedding_length = 128

    vectorizer_layer = TextVectorization(max_tokens = max_tokens, standardize ='lower_and_strip_punctuation', ngrams = (1,2), output_mode = 'int', output_sequence_length = sequence_length)
    vectorizer_layer.adapt(X_train)

    embedding_layer = Embedding(input_dim = max_tokens, output_dim = embedding_length, input_length = sequence_length)

    return X_train, y_train, X_test, y_test, vectorizer_layer, embedding_layer, sequence_length, embedding_length

def Build_shallow_neural_network():
    X_train, y_train, X_test, y_test, vectorizer_layer, embedding_layer, sequence_length, embedding_length = prepare_data()
    model_shallow_nn = create_nn_model(vectorizer_layer, embedding_layer)
    evaluate_model(model_shallow_nn, X_train, y_train, X_test, y_test)

def create_deep_nn_model(vectorizer_layer, embedding_layer):

    inputs = Input(shape = (1,), dtype = tf.string)

    x = vectorizer_layer(inputs)
    x = embedding_layer(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation = 'relu', kernel_regularizer = L2(l2 = 0.001))(x)
    x = Dropout(rate = 0.2)(x)
    x = Dense(32, activation = 'relu', kernel_regularizer = L2(l2 = 0.001))(x)
    x = Dropout(rate = 0.1)(x)
    x = Dense(16, activation = 'relu', kernel_regularizer = L2(l2 = 0.0005))(x)
    x = Dense(4, activation = 'relu', kernel_regularizer = L2(l2 = 0.0005))(x)
    outputs = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer = Adam(learning_rate = 0.01), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model

def Build_Multilayer_Deep_Text_Classification_Model():
    X_train, y_train, X_test, y_test, vectorizer_layer, embedding_layer, sequence_length, embedding_length = prepare_data()

    model_deep_nn = create_deep_nn_model(vectorizer_layer, embedding_layer)
    evaluate_model(model_deep_nn, X_train, y_train, X_test, y_test)

def create_bi_lstm_model(vectorizer_layer, embedding_layer):

    inputs = Input(shape = (1,), dtype = tf.string)

    x = vectorizer_layer(inputs)
    x = embedding_layer(x)
    x = Bidirectional(LSTM(64, return_sequences = True), merge_mode = 'ave')(x)
    x = Bidirectional(LSTM(64, return_sequences = True), merge_mode = 'ave')(x)
    x = Bidirectional(LSTM(32), merge_mode = 'ave')(x)
    x = Dense(32, activation = 'relu', kernel_regularizer = L2(l2 = 0.001))(x)
    x = Dropout(rate = 0.2)(x)
    x = Dense(16, activation = 'relu', kernel_regularizer = L2(l2 = 0.0005))(x)
    x = Dense(8, activation = 'relu', kernel_regularizer = L2(l2 = 0.0005))(x)
    x = Dense(4, activation = 'relu', kernel_regularizer = L2(l2 = 0.0005))(x)
    outputs = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer = Adam(learning_rate = 0.01), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model

def Building_Multilayer_bidirectional_lstm_model():
    X_train, y_train, X_test, y_test, vectorizer_layer, embedding_layer, sequence_length, embedding_length = prepare_data()
    model_bi_lstm = create_bi_lstm_model(vectorizer_layer, embedding_layer)

def Building_tranformer_model():
    X_train, y_train, X_test, y_test, vectorizer_layer, embedding_layer,sequence_length, embedding_length = prepare_data()
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer

    train_encodings = tokenizer(list(X_train), max_length = sequence_length, truncation = True, padding = True)
    test_encodings = tokenizer(list(X_test), max_length = sequence_length, truncation = True, padding = True)

    train_dataset = Dataset.from_tensor_slices(
    (dict(train_encodings), tf.constant(y_train, dtype = tf.int32))
                                           )

    test_dataset = Dataset.from_tensor_slices(
    (dict(test_encodings), tf.constant(y_test, dtype = tf.int32))
                                           )
    
    train_dataset = train_dataset.shuffle(len(X_train)).batch(32)
    test_dataset = test_dataset.batch(32)

    model_transformer = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 2)
    optimizer = Adam(learning_rate = 2e-5)
    loss = SparseCategoricalCrossentropy(from_logits = True)
    metrics = SparseCategoricalAccuracy()   

    model_transformer.compile(optimizer = optimizer, loss = loss, metrics = [metrics])

    model_transformer.fit(train_dataset, epochs = 3, validation_data = test_dataset)
    

with DAG(
    dag_id="twiter_real_or_fake",
    schedule_interval = '@daily',
    start_date = pendulum.datetime(2023,12,1),
    catchup = False
) as dag: 
     with TaskGroup("Fetching_Data") as fetching_data:              
        download_train_data = PythonOperator(
                task_id="download_train_data",
                python_callable=download_data,
                op_kwargs={
                    "url": "https://dsserver-prod-resources-1.s3.amazonaws.com/nlp/train.csv",
                    "file_name": "train.csv"
                },
            )
        download_test_data = PythonOperator(
                task_id="download_test_data",
                python_callable=download_data,
                op_kwargs={
                    "url": "https://dsserver-prod-resources-1.s3.amazonaws.com/nlp/test.csv",
                    "file_name": "test.csv"
                },
            )
     clean_data = PythonOperator(
            task_id="clean_data",
            python_callable=clean_data,
           
        )
     Text_Preprocessing = PythonOperator(
            task_id="Text_Preprocessing",
            python_callable=text_preprocessing,
           
        )
     Build_shallow_neural_network = PythonOperator(
            task_id="Build_shallow_neural_network",
            python_callable=Build_shallow_neural_network,
           
        )
     Build_Multilayer_Deep_Text_Classification_Model = PythonOperator(
            task_id="Build_Multilayer_Deep_Text_Classification_Model",
            python_callable=Build_Multilayer_Deep_Text_Classification_Model,
           
        )
     Building_Multilayer_bidirectional_lstm_model = PythonOperator(
            task_id="Building_Multilayer_bidirectional_lstm_model",
            python_callable=Building_Multilayer_bidirectional_lstm_model,
           
        )
     Building_tranformer_model = PythonOperator(
            task_id="Building_tranformer_model",
            python_callable=Building_tranformer_model,
           
        )
        
     fetching_data >> clean_data >> Text_Preprocessing >> Build_shallow_neural_network >> Build_Multilayer_Deep_Text_Classification_Model >> Building_Multilayer_bidirectional_lstm_model >> Building_tranformer_model
    