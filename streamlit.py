# Import required modules
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint

import streamlit as st
from io import StringIO
import shutil
import os

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import collections
from matplotlib import pyplot as plt


import numpy as np
import random
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
		"https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]



# Assign credentials ann path of style sheet
creds = ServiceAccountCredentials.from_json_keyfile_name("/content/thesis-367714-64137e9747ad.json", scope)
client = gspread.authorize(creds)
spreadsheet = client.open("thesis_aliya")
sheet = spreadsheet.sheet1

def load(paths, verbose=-1):
    '''expects images for each class in seperate dir'''
    data = list()
    labels = list()
    # loop over the input images
    for (i, imgpath) in enumerate(paths):
        # load the image and extract the class labels
        im_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        image = np.array(im_gray).flatten()
        label = imgpath.split(os.path.sep)[-2] # Metaplastic, Parabasal, Koilocytotic, Superficial-Intermediate, Dyskeratotic
        # scale the image to [0, 1] and add to list
        data.append(image/255)
        labels.append(label)
        # show an update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print(label);
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))
    # return a tuple of the data and labels
    return data, labels

class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

#process and batch the training data for each client
clients_batched = dict()

comms_round = 100

#create optimizer
lr = 0.01 
loss='categorical_crossentropy'
metrics = ['accuracy'] # f1_score, precision_score
optimizer = SGD(lr=lr, 
                decay=lr / comms_round, 
                momentum=0.9
               ) 


# Function for preparing client data for training
def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    #get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    return local_count/global_count

def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def train_single_model(global_model,client,image_list,label_list):
    data = list(zip(image_list, label_list))
    random.shuffle(data)
    
    smlp_local = SimpleMLP()
    local_model = smlp_local.build(4356, 5)
    local_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    # get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()
    
    # set local model weight to the weight of the global model
    local_model.set_weights(global_weights)
    
    # batch the data
    clients_batched[client] = batch_data(data)
    
    #fit local model with client's data
    tf.autograph.experimental.do_not_convert(local_model.fit(clients_batched[client], epochs=1, verbose=0))
    
    steps = len(local_model.get_weights())
    st.write(steps)

    #scale the model weights and add to list
    scaling_factor = weight_scalling_factor(clients_batched, client)
    scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
    
    for i in range(0,len(scaled_weights)):
      arr = scaled_weights[i]
      lis = arr.tolist()
      j=0
      for x in lis:
        j+=1
        sheet.insert_row(x, j)
    
    #clear session to free memory after each communication round
    K.clear_session()

    return scaled_weights
    
import zipfile

app_mode = st.sidebar.selectbox('Select Page',['Home','About'])
if app_mode=='Home': 
    st.title('Classify Cervical Cancer') 
    st.markdown('upload photographs in bulk :')
    #adding a single-line text input widget
    name = st.text_input('Enter the id of your institution: ', '0')
    #displaying the entered text
    st.write('the name is ', name) 
    if(os.path.isdir('/content/content/Client_'+name)==True):
          shutil.rmtree('/content/content/Client_'+name)
    uploaded_file = st.file_uploader("Choose a file",type="zip")
    if uploaded_file is not None:
      with zipfile.ZipFile(uploaded_file, "r") as z:
        z.extractall(".")
   
      # To convert to a string based IO:
      # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

      # To read file as string:
      # string_data = stringio.read()

    
    #adding a button
    
    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        train = st.button('Train')
    with col2:
        test = st.button('Test')
    with col3:
        aggregate = st.button('Aggregate')

    if test:
       st.write('testing') #displayed when the button is clicked
    
    if train:
      
       
       #declear path to your data folder
       img_path = '/content/content/Client_'+name


       #get the path list using the path object
       image_paths = list(paths.list_images(img_path))

       #apply our function
       image_list, label_list = load(image_paths, verbose=10000)
           
       #binarize the labels
       lb = LabelBinarizer()
       label_list = lb.fit_transform(label_list)

       #split data into training and test set
       X_train, X_test, y_train, y_test = train_test_split(image_list, 
                                                    label_list, 
                                                    test_size=0.1, 
                                                    random_state=42)
       smlp_global = SimpleMLP()
       global_model = smlp_global.build(4356, 5)
       train_single_model(global_model,'aliya',image_list,label_list)
       st.write('training') #displayed when the button is clicked
    
    if aggregate:
       st.write('aggregating') #displayed when the button is clicked

if app_mode=='About': 
      st.title('Classify Cervical Cancer') 
