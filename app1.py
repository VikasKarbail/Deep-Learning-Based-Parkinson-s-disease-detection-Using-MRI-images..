import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, request
import sqlite3
import cv2
import shutil
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img,img_to_array
from keras import layers
from PIL import ImageTk, Image
from keras.models import load_model
import os 
import cv2
import glob

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from keras.layers import Convolution2D,Dense,MaxPool2D,Activation,Dropout,Flatten
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('userlog.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
   if request.method == 'POST':

       connection = sqlite3.connect('data.db')
       cursor = connection.cursor()

       name = request.form['name']
       password = request.form['password']

       query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
       cursor.execute(query)

       result = cursor.fetchall()

       if len(result) == 0:
           return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
       else:
           return render_template('userlog.html')

   return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
   if request.method == 'POST':

       connection = sqlite3.connect('data.db')
       cursor = connection.cursor()

       name = request.form['name']
       password = request.form['password']
       mobile = request.form['phone']
       email = request.form['email']
       
       print(name, mobile, email, password)

       command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
       cursor.execute(command)

       cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
       connection.commit()

       return render_template('index.html', msg='Successfully Registered')
   
   return render_template('index.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
 
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        

        shutil.copy("test/"+fileName, dst)
        image = cv2.imread("test/"+fileName)
        #color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        #apply the Canny edge detection
        edges = cv2.Canny(image, 100, 200)
        cv2.imwrite('static/edges.jpg', edges)
        #apply thresholding to segment the image
        retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)
        
        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        model=load_model('inception.h5')
        path='C:\\Users\\VISHWAS R\\OneDrive\\Desktop\\PARKINSON_INCEPTIONV3\\static\\images\\'+fileName
    ##    MODEL_NAME='keras_model.h5'
        
        model_out = (path,model)
        img = load_img(path,target_size=(224,224))
        plt.imshow(img)
       #img = load_img(path,target_size=(img_size,img_size))
        i = img_to_array(img)
       #im = preprocess_input(i)
        img = np.expand_dims(img,axis=0)
        model_out= model.predict(img)
        print(model_out)
        
        model_out = model_out[0]
        out = np.argmax(model_out)
        if np.argmax(model_out) == 0:
            str_label = "stage1"
            print("The predicted image of the stage1 is with a accuracy of {} %".format(model_out[out]*100))
            accuracy="The predicted image of the stage1 is with a accuracy of {}%".format(model_out[out]*100)

          
       
            
            
        elif np.argmax(model_out) == 1:
            str_label  = "stage2"
            print("The predicted image of the stage2 is with a accuracy of {} %".format(model_out[out]*100))
            accuracy="The predicted image of the stage2 is with a accuracy of {}%".format(model_out[out]*100)
            

        elif np.argmax(model_out) == 2:
            str_label  = "stage3"
            print("The predicted image of the stage3 is with a accuracy of {} %".format(model_out[out]*100))
            accuracy="The predicted image of the stage3 is with a accuracy of {}%".format(model_out[out]*100)
           

        elif np.argmax(model_out) == 3:
            str_label  = "stage1"
            print("The predicted image of the parkinson is with a accuracy of {} %".format(model_out[out]*100))
            accuracy="The predicted image of the parkinson is with a accuracy of {}%".format(model_out[out]*100)
           

        elif np.argmax(model_out) == 4:
            str_label  = "normal"
            print("The predicted image of the normal is with a accuracy of {} %".format(model_out[out]*100))
            accuracy="The predicted image of the normal is with a accuracy of {}%".format(model_out[out]*100)
            
        A=float(model_out[0])
        B=float(model_out[1])
        C=float(model_out[2])
        D=float(model_out[3])
        E=float(model_out[4])
        
        dic={'stage1':A,'stage2':B,'stage3':C,'parkinson':D,'normal':E}
        algm = list(dic.keys()) 
        accu = list(dic.values()) 
        fig = plt.figure(figsize = (5, 5))  
        plt.bar(algm, accu, color ='maroon', width = 0.3)  
        plt.xlabel("Comparision") 
        plt.ylabel("Accuracy Level") 
        plt.title("Accuracy Comparision between Parkinson detection....")
        plt.savefig('static/matrix.png')
        
                
            
                            

        return render_template('userlog.html', status=str_label,accuracy=accuracy,ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName,ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg",ImageDisplay4="http://127.0.0.1:5000/static/matrix.png")
        
    return render_template('index.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
