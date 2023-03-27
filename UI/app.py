from flask import Flask, render_template, request
app = Flask(__name__)
from keras.models import load_model
from keras.utils import pad_sequences
import numpy as np
import pickle
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input


@app.route("/")
def home():    
    return render_template('home.html', image = '', caption = '')

@app.route('/smh', methods = ['GET', 'POST'])
def smh():
    if request.method == 'POST':  
        f = request.files['file']
        img_path = 'files/' + f.filename
        path = 'static/files/' + f.filename
        f.save(path)  
        caption = run(path)
        caption = caption[caption.find('sos') + 4: caption.find('eos')].capitalize()
        return render_template('home.html', image = img_path, caption = caption)        
    return "Error"

def run(path):
    model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224, 224,3), classes = 1000, classifier_activation = False)    
    enc = Encoder(model)

    image = enc.encode(path)
    model = load_model('model.h5')

    with open('word_to_index.pkl', 'rb') as file:
        word_to_index = pickle.load(file)

    with open('index_to_word.pkl', 'rb') as file:
        index_to_word = pickle.load(file)

    with open('test_images.pkl', 'rb') as file:
        test_images = pickle.load(file)

    cap_max_len = 51

    # image = test_images[9]    
    caption = 'sos'
    for i in range(cap_max_len):
        indexes = [word_to_index[key] for key in caption.split() if key in word_to_index]
        indexes = pad_sequences([indexes], maxlen = cap_max_len)
        y_pred = model.predict([image, indexes], verbose = 0)
        y_pred = np.argmax(y_pred)
        word_pred = index_to_word[y_pred]
        caption = caption + ' ' + word_pred
        if word_pred == 'eos':
            break

    return caption

@app.route('/temp')
def temp():
    s = 'sos yay at hot winter body row bloom with the villages of the hello fragrance hello round is a star allapartment residence with the hello fragrance hello round is the fragrance hello round is drying journaling and well to take a discussion to take a discussion outdoordog fit and warm fit and'
    return render_template('home.html', image = 'files/2023-01-03_17-06-48_UTC.jpg', caption = s)        

if __name__ == '__main__':  
    class Encoder():
        def __init__(self, model):
            self.model = model
        
        def encode(self, img_path):

        #get the image path
            img = image.load_img(img_path, target_size=(224, 224, 3)) #load the image
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) 
            x = preprocess_input(x) # convert to (1, 224, 224, 3) shaped nd-array

            ftr_vector = self.model.predict(x) #get the output as (batch_size, 7, 7, 2048)
            return ftr_vector
    app.run(debug=True)
    
