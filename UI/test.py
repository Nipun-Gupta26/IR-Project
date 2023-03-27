from keras.models import load_model
from keras.utils import pad_sequences
import numpy as np
import pickle
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

if(__name__ == '__main__'):
    class Encoder():
        def __init__(self, model):
            self.model = model
        
        def encode(self, img_path):

        #get the image path
            img = image.load_img(img_path, target_size=(224, 224, 3)) #load the image
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) 
            x = preprocess_input(x) # convert to (1, 224, 224, 3) shaped nd-array

            ftr_vector = model.predict(x) #get the output as (batch_size, 7, 7, 2048)
            return ftr_vector
    
    model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224, 224,3), classes = 1000, classifier_activation = False)    
    enc = Encoder(model)

    image = enc.encode('1.jpg')
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

    print(caption)    