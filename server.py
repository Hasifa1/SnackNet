import os
import keras
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'vgg19_model (3).h5'

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model, preds=None):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    img_test = np.expand_dims(x, axis=0)

    classes = model.predict(img_test)

    print(classes)

    values = classes[0]
    index1 = np.argmax(values)
    print(index1)
    if index1 == 0:
        preds='''Sakinalu, a crispy Telangana snack,
                Made with rice flour and sesame seeds.
                '''
    elif index1 == 1:
        preds='Garalu,  is a popular Telangana snack made from rice flour, lentils, and spices. ' \
              'It is a savory doughnut-shaped snack that is deep-fried until golden brown and crispy.' \
              ' Garalu is often served with chutney or sambar.'
    elif index1==2:
        preds='Chegodi,a crunchy savory snack,' \
              'Made with rice flour and spices,' \
              'a delicious and versatile snack that can be enjoyed in many different ways. ' \
              'It is a popular snack in Telangana, but it is also becoming increasingly popular in other parts of India ' \
              'and the world.'
    elif index1==3:
        preds='Garijalu,is a popular Telangana snack item. ' \
              'It is a deep-fried, half-moon-shaped pastry made with rice flour, lentils, and spices.' \
              ' Garijalu is often stuffed with a sweet or savory filling, such as grated coconut, sugar, and cardamom'
    elif index1==4:
        preds='Sarvapindi, It is a savory pancake made with rice flour, chana dal, ginger, garlic, sesame seeds,' \
              ' curry leaves, and green chilies. The mixture of grains, pulses, and spices makes it a lip-smacking dish.'
    elif index1==5:
        preds='Gavvalu is a Telangana snack item. It is a crispy, crunchy shell-shaped snack made with rice flour,' \
              ' semolina (rava), and sugar or jaggery'
    return preds

@app.route('/', methods=['GET'])
def index():
        # Main page
        return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
        if request.method == 'POST':
            # Get the file from post request
            f = request.files['file']

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            # Make prediction
            preds = model_predict(file_path, model)
            result = preds
            return render_template('index.html',pred='{}'.format(result))
        return None
if __name__ == '__main__':
    app.run(debug=True)