from cgi import test
from flask import Flask, render_template, request
import os
from keras.preprocessing import image
import numpy as np
import keras
import tensorflow
import PIL
app = Flask(__name__)

UPLOAD_FOLDER = 'E:\\Portfolio projects\\ct scan dataset\\webApp\\images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def hello_world():
    return render_template('index.html')


@app.route("/predict", methods=['GET', "POST"])
def prediction():
	if request.method == 'POST':
	
		file1 = request.files['file1']
		path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
		file1.save(path)
		print(path)
	
		test_image=image.load_img(path,target_size=(256,256))
		test_image=image.img_to_array(test_image)
		print(test_image)
		test_image=np.expand_dims(test_image,axis=0)

		model = tensorflow.keras.models.load_model("model.h5")

		result=np.argmax(model.predict(test_image))
	
		return render_template('result.html' , result=result)



if __name__=='__main__':	
	app.run(debug=True)