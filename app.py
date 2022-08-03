import os
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array

model = load_model('best_model.h5')

# importing Flask and other modules
from flask import Flask, request, render_template, request
  
# Flask constructor
app = Flask(__name__)   
  
# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods = ["GET", "POST"])

def decode():
    if request.method == "POST":
        f = request.files['file']  
        f.save(f.filename)
        path = 'image.png'
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_at = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)
        img_dil = cv2.dilate(img_at, np.ones((2,2), np.uint8), iterations = 1)
        image_list = [img_dil[10:50, 30:50], img_dil[10:50, 50:70], img_dil[10:50, 70:90], img_dil[10:50, 90:110], img_dil[10:50, 110:130]]
        info = {0: '2', 4: '6', 13: 'm', 9: 'd', 3: '5', 14: 'n', 1: '3', 12: 'g', 6: '8', 2: '4', 10: 'e', 18: 'y', 11: 'f', 16: 'w', 15: 'p', 5: '7', 8: 'c', 17: 'x', 7: 'b'}
        Xdemo = []
        for i in range(5) :
            Xdemo.append(img_to_array(Image.fromarray(image_list[i])))
            
        Xdemo = np.array(Xdemo)
        Xdemo/= 255.0
            
        ydemo = model.predict(Xdemo)
        ydemo = np.argmax(ydemo, axis = 1)

        output ='\0'
        for res in ydemo:
            output+= str(info[res])
        return render_template('main.html', output=output)
    return render_template('main.html')

if __name__=='__main__':
   port = int(os.environ.get("PORT", 5000))
   app.run(host='0.0.0.0', port=port)
