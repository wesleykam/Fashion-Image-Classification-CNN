# Starter code for CS 165B HW4

"""
Implement the testing procedure here. 

Inputs:
    Unzip the hw4_test.zip and place the folder named "hw4_test" in the same directory of your "prediction.py" file, your "prediction.py" need to give the following required output.

Outputs:
    A file named "prediction.txt":
        * The prediction file must have 10000 lines because the testing dataset has 10000 testing images.
        * Each line is an integer prediction label (0 - 9) for the corresponding testing image.
        * The prediction results must follow the same order of the names of testing images (0.png – 9999.png).
    Notes: 
        1. The teaching staff will run your "prediction.py" to obtain your "prediction.txt" after the competition ends.
        2. The output "prediction.txt" must be the same as the final version you submitted to the CodaLab, 
        otherwise you will be given 0 score for your hw4.


**!!!!!!!!!!Important Notes!!!!!!!!!!**
    To open the folder "hw4_test" or load other related files, 
    please use open('./necessary.file') instead of open('some/randomly/local/directory/necessary.file').

    For instance, in the student Jupyter's local computer, he stores the source code like:
    - /Jupyter/Desktop/cs165B/hw4/prediction.py
    - /Jupyter/Desktop/cs165B/hw4/hw4_test
    If he/she use os.chdir('/Jupyter/Desktop/cs165B/hw4/hw4_test'), this will cause an IO error 
    when the teaching staff run his code under other system environments.
    Instead, he should use os.chdir('./hw4_test').


    If you use your local directory, your code will report an IO error when the teaching staff run your code,
    which will cause 0 score for your hw4.
"""

import numpy as np
import os
import cv2
from natsort import natsorted


import tensorflow as tf

testDirectory = './hw4_test'
testing_data = []

for i in range(10000):
    img = str(i) + '.png'
    img_array = cv2.imread(os.path.join(testDirectory, img), cv2.IMREAD_GRAYSCALE)
    testing_data.append(img_array)

testing_data = np.array(testing_data).reshape(-1, 28, 28, 1)

model = tf.keras.models.load_model("image-classification-CNN-64x2.model")

predictions = model.predict([testing_data])

with(open('prediction.txt', 'w')) as f:
    for prediction in predictions:
        f.write(str(np.argmax(prediction)))
        f.write('\n')