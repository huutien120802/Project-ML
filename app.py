from flask import Flask,render_template, request, jsonify, Response
from keras.models import load_model
import numpy as np
import tensorflow_hub as hub
import cv2
import tensorflow as tf
import base64
from cvzone.HandTrackingModule import HandDetector

app = Flask(__name__, template_folder='',static_folder='')
# app = Flask(__name__)
app.config["SECRET_KEY"] = "abcxyz123"

bit_model_url = "https://tfhub.dev/google/bit/m-r50x1/1"
bit_module = hub.KerasLayer(bit_model_url)
model_predict_animal = load_model('./Predict Animal/bit-custom.h5', custom_objects={'KerasLayer':bit_module})

model_predict_traffic = load_model('./Predict Traffic Sign/traffic_model.h5')

model_predict_digit = load_model("./Predict Digit/mnist.h5")

sign_model = load_model('./Regconize Sign Language/sign_model.h5')

camera = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 30

def startCam():
    global camera
    if camera == None:
        camera = cv2.VideoCapture(0)

def stopCam():
    global camera
    if camera != None:
        camera = camera.release()
        cv2.destroyAllWindows()

@app.route('/aboutus')
def aboutus():
    stopCam()
    return render_template('aboutus.html')

@app.route('/')
def home():
    stopCam()
    return render_template('index.html')

@app.route('/animal',methods = ['GET', 'POST'])
def animal():
    stopCam()
    if request.method == 'POST':
        class_names = ['Cat','Cow','Dog','Elephant','Gorilla','Hippo','Monkey','Panda','Tiger','Zebra']
        if request.files["animal_input"]:
            image = request.files["animal_input"]
            image_path = './Predict Animal/input_animal.jpeg'
            image.save(image_path)
            image = cv2.imread(image_path)
            preds = model_predict_animal.predict(image[np.newaxis,...])[0]
            pred_class = class_names[np.argmax(preds)]
            confidence_score = np.round(preds[np.argmax(preds)],2)
        else:
            pred_class = None
            confidence_score = None
            image_path = None
        return render_template('animal.html',pred_class=pred_class,confidence_score=confidence_score,image_path=image_path)
    return render_template('animal.html')

@app.route('/traffic',methods = ['GET','POST'])
def traffic():
    stopCam()
    if request.method == 'POST':
        class_names = ['Speed limit (5km/h)',
                        'Speed limit (15km/h)',
                        'Dont Go straight',
                        'Dont Go Left',
                        'Dont Go Left or Right',
                        'Dont Go Right',
                        'Dont overtake from Left',
                        'No Uturn',
                        'No Car',
                        'No Horn',
                        'Speed limit (40km/h)',
                        'Speed limit (50km/h)',
                        'Speed limit (30km/h)',
                        'Go straight or right',
                        'Go straight',
                        'Go Left',
                        'Go Left or right',
                        'Go Right',
                        'Keep Left',
                        'Keep Right',
                        'Roundabout mandatory',
                        'Watch out for cars',
                        'Horn',
                        'Speed limit (40km/h)',
                        'Bicycles crossing',
                        'Uturn',
                        'Road Divider',
                        'Traffic signals',
                        'Danger Ahead',
                        'Zebra Crossing',
                        'Bicycles crossing',
                        'Children crossing',
                        'Dangerous curve to the left',
                        'Dangerous curve to the right',
                        'Speed limit (50km/h)',
                        'Unknown1',
                        'Unknown2',
                        'Unknown3',
                        'Go right or straight',
                        'Go left or straight',
                        'Unknown4',
                        'ZigZag Curve',
                        'Train Crossing',
                        'Under Construction',
                        'Unknown5',
                        'Speed limit (60km/h)',
                        'Fences',
                        'Heavy Vehicle Accidents',
                        'Unknown6',
                        'Give Way',
                        'No stopping',
                        'No entry',
                        'Unknown7',
                        'Unknown8',
                        'Speed limit (70km/h)',
                        'speed limit (80km/h)',
                        'Dont Go straight or left',
                        'Dont Go straight or Right']
        if request.files["traffic_input"]:
            image = request.files["traffic_input"]
            image_path = './Predict Traffic Sign/input_traffic.jpeg'
            image.save(image_path)
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image)
            image = tf.image.resize(image, (224,224))
            preds = model_predict_traffic.predict(np.expand_dims(image, axis=0))
            pred_class = class_names[np.argmax(preds)]
            # confidence_score = np.round(preds[np.argmax(preds)],2)
        else:
            pred_class = None
            # confidence_score = None
            image_path = None
        return render_template('traffic.html',pred_class=pred_class,image_path=image_path)
    return render_template('traffic.html')


@app.route('/digit',methods = ['GET','POST'])
def digit():
    stopCam()
    if request.method == 'POST':
        data = request.get_json()
        imgBase64 = data['image']
        imBytes = base64.b64decode(imgBase64)
        with open('./Predict Digit/digit_input.png','wb') as digit:
            digit.write(imBytes)
        digit_data = cv2.imread('./Predict Digit/digit_input.png',0)
        digit_data = cv2.resize(digit_data,(28,28))

        digit_data = np.reshape(digit_data,(28,28,1))
        digit_data =digit_data.astype('float')/ 255
        predicted = np.argmax(model_predict_digit.predict(np.array([digit_data])),axis=-1)
        
        return jsonify({
            'prediction': f'{predicted[0]}',
            'status': True
        })
    else:
        return render_template('number.html')

def generate_frames():
    while True:
        class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                       'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
        # read the camera frame
        success, frame = camera.read()
        frame = cv2.flip(frame, 1)
        hands, img = detector.findHands(frame.copy())
        if hands and success:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            if h/w > 1:
                off = int((h - w) / 2)

                img = frame[y - offset: y + h + offset,
                            x - off - offset:x+h - off + offset]
                frame = cv2.rectangle(frame,(x - off - offset,y - offset),(x+h - off + offset,y + h + offset),(141, 94, 231), 2)
                if img is None:
                    print('Wrong path:')
                else:
                    try:
                        img = cv2.resize(img, (28, 28))
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        gray = gray.reshape(-1, 28, 28, 1)
                        gray = gray / 255

                        predict = sign_model.predict(gray)
                        predict = np.argmax(predict, axis=1)
                        if predict[0] < 25:
                            frame = cv2.putText(
                                frame, f'{class_names[predict[0]]}', (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (141, 94, 231), 2)
                    except:
                        frame = cv2.putText(
                                frame, 'ERROR', (530, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (141, 94, 231), 2)

            else:
                off = int((w - h) / 2)
                img = frame[y+off-offset:y+w+off+offset, x-offset:x+w+offset]
                frame = cv2.rectangle(frame,(x-offset,y+off-offset),(x+w+offset,y+w+off+offset),(141, 94, 231), 2)


                if img is None:
                    print('Wrong path:')
                else:
                    try:
                        img = cv2.resize(img, (28, 28))
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        gray = gray.reshape(-1, 28, 28, 1)
                        gray = gray / 255

                        predict = sign_model.predict(gray)
                        predict = np.argmax(predict, axis=1)
                        if predict[0] < 25:
                            frame = cv2.putText(
                                frame, f'{class_names[predict[0]]}', (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (141, 94, 231), 2)
                    except:
                        frame = cv2.putText(
                                frame, 'ERROR', (530, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (141, 94, 231), 2)
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sign_language')
def sign_language():
    startCam()
    return render_template('sign_language.html')

if __name__ == "__main__":
    app.run(debug=False,port=5000)