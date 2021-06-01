from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
from marshmallow import fields  #serialize python object
from marshmallow_sqlalchemy import ModelSchema  #serialize python object
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import bcrypt
import json

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from skimage import transform

load_dotenv('.env')

# Init app
app = Flask(__name__)

# Database
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('APP_SECRET_KEY')  #for socketio

# Init socketio
socketio = SocketIO(app, cors_allowed_origins='*', cors_credentials=False)

# Init database
db = SQLAlchemy(app)

# Init marshmallow
ma = Marshmallow(app)

# User Model
class User(db.Model):
    __tablename__ = "users"
    id_user = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(255))
    email = db.Column(db.String(255), unique=True)
    password = db.Column(db.String(255))
    role = db.Column(db.Integer)

    def __init__(self, fullname, email, password, role):
        self.fullname = fullname
        self.email = email
        self.password = password
        self.role = role
    
    def create(self):
        db.session.add(self)
        db.session.commit()
        return self
# Ambulance Model
class Ambulance(db.Model):
    __tablename__ = "ambulances"
    id_ambulance = db.Column(db.Integer, primary_key=True)
    ambulance_type = db.Column(db.String(255))
    ambulance_status = db.Column(db.Integer)
    ambulance_origin = db.Column(db.String())
    license_plate = db.Column(db.String(100))
    latitude = db.Column(db.String(50))
    longitude = db.Column(db.String(50))

    def __init__(self, ambulance_type, ambulance_status, ambulance_origin, license_plate, latitude, longitude):
        self.ambulance_type = ambulance_type
        self.ambulance_origin = ambulance_origin
        self.ambulance_status = ambulance_status
        self.license_plate = license_plate
        self.latitude = latitude
        self.longitude = longitude

# User Schema
class UserSchema(ModelSchema):
    class Meta(ModelSchema.Meta):
        model = User
        sqla_session = db.session
    
    id_user = fields.Number(dump_only=True)
    fullname = fields.String(required=True)
    email = fields.String(required=True)
    password = fields.String(required=True)
    role = fields.Number(required=True)
# Ambulance Schema
class AmbulanceSchema(ModelSchema):
    class Meta(ModelSchema.Meta):
        model = Ambulance
        sqla_session = db.session
    id_ambulance = fields.Number(dump_only=True)
    ambulance_type = fields.String(required=True)
    ambulance_status = fields.Integer(required=True)
    ambulance_origin = fields.String(required=True)
    license_plate = fields.String(required=True)
    latitude = fields.String(required=True)
    longitude = fields.String(required=True)

# User endpoint
@app.route('/api/user', methods=['POST'])
def store_user():
    data = request.get_json()

    pw_hash = bcrypt.hashpw(data['password'].encode('utf8'), bcrypt.gensalt())
    data['password'] = pw_hash

    user_schema = UserSchema()
    user = user_schema.load(data)
    result = user_schema.dump(user.create())
    result['password'] = None
    return make_response(jsonify({'success': 1, 'result': result}), 200)

@app.route('/api/user/login', methods=['POST'])
def login_user():
    email = request.json.get('email', None)
    password = request.json.get('password', None)

    if not email:
        return make_response(jsonify({'success': 0, 'msg': 'Missing email!'}), 400)
    if not password:
        return make_response(jsonify({'success': 0, 'msg': 'Missing password!'}), 400)

    user = User.query.filter_by(email=email).first()    
    
    if not user:
        return make_response(jsonify({'success': 0, 'msg': 'Email not found!'}), 404)

    if bcrypt.checkpw(password.encode('utf8'), user.password.encode('utf8')):
        return make_response(jsonify({
            'success': 1,
            'msg': 'Login Successful',
            'user': {
                'email': user.email,
                'fullname': user.fullname,
                'role': user.role
            }
        }), 200)
    else:
        return make_response(jsonify({'success': 0, 'msg': 'Wrong password!'}), 400)

    return make_response(jsonify({'email': email, 'password': password}))

# Ambulance endpoint
@app.route('/api/ambulance/<id>', methods=['GET'])
def get_ambulance(id):
    ambulance = Ambulance.query.get(id)    
    return make_response(jsonify({
        'id_ambulance': ambulance.id_ambulance,
        'type': ambulance.ambulance_type,
        'status': ambulance.ambulance_status,
        'origin': ambulance.ambulance_origin,
        'license_plate': ambulance.license_plate,
        'latitude': ambulance.latitude,
        'longitude': ambulance.longitude
    }), 200)

@app.route('/api/ambulance/<id>', methods=['PUT'])
def update_ambulance_status(id):
    status = int(request.json['status'])
    ambulance = Ambulance.query.get(id)

    if not ambulance:
        return make_response(jsonify({'success': 0, 'msg': 'Ambulance not found!'}), 404)
    
    ambulance.ambulance_status = status
    db.session.commit()

    return make_response(jsonify({
        'id_ambulance': ambulance.id_ambulance,
        'type': ambulance.ambulance_type,
        'status': ambulance.ambulance_status,
        'origin': ambulance.ambulance_origin,
        'license_plate': ambulance.license_plate,
        'latitude': ambulance.latitude,
        'longitude': ambulance.longitude
    }), 200)

# Machine Learning
# @app.route('/api/ambulance/predict', methods=['POST'])
def predict_process():
    # Code here
    #loading the model
    my_model = tf.keras.models.load_model('./TheATeam_model_ver2.h5', compile=True)

    #2 Load and pre-process image frames
    def load_frames(frame):
        frames = Image.open(frame)
        frames = np.array(frames).astype('float32')/255
        frames = transform.resize(frames, (224, 224, 3))
        frames = np.expand_dims(frames, axis=0)
        return frames
    
    #3 get video
    vidcap = cv2.VideoCapture('../ambulance.mp4')

    #4 converting video into frame image (jpg format)
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()

        if hasFrames:
            # Specify frame path file
            framePath = "../video-frames/"+str(count)+"_frame.jpg"
            # save frame as JPG file
            cv2.imwrite(framePath, image)
            
            # Predict Frame directly
            image = load_frames(framePath)
            result = my_model.predict(image)

            # Print ambulance detected or not and probability value
            # predict_result = (str(count)+") Ambulance Detected: {}".format("%.3f" % result[0][1]) if result[0][1]>0.03 
            #     else str(count)+") Ambulance not detected: {}".format("%.3f" % result[0][1]))
            predict_result = {
                "ambulance_detected": 1 if result[0][1] > 0.03 else 0,
                "frame_number": count,
                "precentage": "{}".format("%.3f" % result[0][1])
            }
            emit('predict_result', json.dumps(predict_result), broadcast=True)

        return hasFrames

    sec = 0
    frameRate = 5 # Capture image in second
    count=1
    success = getFrame(sec) # Initial function to get the frame and predict frame

    # Looping the function to get the frame and predict frame directly
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)
        

# Socketio
# @socketio.on('connect', namespace='/drive')
# def connected():
#     emit('status', {'status': 'Connected', 'success': 1})

# @socketio.on('disconnect', namespace='/drive')
# def disconnected():
#     emit('status', {'status': 'Disconnected', 'success': 0})

@socketio.on('ambulance_location', namespace='/drive')
def fwd_ambulance_location(data):    
    emit('broadcast_ambulance_location', data, broadcast=True)

@socketio.on('predict')
def predict(data):
    emit('predict_result', 'Predict Start', broadcast=True)
    predict_process()        
    emit('predict_result', 'Predict End', broadcast=True)



# Run server
if __name__ == "__main__":
    if os.environ.get('FLASK_ENV') == 'production':
        # app.run(debug=False, host='0.0.0.0')
        socketio.run(app, debug=False, host='0.0.0.0')
    else:
        # app.run(debug=True)
        socketio.run(app, debug=True)
        