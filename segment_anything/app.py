import os

from PIL.Image import Image
from click import DateTime
from sqlalchemy.orm import backref
from werkzeug.utils import secure_filename

import predict
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from segment_anything import app,accuracy,split_all,split,SplitAllOneTime

from PIL import Image
import base64


def main():
    split_all.main("J118m1-.png")
    predict.main()

if __name__ == '__main__':
    main()
app = Flask(__name__)
CORS(app)

host = '127.0.0.1'
port = "3306"
username = 'root'
password = 'ydx56HW2004'
database = 'Stone'

app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql://{username}:{password}@{host}:{port}/{database}?charset=utf8"
print(app.config['SQLALCHEMY_DATABASE_URI'])

#app config(连接数据库)
db = SQLAlchemy(app)

class User(db.Model):
    __tablename__ = 'user'
    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(255), nullable=False)
    password = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), nullable=False)
    time = db.Column(db.DateTime, nullable=False)
    # 一对多关系
    records = db.relationship('Record', backref='User')
    def __init__(self, username, password, email, time):
        self.username = username
        self.password = password
        self.email = email
        self.time = time

class Record(db.Model):
    __tablename__ = 'record'
    record_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    model_id = db.Column(db.Integer, db.ForeignKey('model.model_id'), nullable=False)
    res_path = db.Column(db.String(255), nullable=False)
    source_path = db.Column(db.String(255), nullable=False)
    time = db.Column(db.DateTime, nullable=False)
    def __init__(self, user_id, model_id, res_path, source_path, time):
        self.user_id = user_id
        self.model_id = model_id
        self.res_path = res_path
        self.source_path = source_path
        self.time = time

class Model(db.Model):
    __tablename__ = 'model'
    model_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    model_name = db.Column(db.String(255), nullable=False)
    model_path = db.Column(db.String(255), nullable=False)
    epoch = db.Column(db.Integer, nullable=False)
    optimizer = db.Column(db.String(255), nullable=False)
    def __init__(self, model_id, model_name, model_path, epoch, optimizer):
        self.model_id = model_id
        self.model_name = model_name
        self.model_path = model_path
        self.epoch = epoch
        self.optimizer = optimizer
@app.route('/')
def index():
    main()
    user = User.query.get(1)
    print(user.records)
    return dict({'status': 'success'})

@app.route('/userLogin', methods=['POST'])
def login():
    request_data = request.get_json()
    print(request_data)
    users = User.query.filter_by(email = request_data['username'])
    if users.count() != 0:
        print(users.first().password)
        if users.first().password == request_data['password']:
            global USERNAME
            USERNAME = users.first().username
            return dict({'status': 'success','uuid':users.first().user_id,'token':"awdwdawafw",'name':users.first().username})

    return dict({'status': 'failed'})

@app.route('/addUser', methods=['POST'])
def signin():
    request_data = request.get_json()
    db.session.add(User(request_data['username'], request_data['password'], request_data['email'], DateTime()))
    db.session.commit()
    print("用户创建成功")
    return jsonify("用户创建成功")



UPLOAD_PATH = os.path.dirname(r"D:\EdgeDownLoad\d2-admin-start-kit-master\d2-admin-start-kit-master\src\assets\userImg\\")

@app.route('/addPic', methods=['POST'])
def addpic():
    file = request.files.get('file')
    filenameUser =  os.path.join(UPLOAD_PATH,USERNAME)
    if not os.path.isdir(filenameUser):
        # 创建文件夹
        os.mkdir(filenameUser)
    filenameFin = os.path.join(filenameUser,file.filename)
    file.save(filenameFin)
    return dict({'url': USERNAME+"/"+file.filename})

@app.route('/deletePic', methods=['POST'])
def deletePic():
    request_data = request.get_json()
    filename = os.path.join(UPLOAD_PATH, request_data['name'])
    print(request_data)
    if os.path.exists(filename):
        os.remove(filename)
        # print('成功删除文件:', file_name)
        return dict({"status":"success"})
    else:
        print('未找到此文件:', filename)
        return dict({"status": "failed"})

@app.route('/onePointPredict', methods=['POST'])
def onePointPredict():
    request_data = request.get_json()

    print(request_data)
    x = request_data['RX']
    y = request_data['RY']
    img = request_data['name']
    filenameUser = os.path.join(UPLOAD_PATH, img)
    save_path = os.path.join(UPLOAD_PATH, USERNAME)
    print(filenameUser)

    checkpoint = request_data['checkpoint']
    model_type = request_data['model_type']

    imgR = Image.open(filenameUser)
    print(imgR)
    w = imgR.width  # 图片的宽
    h = imgR.height  # 图片的高

    print(w)
    print(h)
    print(x)
    print(y)
    x = x*w*0.01
    y = y*h*0.01 #实际位置


    split.main(filenameUser,checkpoint,model_type,x,y,save_path)


    return dict({'url': USERNAME + "/" + "res-1.png"})

@app.route('/SeperateAll', methods=['POST'])
def SeperateAll():
    request_data = request.get_json()
    print(request_data)
    img = request_data['name']
    filenameUser = os.path.join(UPLOAD_PATH, img)
    save_path = os.path.join(UPLOAD_PATH, USERNAME)
    print(filenameUser)

    checkpoint = request_data['checkpoint']
    model_type = request_data['model_type']

    SplitAllOneTime.main(filenameUser,checkpoint,model_type,save_path)

    return dict({'url': USERNAME + "/" + "all.png"})

























