import paho.mqtt.client as mqtt  # import the client1
import time
import random
import numpy as np
import io
from PIL import Image
import base64
import cv2
import tensorflow as tf

# importing encryption libaries
from cryptography.fernet import Fernet
# assymetric creyptography stuff
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa

data = 'placeholder'
wait_val = True
request = False
keys_loaded = False
pic = np.zeros((224, 224))
img_data = np.zeros((224, 224))
new_arr = np.zeros((224))
device_id = 'nano1234'
public_key_to_send = None
private_key = None
public_keys = []
client_ids = []
password = '0123456789'
old_topic="iot_sec_proj_to_pi"
new_topic="iot_sec_proj_to_pi"
out_topic="iot_sec_proj_to_nano"


# load the innate public and private keys
with open("private_key_nano_to_pi.pem", "rb") as key_file:
    private_key_innate = serialization.load_pem_private_key(
        key_file.read(),
        password=None,
        backend=default_backend()
    )
with open("public_key_from_pi.pem", "rb") as key_file:
    public_key_innate = serialization.load_pem_public_key(
        key_file.read(),
        backend=default_backend()
    )

# detector = MTCNN()
model = tf.keras.models.load_model("boundingGray3.h5",
                                   custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU(alpha=0.2)}, compile=False)
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mse'])
blank = np.expand_dims(np.zeros((224, 224, 1)) / 255, axis=0)
print("test running nn")
model.predict(blank)
print("finished")


def image_decode(data):
    im = Image.frombytes('L', (224, 224), data)
    img_arr = np.uint8(np.array(im))
    return img_arr


def wait():
    global wait_val
    global client
    start = time.time()
    while wait_val:
        p = 0
    end = time.time()
    elapsed = end - start
    print("message took ", str(elapsed), " to be received")
    wait_val = True


def extract(img, model):
    img_arr = np.expand_dims(np.array(img) / 255, axis=0)
    img_arr = np.expand_dims(img_arr, axis=3)
    pred = model.predict(img_arr)
    s = np.shape(img)
    height = s[0]
    width = s[1]

    prediction = pred[0]
    prediction = np.array(prediction)
    xmin = limits(prediction[0])
    ymin = limits(prediction[1])
    xmax = limits(prediction[2])
    ymax = limits(prediction[3])

    xmin, ymin, xmax, ymax = fix_points(xmin, ymin, xmax, ymax)

    x1 = "{:.4f}".format(xmin)
    y1 = "{:.4f}".format(ymin)
    x2 = "{:.4f}".format(xmax)
    y2 = "{:.4f}".format(ymax)
    return_string = x1 + '_' + y1 + '_' + x2 + '_' + y2
    return return_string


def limits(a):
    b = a
    if a <= 0:
        b = 0
    elif a >= 1:
        b = .9999
    return b


def fix_points(x1, y1, x2, y2):
    if x1 >= x2:
        x1 = 0
        x2 = .9999
    if y1 >= y2:
        y1 = 0
        y2 = .9999

    if x2 <= 0:
        x1 = 0
        x2 = .9999

    if y2 <= 0:
        y1 = 0
        y2 = .9999
    return x1, y1, x2, y2

def image_decrypt(data):
    global f,compromised,private_key,private_key_innate,out_topic
    expected_length=68024
    pass_part=data[0:1024]
    #ensure if data is not compromised
    #decrypt the password part
    enc = private_key_innate.decrypt(
        pass_part,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    keypart=enc[0:44]
    id_part=enc[44:51]
    pass_part=enc[51:61].decode('utf-8')
    id_str=id_part.decode('utf-8')
    out_topic=str(id_str)
    decryptor=Fernet(keypart)
    image_part=data[1024:len(data)]
    original_image=decryptor.decrypt(image_part)
    im=Image.frombytes('L',(224,224),original_image)
    img_arr=np.uint8(np.array(im))
    password=str(pass_part)
    return img_arr , password,out_topic


############
def on_message(client_in, userdata, message):
    print("received message")
    global wait_val, out_id, public_keys
    global client, request, public_key
    global pic, model, data,out_topic
    #check if data correct size
    if len(message.payload)!=68024:
        print("received data is wrong size")
        return

    topic = message.topic
    payload=message.payload
    pic,password_0,OT=image_decrypt(payload)
    if password_0 !="0123456789":
        print("incorrect password in image data")
        return

    out_id=OT
    request=True
    p=request
    wait_val=False

# 50176
def check_data_integrity(data):
    passed = True
    expected_lenth = 50176
    if len(data) != expected_lenth:
        passed = False
    return passed


########################################
# D=security_check()
broker_address = "iot.eclipse.org"
print("creating new instance")
client = mqtt.Client("bdtest457")  # create new instance
client.on_message = on_message  # attach function to callback
print("connecting to broker")
client.connect('test.mosquitto.org', 1883)  # connect to broker
client.loop_start()  # start the loop
print("connected")
print("subscribing to search topic")
client.subscribe('iot_sec_proj_to_nano')
# subscribe to key exchange
# create its keys
print("waiting for pi")
wait()
while 1 == 1:
    if request:
        #pic = image_decode(data)
        print("predicting")
        bound_data = extract(pic, model)
        b_data=bound_data.encode('ascii')
        print("finished prediction")
        number=str(random.randint(1000000,9999999))
        n_string=number.encode('ascii')
        password="0123456789".encode('ascii')
        message_to=b_data+n_string+password
        old_topic=new_topic
        new_topic=str(number)
        #encrypt message back
        encrypted = public_key_innate.encrypt(
            message_to,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        client.publish(out_topic, encrypted, qos=1)
        print("sent data to topic", str(out_topic))
        client.unsubscribe(old_topic)
        client.subscribe(new_topic)
        print("unsuscribing to", str(old_topic), " subscribing to ", str(new_topic))
        request = False