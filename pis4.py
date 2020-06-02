from mss import mss

sct = mss()
import paho.mqtt.client as mqtt  # import the client1
import time
import random
import numpy as np
import io
from PIL import Image
import base64
import cv2

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

keep_going = True

img_data = np.zeros((224, 224))
new_arr = np.zeros((224))
x1 = 0
y1 = 0
x2 = .9999
y2 = .9999
keys_loaded = False
device_id = 'pi1234'
innate_password = 'testing1234'

pic = np.zeros((224, 224))

a = None
b = None

wait_val = True

old_topic = "iot_sec_proj_to_pi"
new_topic = "iot_sec_proj_to_pi"
out_topic = "iot_sec_proj_to_nano"

# load the innate public and private keys
with open("private_key_pi_to_nano.pem", "rb") as key_file:
    private_key_innate = serialization.load_pem_private_key(
        key_file.read(),
        password=None,
        backend=default_backend()
    )
with open("public_key_from_nano.pem", "rb") as key_file:
    public_key_innate = serialization.load_pem_public_key(
        key_file.read(),
        backend=default_backend()
    )

public_key = None
private_key = None


# converts image to the needed values
def img_conv_to(img, target_size):
    img_pre = np.uint8(np.array(img))
    # resize image to appropriate dimension
    img_resized = cv2.resize(np.array(img), target_size)
    # convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    img_arr = np.uint8(np.array(gray))
    # convert into a byte array
    data = bytes((img_arr))
    # return the two chunks
    return data


############
def on_message(client, userdata, message):
    global x1, y1, x2, y2
    global wait_val,out_topic,old_topic,new_topic
    print('message received')
    topic = message.topic
    payload = message.payload
    # make sure payload is correct size
    if len(payload) != 1024:
        print("bad sized payload")
        return
    # decrypt data
    decrypted = private_key_innate.decrypt(
        payload,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None))
    # check data integrity
    id_part = decrypted[27:34]
    bound_part = decrypted[0:27]
    pass_part =decrypted[34:44]
    pass_str=str(pass_part.decode('utf-8'))
    #check if password is correct
    if pass_str!="0123456789":
        print("incorrect password")
        return
    passed = check_data_integrity(bound_part)
    if not passed:
        print("failed data integrity")
        return
    mess = bound_part.decode("utf-8")
    out = id_part.decode('utf-8')
    out_topic=out
    x1 = float(mess[0:6])
    y1 = float(mess[7:13])
    x2 = float(mess[14:20])
    y2 = float(mess[21:27])
    points = [x1, y1, x2, y2]
    wait_val = False


def wait():
    global wait_val
    start = time.time()
    while wait_val:
        if (time.time() - start >= 30):
            print("timeout")
            return
        if not wait_val:
            break
    end = time.time()
    elapsed = end - start
    # print("message took ",str(elapsed)," to be received")
    wait_val = True


def check_data_integrity(data):
    passed = True
    expected_length = 27
    # check if data is correct length
    if len(data) != expected_length:
        Passed = False
    # check if data is in correct format
    # data should be in 1.1111_1.1111_1.1111_1.1111 format
    # 1.1111_1.1111_1.1111_1.1111
    # 0123456789
    ds = data.decode('utf-8')
    correct_format = True
    # check if puncuation marks and underscored are in correct place
    puncuation_marks = (ds[1] == '.') and (ds[8] == '.') and (ds[15] == '.') and (ds[22] == '.')
    underscores = (ds[6] == '_') and (ds[13] == '_') and (ds[20] == '_')
    all_numbers = True
    for i in range(expected_length):
        already_checked = i == 1 or i == 8 or i == 15 or i == 22 or i == 6 or i == 13 or i == 20
        if not already_checked:
            if not number_test(ds[i]):
                all_numbers = False

    less_than_one=ds[0]=='0'and ds[7]=='0' and ds[14]=='0' and ds[21]=='0'
    points=[all_numbers,puncuation_marks,underscores,less_than_one]
    passed = all_numbers and puncuation_marks and underscores
    return passed


def number_test(c):
    character = ord(c)
    if (character >= 48) and (character <= 57):
        return True
    else:
        return False


def image_encrypt(img):
    global public_key, nano_public_key
    global private_key, old_topic, new_topic
    global key, f
    # chunk=[]
    img_arr = np.array(img)
    img_data = bytes(img_arr)
    length = len(img_data)
    chunk = img_data
    # generate new random topic
    number = random.randint(1000000, 9999999)
    number_string=str(number)
    n_string = number_string.encode('ascii')
    old_topic = new_topic
    new_topic = str(number)
    # generate random key
    random_key = Fernet.generate_key()
    password='0123456789'.encode('ascii')
    message_to = random_key + n_string+password
    encryptor = Fernet(random_key)
    encrypted = encryptor.encrypt(chunk)
    encrypted_key = public_key_innate.encrypt(
        message_to,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    encrypted_sent = encrypted_key + encrypted
    p = 0
    return encrypted_sent


########################################

# broker_address="192.168.1.184"
# iot.eclipse.org
broker_address = "test.mosquitto.org"
print("creating new instance")
client = mqtt.Client("bdtest456")  # create new instance
client.on_message = on_message  # attach function to callback
print("connecting to broker")
client.connect('test.mosquitto.org', 1883)  # connect to broker
client.loop_start()  # start the loop
print("Subscribing to topic", "iot_sec_proj_to_pi")
client.subscribe("iot_sec_proj_to_pi")
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
while keep_going:
    begin = time.time()
    #Capture frame-by-frame
    ret, frame = cap.read()
    # if ret==False:
    #    print("false ret")
    printscreen=frame
    #printscreen = np.array(ImageGrab.grab(bbox=(100, 600, 324, 824)))
    # B = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # out_img = img_conv_to(printscreen, (224, 224))
    out_img = cv2.resize(printscreen, (224, 224))
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY)
    out_img = image_encrypt(out_img)
    print("sending out image")
    client.publish(out_topic, out_img, qos=1)
    print("sent data to ",str(out_topic))
    client.unsubscribe(old_topic)
    client.subscribe(new_topic)
    print("unsuscribing to", str(old_topic), " subscribing to ", str(new_topic))
    wait()
    # switch topics
    end = time.time()
    print(str(end - begin))
    print("proceeding")
    print("sent")
    s = np.shape(printscreen)
    height = s[0]
    width = s[1]

    xmin = int(x1 * width)
    ymin = int(y1 * height)
    xmax = int(x2 * width)
    ymax = int(y2 * height)

    new_img = printscreen[ymin:ymax, xmin:xmax]
    new_img2 = cv2.resize(new_img, (height, width))
    # im_v = cv2.vconcat([printscreen, new_img2])
    # im_v= cv2.cvtColor(im_v, cv2.COLOR_RGB2BGR)
    box = np.array(printscreen)
    # Start coordinate, here (5, 5)
    # represents the top left corner of rectangle
    start_point = (xmin, ymin)
    # Ending coordinate, here (220, 220)
    # represents the bottom right corner of rectangle
    end_point = (xmax, ymax)
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    image = cv2.rectangle(box, start_point, end_point, color, thickness)
    cv2.imshow('window', image)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        break

client.loop_stop()  # stop the loop
