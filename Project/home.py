from flask import Flask, render_template, request, Response
import dlib 
import torch
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt


import os
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
from datetime import datetime
import time
import numpy as np

from flask.helpers import url_for

##################################  remove background bằng devlabv3  #####################################
def load_model():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()
    return model

def make_transparent_foreground(pic, mask):
  # split the image into channels
  b, g, r = cv2.split(np.array(pic).astype('uint8'))
  # add an alpha channel with and fill all with transparent pixels (max 255)
  a = np.ones(mask.shape, dtype='uint8') * 255
  # merge the alpha channel back
  alpha_im = cv2.merge([b, g, r, a], 4)
  # create a transparent background
  bg = np.zeros(alpha_im.shape)
  # setup the new mask
  new_mask = np.stack([mask, mask, mask, mask], axis=2)
  # copy only the foreground color pixels from the original image where mask is set
  foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)
  return foreground

def remove_background(model,input_file):
  input_image = Image.open(input_file).convert('RGB')
  # input_image = Image.open(test_image_name).convert('RGB')
  input_image = input_image.resize((300,300))
  preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  input_tensor = preprocess(input_image)
  input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

  # move the input and model to GPU for speed if available
  if torch.cuda.is_available():
      input_batch = input_batch.to('cuda')
      model.to('cuda')

  with torch.no_grad():
      output = model(input_batch)['out'][0]
  output_predictions = output.argmax(0)

  # create a binary (black and white) mask of the profile foreground
  mask = output_predictions.byte().cpu().numpy()
  background = np.zeros(mask.shape)
  bin_mask = np.where(mask, 255, background).astype(np.uint8)

  foreground = make_transparent_foreground(input_image ,bin_mask)

  return foreground



def custom_background(background_file, foreground):


  final_foreground = Image.fromarray(foreground)
  background = Image.open(background_file).convert('RGB')

  background = background.resize((300,300))
  x = (background.size[0]-final_foreground.size[0])/2
  y = (background.size[1]-final_foreground.size[1])/2
  box = (x, y, final_foreground.size[0] + x, final_foreground.size[1] + y)
  crop = background.crop(box)
  final_image = crop.copy()
  # put the foreground in the centre of the background
  paste_box = (0, final_image.size[1] - final_foreground.size[1], final_image.size[0], final_image.size[1])
  final_image.paste(final_foreground, paste_box, mask=final_foreground)
  return final_image



deeplab_model = load_model()




############################## END #########################################################################

################### remove phong xanh ##############################
def tack( p_persion ,p_bg):
    image = cv2.imread(p_bg,1)
    ##image = cv2.resize(image, (300, 300))

    frame = cv2.imread(p_persion,1)
    ##frame = cv2.resize(frame, (300, 300))
    
    image = cv2.resize(image, (frame.shape[1], frame.shape[0]))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    x = np.array([0,137,123])
    y = np.array([107,255,255])

    Mask = cv2.inRange(hsv,x,y)
    res = cv2.bitwise_and(frame, frame, mask = Mask)
    f = frame - res
    not_mask = cv2.bitwise_not(Mask)
    img_gb2 = cv2.bitwise_and(image,image,mask= Mask)
    imt = f + img_gb2
    
    return imt
#################### END ###########################################
ind = 0
segmentor = SelfiSegmentation()
def remove_background2(frame , path):
    bg = cv2.imread(path)
    bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))
    imgOut = segmentor.removeBG(frame, bg, threshold=0.78)
    return imgOut




app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = "static"


##############SERVICE 1###########################################
@app.route("/", methods =['POST', 'GET'])
def home():
    if request.method == "GET":
        return render_template("index.html")
    else:
        return render_template("index.html")

@app.route("/inner-page" , methods = ['POST', 'GET'])
def inner_page():
    global ind
    path_background = "/static/assets/back_ground/nen-trang-47.jpg"    
    if request.method == "GET":
        return render_template("inner-page.html", img_background= path_background)
    else:
        try:
            next = request.form["next"]
            if (next == "❯" ):
                ind +=1
            elif (next =="❮"):
                ind -=1
        except:
            ind = ind
        if (ind%5) == 0:
            path_background = "static/nen-trang-47.jpg"
        elif ind%5==1:
            path_background = "static/img_bg_1.jpg"
        elif ind%5==2:
            path_background = "static/img_bg_2.jpg"
        elif ind%5==3:
            path_background = "static/img_bg_3.jpg"
        else:
            path_background = "static/img_bg_4.jpg"
        
            
        background_file = request.files["file2"]
        if (background_file.filename != ""):
                print("file2:" + background_file.filename)
                path_background = os.path.join(app.config['UPLOAD_FOLDER'], background_file.filename)
                background_file.save(path_background)
      
        try:
            image_file = request.files['file']        
            # path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            now = datetime.now()
            current_time = now.strftime("%H_%M_%S")
            path_to_save = "static/assets/save_image/" + current_time +image_file.filename
            image_file.save(path_to_save)
            print(path_to_save)
            print(path_background)
            #final_image = remove_background3(path_to_save)
        except:
            return render_template("inner-page.html", img_background = path_background)
        btn_submit = request.form["btn_RM"]
        if (btn_submit == "RM1"):
            foreground= remove_background(deeplab_model, path_to_save)  ### lấy path image
            final_image = custom_background( path_background , foreground)    ## lấy ảnh của background
            final_image.save(path_to_save)  
        else:
            print("có qua đây")
            final_image= tack(path_to_save, path_background)
            cv2.imwrite(path_to_save, final_image)

        #save img_destination

        
        #img = remove_background(path_to_save)             
        return  render_template("inner-page.html", msg =path_to_save, user_image = path_to_save, img_background = path_background)
    


#########################  SERVICE 2 ###############################
### video
flag = True
camera = cv2.VideoCapture(0)
capture = False
def generate_frame(path_bg):        
    global flag
    global capture
    global path_save
    while flag == True:
        success, frame = camera.read()
        if not success:
            break
        else:
            
            ## có frame 
            frame = remove_background2(frame, path_bg)
            # print("FRame")
            
            if capture == True:  
                print("có capture")
                now = datetime.now()
                current_time = now.strftime("%H_%M_%S")

                path_save = "static/assets/save_image/img" + current_time +".jpg"
                cv2.imwrite(path_save, frame)
                camera.release()
                flag = False
                capture = False
                


            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            key =cv2.waitKey(20)
        if key == ord('q'):
    	    break
        
        yield (b'--frame\r\n' 
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        


@app.route('/inner-page-2', methods = ["POST", "GET"])
def index():
    global flag
    global capture
    global camera
    global ind
    global path_background
    path_background = "/static/assets/back_ground/nen-trang-47.jpg"    
    if request.method == "GET":
        return render_template("inner-page-2.html", img_background = path_background)
    else:
        try:
            next = request.form["next"]
            if (next == "❯" ):
                ind +=1
            elif (next =="❮"):
                ind -=1
        except:
            ind = ind
        if (ind%5) == 0:
            path_background = "static/nen-trang-47.jpg"
        elif ind%5==1:
            path_background = "static/img_bg_1.jpg"
        elif ind%5==2:
            path_background = "static/img_bg_2.jpg"
        elif ind%5==3:
            path_background = "static/img_bg_3.jpg"
        else:
            path_background = "static/img_bg_4.jpg"
        
            
        
        background_file = request.files["file2"]
        if (background_file.filename != ""):
                print("file2:" + background_file.filename)
                path_background = os.path.join(app.config['UPLOAD_FOLDER'], background_file.filename)
                background_file.save(path_background)
        try:        
            btn = request.form["button"]
            if btn == "STOP":
                print("STOP")
                flag = False
                camera.release()
                return render_template("inner-page-2.html", img_background= path_background)
            if btn == "START":
                print("START")
                flag = True
                camera = cv2.VideoCapture(0)
                return render_template("inner-page-2.html", url_img = url_for('video'))
            if btn =="CAPTURE":
                capture = True            
                if capture == True  :
                    time.sleep(1)
                    print("first")
                    return render_template("inner-page-2.html", url_img = path_save, img_background= path_background)
        except:
            return render_template("inner-page-2.html", img_background = path_background)
        return render_template("inner-page-2.html", img_background = path_background)
            
            
        

            
@app.route('/inner-page-2/video', methods = ["POST", "GET"])
def video():
    global camera 
    camera = cv2.VideoCapture(0)
    if request.method == "GET":
        if flag== False:
            camera.release()
        else:
            camera = cv2.VideoCapture(0)
        return Response(generate_frame(path_background),mimetype='multipart/x-mixed-replace; boundary=frame') 

    






## start server 
if __name__ == '__main__':
    app.run(host ='', port=9999, debug= True)
