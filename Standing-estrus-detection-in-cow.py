import cv2 as cv
import numpy as np

#Load video
cap = cv.VideoCapture('dataset/cowip (1).mp4')
 
#Load Model
#Coco names file
classesFile = "coconame/coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

#Model files
modelConfiguration = "cfg/Mounting-cow-yolov4-tiny-detector.cfg"
modelWeights = "weights/Mounting-cow-yolov4-tiny-detector_final.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
confThreshold = 0.5
nmsThreshold= 0.5 
whT = 320

#Detecting objects and showing informations on the screen
def findObjects(outputs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
 
    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
 
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]          
        cv.rectangle(img, (x, y), (x+w,y+h), (0,255,0), 2)
        Y = y - 15 if y-15>15 else y+15
        cv.putText(img, f'{classNames[classIds[i]].upper()}',(x+5, Y+5), 0, 0.6, (255,255,255),2)
        print(f'{classNames[classIds[i]].upper()} [{int(confs[i]*100)}%]') #Class detected to show on windows
                  
 
while True:
    success, img = cap.read()
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs,img)
    imgResize = cv.resize(img, (960, 540)) #Resize video show                   
    cv.imshow("Output", imgResize)    

    key = cv.waitKey(1) #Press Esc button to stop processing
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
