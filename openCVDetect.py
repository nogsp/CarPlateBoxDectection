import cv2
import glob
import easyocr
import imutils
import numpy as np

def findPlate(imgRef, frames):
    reader = easyocr.Reader(['en'])
    boxWithText = []
    for f in frames:
        x,y,w,h = f[1]
        Cropped = imgRef[y:y+h, x:x+w]
        ans = reader.readtext(Cropped)
        for element in ans:
            (_, text, _) = element
            if len(text) > 1:
                boxWithText.append(f[1])
    
    return boxWithText

list_images = glob.iglob("images/*")

for image_title in list_images:
    img = cv2.imread(image_title, cv2.IMREAD_COLOR)
    img = imutils.resize(img, width=512)
    imgBase = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale
    gray = cv2.bilateralFilter(gray, 11, 17, 17) #Blur to reduce noise
    edged = cv2.Canny(gray, 30, 200) #Perform Edge detection
    cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img1=img.copy()
    cv2.drawContours(img1,cnts,-1,(0,255,0),3)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
    screenCnt = None #will store the number plate contour
    img2 = img.copy()
    selectedsFrames = []
    cv2.drawContours(img2,cnts,-1,(0,255,0),3) 

    # loop over contours
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        
        if((w>(h*0.75))):
            selectedsFrames.append([w*h,[x,y,w,h]])

    selectedsFrames = sorted(selectedsFrames, key=lambda x: x[0])
    screenRect = findPlate(imgBase, selectedsFrames)

    for select in selectedsFrames:
        x,y,w,h = select[1]
        cv2.rectangle(imgBase, (x,y), (x+w, y+h), (0, 0, 255), 2)

    if screenRect is None:
        detected = 0
        print ("No contour detected")
        
    else:
        detected = 1
        for b in screenRect:
            x,y,w,h = b
            cv2.rectangle(imgBase, (x,y), (x+w, y+h), (255, 0, 0), 2)

    edged = cv2.cvtColor(edged,cv2.COLOR_GRAY2RGB)
    img_and_magnitude = np.concatenate((imgBase, edged), axis=1)

    cv2.imshow('image', img_and_magnitude)
    cv2.waitKey(0)


cv2.destroyAllWindows() 