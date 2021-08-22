#@programming_fever
import numpy as np
import cv2
import imutils
import glob
import easyocr
from matplotlib import pyplot as plt
from numpy.fft import fft
#from scipy.ndimage.filters import gaussian_filter
from graham import graham_scan
from graham import pt

def highligthPixels(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j] > 128):
                img[i][j] = max(128, img[i][j] - img[i][j]*0.15)
            else:
                img[i][j] = min(128, img[i][j] + img[i][j]*0.15)
    return img

def sharpFilter(img):
    ker = np.array([
                [-1, -1, -1],
                [-1, 18, -1],
                [-1, -1, -1]])
    ker = (1.0/10.0) * ker
    img = cv2.filter2D(img,-1,ker, delta=0)
    return img

def fftFilter(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)

    #sz = 25
    #rows, cols = img.shape
    #crow,ccol = rows//2 , cols//2
    #fshift[crow-sz:crow+(sz+1), ccol-sz:ccol+(sz+1)] = 0

    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    return img_back

def normalizer(img):
    normalizedImg = np.zeros(img.shape)
    normalizedImg = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    (thresh, blackAndWhiteImage) = cv2.threshold(normalizedImg, 127, 255, cv2.THRESH_BINARY_INV)
    img = blackAndWhiteImage

    # img[img < 0] = 0
    img = img.astype(np.uint8)
    return img

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

def getcontours(img, img_back, imgRef):
    contours = cv2.findContours(img_back.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours,key=cv2.contourArea, reverse = True)[:30]
    screenRect = None
    selectedsFrames = []
    
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        
        if((w>(h*0.75))):
            selectedsFrames.append([w*h,[x,y,w,h]])

        '''
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        #print(approx)
        #contourHull = cv2.convexHull(c)
        #cv2.drawContours(img, [c], -1, (0, 0, 255), 3)
        #cv2.drawContours(img, [c], -1, (0, 0, 255), 3)
        #if(len(approx) == 5 or len(approx) == 4):
        #    cv2.drawContours(img, [c], -1, (0, 0, 255), 3)
        lisandra = []
        for p in approx:
            lisandra.append(pt(p[0][0], p[0][1]))

        lisandraCopy = lisandra.copy()

        hull = graham_scan(lisandra)
        allPointsInHull = True

        for p in lisandraCopy:
            foundInArray = False
            for q in hull:
                if(p == q):
                    foundInArray = True

            if(not foundInArray):
                allPointsInHull = False
                break

        if(allPointsInHull):
            if((len(approx) > 3) and len(approx) < 7):
                screenCnt = approx
                cv2.drawContours(img, [approx], -1, (255, 0, 0), 3)
                #break
        '''
    selectedsFrames = sorted(selectedsFrames, key=lambda x: x[0])
    screenRect = findPlate(imgRef, selectedsFrames)

    for select in selectedsFrames:
        x,y,w,h = select[1]
        cv2.rectangle(imgRef, (x,y), (x+w, y+h), (0, 0, 255), 2)

    if screenRect is None:
        detected = 0
        print ("No contour detected")
        
    else:
        detected = 1
        for b in screenRect:
            x,y,w,h = b
            cv2.rectangle(imgRef, (x,y), (x+w, y+h), (255, 0, 0), 2)

    #print(img_back)
    #print(img_back.dtype)
    return imgRef, img_back

def someFilters(img):
        #kernelmatrix = np.ones((3,3),np.float32)/25
        #img = cv2.filter2D(img, -1, kernelmatrix)
        #img = cv2.bilateralFilter(img, 15, 40, 20)
        #img = gaussian_filter(img, sigma=1)
        #img = cv2.bilateralFilter(img, 13, 15, 15) 
        #img = cv2.blur(img,(5,5))
        return img

def main():
    list_images = glob.iglob("images/*")
    for image_title in list_images:
        # Open an treat image ###########
        img = cv2.imread(image_title, cv2.IMREAD_COLOR)
        img = imutils.resize(img, width=512)
        imgBase = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # highlight black and white points
        img = highligthPixels(img)

        # sharpFilter using statistical method to sharp image
        img = sharpFilter(img)
        img = sharpFilter(img)
        img = sharpFilter(img)
        
        #img = someFilters(img)

        # Using FFT to detect border
        img_back = fftFilter(img)

        # normalizing to remove negative pixels
        img_back = normalizer(img_back)

        #finding contours in the image
        img, img_back = getcontours(img, img_back, imgBase)
        img_back = cv2.cvtColor(img_back,cv2.COLOR_GRAY2RGB)
        
        img_and_magnitude = np.concatenate((imgBase, img_back), axis=1)

        cv2.imshow('image_title', img_and_magnitude)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    '''
    mask = np.zeros(img.shape,np.uint8)
    new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    new_image = cv2.bitwise_and(img,img,mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = img[topx:bottomx+1, topy:bottomy+1]

    cv2.imshow('Cropped',Cropped)
    cv2.waitKey(0)
    '''
    
if __name__ == "__main__":
    main()