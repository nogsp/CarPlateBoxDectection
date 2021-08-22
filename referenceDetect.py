import cv2
import glob


list_images = glob.iglob("images/*")

for image_title in list_images:
    img = cv2.imread(image_title, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale
    gray = cv2.bilateralFilter(gray, 11, 17, 17) #Blur to reduce noise
    edged = cv2.Canny(gray, 30, 200) #Perform Edge detection
    cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img1=img.copy()
    cv2.drawContours(img1,cnts,-1,(0,255,0),3)
    #cv2.imshow("img1",img1)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
    screenCnt = None #will store the number plate contour
    img2 = img.copy()
    cv2.drawContours(img2,cnts,-1,(0,255,0),3) 
    #cv2.imshow("img2",img2) #top 30 contours

    # loop over contours
    for c in cnts:
      # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) > 3 and len(approx) < 7:#chooses contours with 4 corners
                    screenCnt = approx
                    break
                #draws the selected contour on original image        
    cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
    cv2.imshow("Final image with plate detected",img)
    cv2.waitKey(0)


cv2.destroyAllWindows() 