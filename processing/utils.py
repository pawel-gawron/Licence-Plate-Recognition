import numpy as np
import cv2
import os
import glob

import tensorflow as tf
from tensorflow import keras

# print(tf.version.VERSION)

# # Tensorflow normal model
# lettersRecognitionModel = tf.keras.models.load_model('content/saved_model/my_model')

localization = os.path.abspath(os.path.dirname(__file__))
master_catalog = os.path.abspath(os.path.join(localization, '..'))

model_path = 'model/model.tflite'
model_path = os.path.join(master_catalog, model_path)

# Tensorflow lite model
lettersRecognitionModel = tf.lite.Interpreter(model_path=model_path)
lettersRecognitionModel.allocate_tensors()

inputDetails = lettersRecognitionModel.get_input_details()
outputDetails = lettersRecognitionModel.get_output_details()
inputShape = inputDetails[0]['shape']

def increase_brightness_contrast(input_img, brightness, contrast):    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def perform_processing(image: np.ndarray) -> str:
    # print(f'image.shape: {image.shape}')

    gaussianWindow = 7
    gaussianDeviation = 3.5

    gaussianWindow1 = 7
    gaussianDeviation1 = 0.1

    gaussianWindow2 = 21
    gaussianDeviation2 = 1.1
    IMG_WIDTH, IMG_HEIGHT = 32, 40
    BRIGHTNESS, CONTRAST = 15, 30

    letterLabels = np.array(['0', '1', '2', '3', '4', '5', '6',
                             '7', '8', '9', 'A', 'B', 'C', 'D',
                             'E', 'F', 'G', 'H', 'I', 'J', 'K',
                             'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                             'S', 'T', 'U', 'V', 'W', 'X', 'Y',
                             'Z'])
    
    imageRatio = image.shape[1]/image.shape[0]

    dim = (960, int(960 / imageRatio))

    image = cv2.resize(image, (dim), interpolation = cv2.INTER_AREA)

    mask = np.zeros_like(image) # Create mask where white is what we want, black otherwise
    out = np.zeros_like(image) # Extract out the object and place into output image

    imageCopy = image.copy()
    imageCopy = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2GRAY)

    widthImage = int(imageCopy.shape[1])
    heightImage = int(imageCopy.shape[0])

    imageCopy = increase_brightness_contrast(imageCopy, BRIGHTNESS, CONTRAST)

    imgBlur = cv2.GaussianBlur(imageCopy, (gaussianWindow, gaussianWindow), gaussianDeviation)
    gaussianBlur1 = cv2.GaussianBlur(imgBlur, (gaussianWindow1, gaussianWindow1), gaussianDeviation1)
    gaussianBlur2 = cv2.GaussianBlur(imgBlur, (gaussianWindow2, gaussianWindow2), gaussianDeviation2)
    ret, thg = cv2.threshold(gaussianBlur2-gaussianBlur1, 160, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(thg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    finalLetters = []

    for i in range(len(contours)):

        if hierarchy[0][i][2] == -1:
                    continue

        x ,y, w, h = cv2.boundingRect(contours[i]) 
        aspectRatio = float(w/h)
        if aspectRatio >= 1.5 and w > 0.33*widthImage and h < 0.6*heightImage and aspectRatio <= 6:          
            approx = cv2.approxPolyDP(contours[i], 0.05* cv2.arcLength(contours[i], True), True)
            if len(approx) == 4: 
                area = cv2.contourArea(contours[i])
                if area >= 10000 and area <= 120000:
                    image = cv2.drawContours(image.copy(),[contours[i]],0,(255,255,0),2)
                    mask = cv2.drawContours(mask,[contours[i]],0,(255,255,0),2)
                    out[mask == 255] = image[mask == 255]

                    (y, x, _) = np.where(mask == 255)
                    (topy, topx) = (np.min(y), np.min(x))
                    (bottomy, bottomx) = (np.max(y), np.max(x))
                    out2 = out[topy:bottomy, topx:bottomx]

                    imageShow = out2.copy()
                    out2 = cv2.cvtColor(out2,cv2.COLOR_BGR2GRAY)

                    out2 = np.float32(out2)
                    dst = cv2.cornerHarris(out2,20,21,0.04)

                    dst = cv2.dilate(dst,None)
                    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
                    dst = np.uint8(dst)

                    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                    corners = cv2.cornerSubPix(out2,np.float32(centroids),(5,5),(-1,-1),criteria)

                    corners = np.intp(corners)
                    imageShow[corners[:,1],corners[:,0]] = [0,255,0]

                    widthImageShow = int(imageShow.shape[1])
                    heightImageShow = int(imageShow.shape[0])

                    rmsd = 0
                    rmsdDict = {}
                    pointsDict = {}
                    rmsdStaticSoints = np.array([[0,0], [widthImageShow,0], [0, heightImageShow], [widthImageShow, heightImageShow]])

                    for point in range(4):
                        modelPoint = rmsdStaticSoints[point]
                        for i in range(len(corners)):
                            rmsdCorner = (0-np.sqrt((corners[i, 0] - modelPoint[0])**2 + (corners[i, 1] - modelPoint[1])**2))**2
                            rmsdDict[tuple(corners[i])] = np.sqrt(rmsdCorner/1)

                        minRmsd = min(rmsdDict, key=rmsdDict.get)
                        pointsDict[point] = minRmsd

                    points = np.array([pointsDict[0], pointsDict[1], pointsDict[2], pointsDict[3]])
                    # pts1 = np.float32([points[0], points[1], points[2], points[3]])
                    pts1 = np.float32([np.subtract(points[0], [0, 0]), np.subtract(points[1], [0, 0]), np.subtract(points[2], [0, 0]), np.subtract(points[3], [0, 0])])
                    pts2 = np.float32([[0,0],[widthImageShow,0],[0,heightImageShow],[widthImageShow,heightImageShow]])
                    M = cv2.getPerspectiveTransform(pts1,pts2)
                    dst = cv2.warpPerspective(image[topy:bottomy+1, topx:bottomx+1],M,(widthImageShow,heightImageShow))

                    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
                    ret3,th3 = cv2.threshold(dst,100,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

                    contoursPlate, hierarchyPlate = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                    boundingBoxes = [cv2.boundingRect(c) for c in contoursPlate]

                    if len(contoursPlate) != 0:
                                (contoursPlate, boundingBoxes) = zip(*sorted(zip(contoursPlate, boundingBoxes),
                                                                    key=lambda b:b[1][0], reverse=False))

                    finalLetters = []
                    th3 = cv2.cvtColor(th3, cv2.COLOR_GRAY2BGR)

                    letterIterator = 0

                    for i in range(len(contoursPlate)):
                        xPlate ,yPlate, wPlate, hPlate = cv2.boundingRect(contoursPlate[i])

                        if hPlate > dst.shape[0]/2 and wPlate < dst.shape[1]/4:
                            letter = dst[yPlate:yPlate+hPlate, xPlate:xPlate+wPlate].copy()
                            # letter= cv2.copyMakeBorder(letter,5,5,5,5,cv2.BORDER_CONSTANT,value=black_padding)

                            letterCopy = cv2.resize(letter, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_AREA)
                            letterCopy = np.array(letterCopy, dtype=np.float32) / 255.0

                            # # Tensorflow normal model
                            # prediction = lettersRecognitionModel.predict(letterCopy[None,:,:], batch_size=1, verbose = 0)
                            # yClasses = prediction.argmax(axis=-1)

                            # Tensorflow lite model
                            letterCopy = np.reshape(letterCopy, inputShape)
                            prediction = lettersRecognitionModel.set_tensor(inputDetails[0]['index'], letterCopy)
                            lettersRecognitionModel.invoke()
                            outputData = lettersRecognitionModel.get_tensor(outputDetails[0]['index'])
                            yClasses = np.argmax(outputData)

                            letterRecognition = letterLabels[yClasses]

                            found_wrong_letter = True

                            if letterIterator == 0 or letterIterator == 1:
                                while found_wrong_letter:
                                    found_wrong_letter = False
                                    if letterRecognition in "0":
                                        letterRecognition = "O"
                                        found_wrong_letter = True
                                    if letterRecognition in "123456789":
                                        outputData = np.delete(outputData, yClasses)
                                        yClasses = np.argmax(outputData)
                                        letterRecognition = letterLabels[yClasses]
                                        found_wrong_letter = True

                            found_wrong_letter = True

                            if letterIterator >= 3:
                                while found_wrong_letter:
                                    found_wrong_letter = False
                                    if letterRecognition in "O":
                                        letterRecognition = "0"
                                        found_wrong_letter = True
                                    if letterRecognition in "BDIZ":
                                        outputData = np.delete(outputData, yClasses)
                                        yClasses = np.argmax(outputData)
                                        letterRecognition = letterLabels[yClasses]
                                        found_wrong_letter = True

                            # print(outputData[0, yClasses])

                            finalLetters.append(''.join(letterRecognition))

                            # cv2.putText(letter, str(letterLabels[yClasses]), (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
                            # cv2.imshow("image crop letter: " + str(i), letter)

                            letterIterator += 1

                    # cv2.imshow("image perspective transform", imageShow)
                    # cv2.imshow("image crop", dst)
                    finalLetters = ''.join(map(str, finalLetters))
                    # print("finalLetters: ", finalLetters)
    # cv2.imshow("image final", image)

                    if len(finalLetters) >= 7:

                        return finalLetters
    if len(finalLetters) == 0:

        return "PO77777"


# # cv2.imshow("image final", image)

    # cv2.waitKey(0)
        
    # # closing all open windows
    # cv2.destroyAllWindows()