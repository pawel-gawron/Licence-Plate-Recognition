import cv2
import os
import numpy as np
import glob

import tensorflow as tf
from tensorflow import keras

# print(tf.version.VERSION)

# # Tensorflow normal model
# lettersRecognitionModel = tf.keras.models.load_model('content/saved_model/my_model')

# Tensorflow lite model
lettersRecognitionModel = tf.lite.Interpreter(model_path='/home/pawel/Documents/RISA/sem1/SW/Licence-Plate-Recognition/model.tflite')
lettersRecognitionModel.allocate_tensors()

inputDetails = lettersRecognitionModel.get_input_details()
outputDetails = lettersRecognitionModel.get_output_details()
inputShape = inputDetails[0]['shape']

# print("input_details: ", inputDetails)

# Check its architecture
# letters_recognition_model.summary()
# np.testing.assert_allclose(letters_recognition_model(input_arr), outputs)

def empty_callback(value):
    pass

def on_blockSize_trackbar(val, name):
    global blockSize
    if (val%2)==0:
        blockSize = val+1
        cv2.setTrackbarPos(name, "image", blockSize)
    else:
        blockSize=val
        cv2.setTrackbarPos(name, "image", blockSize)
    blockSize = max(blockSize, 1)

    return blockSize

def detect():
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path + "/train/")

    for image in sorted(glob.glob(path + "*.jpg")):
        image = cv2.imread(image, -1)

        return image


if __name__ == '__main__':
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path + "/train/")

    gaussianWindow = 7
    gaussianDeviation = 37

    gaussianWindow1 = 7
    gaussianDeviation1 = 1

    gaussianWindow2 = 21
    gaussianDeviation2 = 11

    letterLabels = np.array(['0', '1', '2', '3', '4', '5', '6',
                             '7', '8', '9', 'A', 'B', 'C', 'D',
                             'E', 'F', 'G', 'H', 'I', 'J', 'K',
                             'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                             'S', 'T', 'U', 'V', 'W', 'X', 'Y',
                             'Z'])

    for imageName in sorted(glob.glob(path + "*.jpg")):

        cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)

        cv2.createTrackbar('gw', 'image', 2, 20, empty_callback)
        cv2.createTrackbar('gs', 'image', 0, 55, empty_callback)
        cv2.createTrackbar('gw1', 'image', 2, 50, empty_callback)
        cv2.createTrackbar('gs1', 'image', 0, 50, empty_callback)
        cv2.createTrackbar('gw2', 'image', 2, 50, empty_callback)
        cv2.createTrackbar('gs2', 'image', 0, 50, empty_callback)

        # setting position of 'G' trackbar to 100
        cv2.setTrackbarPos('gw', 'image', gaussianWindow)
        cv2.setTrackbarPos('gs', 'image', int(gaussianDeviation))
        cv2.setTrackbarPos('gw1', 'image', gaussianWindow1)
        cv2.setTrackbarPos('gs1', 'image', int(gaussianDeviation1))
        cv2.setTrackbarPos('gw2', 'image', gaussianWindow2)
        cv2.setTrackbarPos('gs2', 'image', int(gaussianDeviation2))

        while True:
            image = cv2.imread(imageName)

            imageRatio = image.shape[1]/image.shape[0]

            dim = (900, int(900 / imageRatio))
            dim = (960, 720)

            print("image_ratio: ", imageRatio)

            gaussianWindow = on_blockSize_trackbar(cv2.getTrackbarPos('gw', 'image'), 'gw')
            gaussianDeviation = cv2.getTrackbarPos('gs', 'image')
            gaussianWindow1 = on_blockSize_trackbar(cv2.getTrackbarPos('gw1', 'image'), 'gw1')
            gaussianDeviation1 = cv2.getTrackbarPos('gs1', 'image')
            gaussianWindow2 = on_blockSize_trackbar(cv2.getTrackbarPos('gw2', 'image'), 'gw2')
            gaussianDeviation2 = cv2.getTrackbarPos('gs2', 'image')

            gaussianDeviation /= 10
            gaussianDeviation1 /= 10
            gaussianDeviation2 /= 10 

            imageCopy = image.copy()

            image = cv2.resize(image, (dim), interpolation = cv2.INTER_AREA)

            mask = np.zeros_like(image) # Create mask where white is what we want, black otherwise
            out = np.zeros_like(image) # Extract out the object and place into output image

            imageCopy = cv2.resize(imageCopy, (dim), interpolation = cv2.INTER_AREA)
            imageCopy = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2GRAY)

            widthImage = int(imageCopy.shape[1])
            heightImage = int(imageCopy.shape[0])

            imgBlur = cv2.GaussianBlur(imageCopy, (gaussianWindow, gaussianWindow), gaussianDeviation)
            gaussianBlur1 = cv2.GaussianBlur(imgBlur, (gaussianWindow1, gaussianWindow1), gaussianDeviation1)
            gaussianBlur2 = cv2.GaussianBlur(imgBlur, (gaussianWindow2, gaussianWindow2), gaussianDeviation2)
            ret, thg = cv2.threshold(gaussianBlur2-gaussianBlur1, 160, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            gaussianDeviation *= 10
            gaussianDeviation1 *= 10
            gaussianDeviation2 *= 10

            cv2.imshow("image", thg)

            contours, hierarchy = cv2.findContours(thg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            print("len(contours): ", len(contours))

            for i in range(len(contours)):

                # if hierarchy[0][i][2] == -1:
                #     continue

                x ,y, w, h = cv2.boundingRect(contours[i]) 
                aspectRatio = float(w/h)
                if aspectRatio >= 1.5 and w > 0.33*widthImage and h < 0.6*heightImage and aspectRatio <= 6:          
                    approx = cv2.approxPolyDP(contours[i], 0.05* cv2.arcLength(contours[i], True), True)
                    if len(approx) == 4: 
                        area = cv2.contourArea(contours[i])
                        if area >= 10000 and area <= 120000:
                            image = cv2.drawContours(image,[contours[i]],0,(255,255,0),2)
                            mask = cv2.drawContours(mask,[contours[i]],0,(255,255,0),2)
                            out[mask == 255] = image[mask == 255]
                            # print("contours[i]: ", contours[i])

                            # Now crop
                            (y, x, _) = np.where(mask == 255)
                            (topy, topx) = (np.min(y), np.min(x))
                            (bottomy, bottomx) = (np.max(y), np.max(x))
                            out2 = out[topy:bottomy, topx:bottomx]

                            imageShow = out2.copy()
                            out2 = cv2.cvtColor(out2,cv2.COLOR_BGR2GRAY)

                            out2 = np.float32(out2)
                            dst = cv2.cornerHarris(out2,20,21,0.04)
                            #result is dilated for marking the corners, not important
                            dst = cv2.dilate(dst,None)
                            ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
                            dst = np.uint8(dst)

                            # find centroids
                            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
                            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                            corners = cv2.cornerSubPix(out2,np.float32(centroids),(5,5),(-1,-1),criteria)

                            # Now draw them
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
                            print("points_dict: ", pointsDict)

                            points = np.array([pointsDict[0], pointsDict[1], pointsDict[2], pointsDict[3]])
                            pts1 = np.float32([points[0], points[1], points[2], points[3]])
                            pts2 = np.float32([[0,0],[widthImageShow,0],[0,heightImageShow],[widthImageShow,heightImageShow]])
                            M = cv2.getPerspectiveTransform(pts1,pts2)
                            dst = cv2.warpPerspective(image[topy:bottomy+1, topx:bottomx+1],M,(widthImageShow,heightImageShow))

                            dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
                            ret3,th3 = cv2.threshold(dst,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

                            blackPadding = [0,0,0]
                            # th3= cv2.copyMakeBorder(th3,10,10,10,10,cv2.BORDER_CONSTANT,value=black_padding)

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

                                    letterCopy = cv2.resize(letter, (32, 40), interpolation = cv2.INTER_AREA)
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
                                            if letterRecognition in "1234567890":
                                                outputData = np.delete(outputData, yClasses)
                                                yClasses = np.argmax(outputData)
                                                letterRecognition = letterLabels[yClasses]
                                                found_wrong_letter = True

                                    found_wrong_letter = True

                                    if letterIterator >= 3:
                                        while found_wrong_letter:
                                            found_wrong_letter = False
                                            if letterRecognition in "BDIOZ":
                                                outputData = np.delete(outputData, yClasses)
                                                yClasses = np.argmax(outputData)
                                                letterRecognition = letterLabels[yClasses]
                                                found_wrong_letter = True

                                    finalLetters.append(''.join(letterRecognition))
                                    cv2.putText(letter, str(letterLabels[yClasses]), (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
                                    cv2.imshow("image crop letter: " + str(i), letter)

                                    letterIterator += 1

                            cv2.imshow("image perspective transform", imageShow)
                            cv2.imshow("image crop", dst)
                            finalLetters = ''.join(map(str, finalLetters))
                            print("finalLetters: ", finalLetters)
                        

            cv2.imshow("image final", image)


            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            
        # closing all open windows
        cv2.destroyAllWindows()