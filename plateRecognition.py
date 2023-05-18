import cv2
import os
import numpy as np
import glob

import tensorflow as tf
from tensorflow import keras

# print(tf.version.VERSION)

letters_recognition_model = tf.keras.models.load_model('content/saved_model/my_model')

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

    gw, gs, gw1, gs1, gw2, gs2 = (3, 1, 3, 2, 5, 1)

    letterLabels = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

    for image_name in sorted(glob.glob(path + "*.jpg")):

        cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)

        cv2.createTrackbar('gw', 'image', 2, 10, empty_callback)
        cv2.createTrackbar('gs', 'image', 0, 10, empty_callback)
        cv2.createTrackbar('gw1', 'image', 2, 10, empty_callback)
        cv2.createTrackbar('gs1', 'image', 0, 10, empty_callback)
        cv2.createTrackbar('gw2', 'image', 2, 10, empty_callback)
        cv2.createTrackbar('gs2', 'image', 0, 10, empty_callback)

        # setting position of 'G' trackbar to 100
        cv2.setTrackbarPos('gw', 'image', gw)
        cv2.setTrackbarPos('gs', 'image', gs)
        cv2.setTrackbarPos('gw1', 'image', gw1)
        cv2.setTrackbarPos('gs1', 'image', gs1)
        cv2.setTrackbarPos('gw2', 'image', gw2)
        cv2.setTrackbarPos('gs2', 'image', gs2)

        while True:
            image = cv2.imread(image_name)

            image_ratio = image.shape[1]/image.shape[0]

            dim = (800, int(800 / image_ratio))

            print("image_ratio: ", image_ratio)

            gw = on_blockSize_trackbar(cv2.getTrackbarPos('gw', 'image'), 'gw')
            gs = cv2.getTrackbarPos('gs', 'image')
            gw1 = on_blockSize_trackbar(cv2.getTrackbarPos('gw1', 'image'), 'gw1')
            gs1 = cv2.getTrackbarPos('gs1', 'image')
            gw2 = on_blockSize_trackbar(cv2.getTrackbarPos('gw2', 'image'), 'gw2')
            gs2 = cv2.getTrackbarPos('gs2', 'image')

            image_copy = image.copy()

            image = cv2.resize(image, (dim), interpolation = cv2.INTER_AREA)

            mask = np.zeros_like(image) # Create mask where white is what we want, black otherwise
            out = np.zeros_like(image) # Extract out the object and place into output image

            image_copy = cv2.resize(image_copy, (dim), interpolation = cv2.INTER_AREA)
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

            width_image = int(image_copy.shape[1])
            height_image = int(image_copy.shape[0])

            img_blur = cv2.GaussianBlur(image_copy, (gw, gw), gs)
            g1 = cv2.GaussianBlur(img_blur, (gw1, gw1), gs1)
            g2 = cv2.GaussianBlur(img_blur, (gw2, gw2), gs2)
            ret, thg = cv2.threshold(g2-g1, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            cv2.imshow("image", thg)

            contours, hierarchy = cv2.findContours(thg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            res = 0

            rects = [] 
            end_x=0 
            end_y=0

            print("len(contours): ", len(contours))


            for i in range(len(contours)):

                if hierarchy[0][i][2] == -1:
                    continue


                x ,y, w, h = cv2.boundingRect(contours[i]) 
                aspectRatio = float(w)/h
                if aspectRatio >= 1.5 and w> 0.33*width_image and aspectRatio <= 5:          
                    approx = cv2.approxPolyDP(contours[i], 0.05* cv2.arcLength(contours[i], True), True)
                    if len(approx) == 4:
                        end_x=x+w
                        end_y=y+h   
                        rect = cv2.minAreaRect(contours[i])
                        box = cv2.boxPoints(rect)
                        box = np.intp(box)

                        area = cv2.contourArea(contours[i])

                        if area >= 15000 and area <=60000:
                            # image = cv2.drawContours(image,[box],0,(0,255,255),1)
                            image = cv2.drawContours(image,[contours[i]],0,(255,255,0),2)
                            mask = cv2.drawContours(mask,[contours[i]],0,(255,255,0),2)
                            # mask = cv2.drawContours(mask,[box],0,(255,255,255),1)
                            out[mask == 255] = image[mask == 255]
                            cv2.putText(image, "rectangle "+str(x)+" , " +str(y-5), (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                            # print("contours[i]: ", contours[i])

                            # Now crop
                            (y, x, _) = np.where(mask == 255)
                            (topy, topx) = (np.min(y), np.min(x))
                            (bottomy, bottomx) = (np.max(y), np.max(x))
                            out2 = out[topy:bottomy, topx:bottomx]

                            image_show = out2.copy()
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
                            # Threshold for an optimal value, it may vary depending on the image.

                            # Now draw them
                            corners = np.intp(corners)
                            image_show[corners[:,1],corners[:,0]] = [0,255,0]
                            # print("corners: ", corners[:,:])

                            width_image_show = int(image_show.shape[1])
                            height_image_show = int(image_show.shape[0])

                            rmsd = 0
                            rmsd_dict = {}
                            points_dict = {}
                            rmsd_static_points = np.array([[0,0], [width_image_show,0], [0, height_image_show], [width_image_show, height_image_show]])

                            for point in range(4):
                                model_point = rmsd_static_points[point]
                                for i in range(len(corners)):
                                    rmsd_corner = (0-np.sqrt((corners[i, 0] - model_point[0])**2 + (corners[i, 1] - model_point[1])**2))**2
                                    rmsd_dict[tuple(corners[i])] = np.sqrt(rmsd_corner/1)

                                min_rmsd = min(rmsd_dict, key=rmsd_dict.get)
                                points_dict[point] = min_rmsd
                            print("points_dict: ", points_dict)

                            points = np.array([points_dict[0], points_dict[1], points_dict[2], points_dict[3]])
                            pts1 = np.float32([points[0], points[1], points[2], points[3]])
                            # print("pts1: ", pts1)
                            pts2 = np.float32([[0,0],[width_image_show,0],[0,height_image_show],[width_image_show,height_image_show]])
                            M = cv2.getPerspectiveTransform(pts1,pts2)
                            dst = cv2.warpPerspective(image[topy:bottomy+1, topx:bottomx+1],M,(width_image_show,height_image_show))

                            dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
                            ret3,th3 = cv2.threshold(dst,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

                            black_padding = [0,0,0]
                            # th3= cv2.copyMakeBorder(th3,10,10,10,10,cv2.BORDER_CONSTANT,value=black_padding)

                            contours_plate, hierarchy_plate = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                            boundingBoxes = [cv2.boundingRect(c) for c in contours_plate]

                            if len(contours_plate) != 0:
                                (contours_plate, boundingBoxes) = zip(*sorted(zip(contours_plate, boundingBoxes),
                                                                    key=lambda b:b[1][0], reverse=False))

                            finalLetters = []
                            th3 = cv2.cvtColor(th3, cv2.COLOR_GRAY2BGR)

                            for i in range(len(contours_plate)):
                                # if hierarchy_plate[0][i][2] == -1:
                                #     continue
                                x_plate ,y_plate, w_plate, h_plate = cv2.boundingRect(contours_plate[i])

                                if h_plate > dst.shape[0]/3:
                                # cv2.rectangle(th3, (x_plate, y_plate), (x_plate + w_plate_plate, y_plate + h_plate), (0,0,255), 2) 
                                # th3 = cv2.drawContours(th3, [contours_plate[i]],0,(0,255,0),1)
                                    letter = dst[y_plate:y_plate+h_plate, x_plate:x_plate+w_plate].copy()
                                    # letter= cv2.copyMakeBorder(letter,5,5,5,5,cv2.BORDER_CONSTANT,value=black_padding)

                                    letter_copy = cv2.resize(letter, (32, 40), interpolation = cv2.INTER_AREA)
                                    # letter_copy = cv2.cvtColor(letter_copy, cv2.COLOR_BGR2GRAY)
                                    letter_copy = np.array(letter_copy) / 255.0
                                    prediction = letters_recognition_model.predict(letter_copy[None,:,:], batch_size=1)
                                    y_classes = prediction.argmax(axis=-1)
                                    # print("prediction: ", letterLabels[y_classes])

                                    # print("letterDescription: ", letterDescription)
                                    finalLetters.append(''.join(letterLabels[y_classes]))
                                    cv2.putText(letter, str(letterLabels[y_classes]), (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
                                    cv2.imshow("image crop letter: " + str(i), letter)

                            cv2.imshow("image perspective transform", image_show)
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