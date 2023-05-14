import cv2
import os
import numpy as np
import glob

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

    for image_name in sorted(glob.glob(path + "*.jpg")):

        cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)

        cv2.createTrackbar('gw', 'image', 2, 10, empty_callback)
        cv2.createTrackbar('gs', 'image', 0, 10, empty_callback)
        cv2.createTrackbar('gw1', 'image', 2, 10, empty_callback)
        cv2.createTrackbar('gs1', 'image', 0, 10, empty_callback)
        cv2.createTrackbar('gw2', 'image', 2, 10, empty_callback)
        cv2.createTrackbar('gs2', 'image', 0, 10, empty_callback)

        gw, gs, gw1, gs1, gw2, gs2 = (3, 1, 7, 2, 3, 1)

        while True:
            dim = (800, 600)

            image = cv2.imread(image_name)

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

            # print(height)
            # print(width)

            # image_copy = cv2.medianBlur(image_copy, 5)
            # image_copy = cv2.GaussianBlur(image_copy, (5, 5), 0)

            img_blur = cv2.GaussianBlur(image_copy, (gw, gw), gs)
            g1 = cv2.GaussianBlur(img_blur, (gw1, gw1), gs1)
            g2 = cv2.GaussianBlur(img_blur, (gw2, gw2), gs2)
            ret, thg = cv2.threshold(g2-g1, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            cv2.imshow("image", thg)

            # cv2.waitKey(0)
                
            # # closing all open windows
            # cv2.destroyAllWindows()


            # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # ret, image_copy = cv2.threshold(image_copy, 150, 255, cv2.THRESH_BINARY)

            # kernel = np.ones((2, 2), np.uint8)
            # image = cv2.dilate(image, kernel, iterations=1)

            # edged = cv2.Canny(image_copy, 50, 150)

            # binary = cv2.bitwise_not(image_copy)

            # contours, hierarchy = cv2.findContours(edged, 
            # cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            contours, hierarchy = cv2.findContours(thg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # image = cv2.drawContours(image, contours, -1, (0,255,255), 3)

            # print(len(contours))
            res = 0

            rects = [] 
            end_x=0 
            end_y=0

            for i in range(len(contours)):

                if hierarchy[0][i][2] == -1:
                    continue


                x ,y, w, h = cv2.boundingRect(contours[i])
                a=w*h    
                aspectRatio = float(w)/h
                if aspectRatio >= 1.5 and w> 0.33*width_image:          
                    approx = cv2.approxPolyDP(contours[i], 0.05* cv2.arcLength(contours[i], True), True)
                    if len(approx) == 4:
                        end_x=x+w
                        end_y=y+h   
                        rect = cv2.minAreaRect(contours[i])
                        box = cv2.boxPoints(rect)
                        box = np.intp(box)
                        image = cv2.drawContours(image,[box],0,(0,255,255),2)
                        # image = cv2.drawContours(image,[contours[i]],0,(255,255,0),2)
                        # mask = cv2.drawContours(mask,[contours[i]],0,(255,255,0),2)
                        mask = cv2.drawContours(mask,[box],0,(255,255,255),2)
                        out[mask == 255] = image[mask == 255]
                        cv2.putText(image, "rectangle "+str(x)+" , " +str(y-5), (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                        # print("contours[i]: ", contours[i])

                        # Now crop
                        (y, x, z) = np.where(mask == 255)
                        (topy, topx) = (np.min(y), np.min(x))
                        (bottomy, bottomx) = (np.max(y), np.max(x))
                        out2 = out[topy:bottomy+1, topx:bottomx+1]
                        image_show = out2.copy()
                        out2 = cv2.cvtColor(out2,cv2.COLOR_BGR2GRAY)

                        out2 = np.float32(out2)
                        dst = cv2.cornerHarris(out2,2,3,0.04)
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
                        res = np.hstack((centroids,corners))
                        res = np.intp(res)
                        print("res: ", res[:,3])
                        image_show[res[:,1],res[:,0]]=[0,0,255]
                        image_show[res[:,3],res[:,2]] = [0,255,0]

                        # image_show[dst>0.01*dst.max()]=[0,0,255]

                        cv2.imshow("image crop", image_show)

            cv2.imshow("image final", image)


            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
                # polygon = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

                # if len(polygon) == 4: 
                #     rects.append(polygon)
                # (x,y,w,h) = cv2.boundingRect(contour)
                # cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
                # x1,y1 = contour[0][0]
                # approx = cv2.approxPolyDP(contour, 0.1*cv2.arcLength(contour, True), True)
                # # if len(approx) == 4:
                # x, y, w, h = cv2.boundingRect(contour)
                # ratio = float(w)/h
                #     # if ratio >= 2.0 and ratio <= 5.0:
                # if True:
                #     rect = cv2.minAreaRect(contour)
                #     box = cv2.boxPoints(rect)
                #     box = np.int0(box)
                #     image = cv2.drawContours(image, [box], -1, (0,255,255), 2, cv2.LINE_AA)
                    # cv2.putText(image, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    # else:
                    #     cv2.putText(image, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    #     image = cv2.drawContours(image, [contour], -1, (0,255,0), 3)

            # for rect in rects: 
            #     cv2.drawContours(image, [rect], 0, (0, 255, 0), 2) 
            # contours = sorted(contours, key = cv2.contourArea, reverse = True) [:30]

            # cv2.drawContours(image, contours, -1, (0,0,255), 3)
            # print(contours)

            # cv2.imshow("image", image)
            # cv2.waitKey(0)
            
        # closing all open windows
        cv2.destroyAllWindows()