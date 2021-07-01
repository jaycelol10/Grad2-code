from __future__ import print_function
import tkinter as tk
from timeit import Timer
from numpy.lib.utils import deprecate
import pretty_errors
from imutils.convenience import is_cv3
from imutils.object_detection import non_max_suppression
from imutils.video import FileVideoStream, filevideostream
from imutils.video import FPS
from imutils import paths
import numpy as np
import argparse
import imutils
import time
import cv2
from numpy.core.fromnumeric import size
import matplotlib.pyplot as plt
from random import randint
import os
import compare
from imutils.video import VideoStream
from imutils.video import FPS 
from shapely.geometry import Polygon
from tracker import *
def project():
    # Create tracker object
    tracker = EuclideanDistTracker()

    def count_frames(path, override=False):
        # grab a pointer to the video file and initialize the total
        # number of frames read
        video = cv2.VideoCapture(path)
        total = 0
        # if the override flag is passed in, revert to the manual
        # method of counting frames
        if override:
            total = count_frames_manual(video)
        # otherwise, let's try the fast way first
        else:
            # lets try to determine the number of frames in a video
            # via video properties; this method can be very buggy
            # and might throw an error based on your OpenCV version
            # or may fail entirely based on your which video codecs
            # you have installed
            try:
                # check if we are using OpenCV 3
                if is_cv3():
                    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                # otherwise, we are using OpenCV 2.4
                else:
                    total = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            # uh-oh, we got an error -- revert to counting manually
            except:
                total = count_frames_manual(video)
        # release the video file pointer
        video.release()
        # return the total number of frames in the video
        return total
    def count_frames_manual(video):
        # initialize the total number of frames read
        total = 0
        # loop over the frames of the video
        while True:
            # grab the current frame
            (grabbed, frame) = video.read()
        
            # check to see if we have reached the end of the
            # video
            if not grabbed:
                break
            # increment the total number of frames read
            total += 1
        # return the total number of frames in the video file
        return total

    class Car:
        carlist=[]
        def __init__(self,i,xi,yi):
            self.i=i
            self.x=xi
            self.y=yi
            self.carlist=[]
            self.R=randint(0,255)
            self.G=randint(0,255)
            self.B=randint(0,255)
    

        def getRGB(self):  #For the RGB colour
            return (self.R,self.G,self.B)

        def red(self):  #Red
            self.R=randint(0,255)

        def getcars(self):
            return self.carlist

        def getId(self): #For the ID
            return self.i

        def getX(self):  #for x coordinate
            return self.x

        def getY(self):  #for y coordinate
            return self.y

    class ped:
        pedlist=[]
        currstat = 0
        def __init__(self,i,xi,yi):
            self.i=i
            self.x=xi
            self.y=yi
            self.pedlist=[]
            self.R=randint(0,255)
            self.G=randint(0,255)
            self.B=randint(0,255)
    

        def getRGB(self):  #For the RGB colour
            return (self.R,self.G,self.B)

        def red(self):  #Red
            self.R=randint(0,255)

        def getcars(self):
            return self.pedlist

        def getId(self): #For the ID
            return self.i

        def getX(self):  #for x coordinate
            return self.x

        def getY(self):  #for y coordinate
            return self.y



    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,
        help="path to input video file") 
    ap.add_argument("-o", "--override", type=int, default=-1,
        help="whether to force manual frame count") 
    args = vars(ap.parse_args())

    # count the total number of frames in the video file
    override = False if args["override"] < 0 else True
    total = count_frames(args["video"], override=override)

    # display the frame count to the terminal
    print("[INFO] {:,} total frames read from {}".format(total,
        args["video"][args["video"].rfind(os.path.sep) + 1:]))

    def pre_proc1(frame):
        frame = imutils.resize(frame,320)           
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret,th=cv2.threshold(frame_gray, 40,255,0)
        return(frame_gray,frame,ret,th)

    
    def get_percentage(newbox,bbox):
        ia =max (0, min(bbox[0] + bbox[3],newbox[0] + newbox[3]) - max(bbox[0],newbox[0])) * max(0,min(bbox[1]+bbox[2] , newbox[1]+newbox[2] ) - max(bbox[1], newbox[1])) 
        bbox_area = bbox[2] * bbox[3]
        newbox_area = newbox[2] * newbox[3]
        union = (bbox_area + newbox_area) - ia
        percent_coverage = (ia / (union) ) * 100
        return (percent_coverage)

    def find_cont(th,frame_gray):
        centers=[] 
        contours, hierarchy = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c)<100:
                continue
            
            elif cv2.contourArea(c)<1000:
                continue
            
            cv2.drawContours(frame_gray, [c], -1, (0,255,0), 3)

            # Find center point of contours:
            M = cv2.moments(c)
            cX = int(M['m10'] /M['m00'])
            cY = int(M['m01'] /M['m00'])
            centers.append([cX,cY])

            # Find the distance D between the two contours:
            if len(centers) >=2:
                dx= centers[0][0] - centers[1][0]
                dy = centers[0][1] - centers[1][1]
                D = np.sqrt(dx*dx+dy*dy)
                g = float("{0:.2f}".format(D))
                return (g)


        
    
                
    

        


    # start the file video stream thread and allow the buffer to
    # start to fill
    print("[INFO] starting video file thread...")
    fvs = FileVideoStream(args["video"]).start()
    time.sleep(1.0)
    # start the FPS timer
    fps = FPS().start()

    #our trained classifiers
    car_tracker_file = 'cars.xml'
    pedstrain ='haarcascade_fullbody.xml'
    pedstrain_upperbody ='haarcascade_upperbody.xml'


    #car and pedstrain  classifier 
    car_tracker= cv2.CascadeClassifier(car_tracker_file)
    pedstrain_tracker= cv2.CascadeClassifier(pedstrain)


    id = 1
    count = 0 
    countped = 0
    p = Car
    pp = ped
    total = 1
    ii = 0
    i = 0
    # Object detection from Stable camera
    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=60)
    while fvs.more():
        start_time = time.time()
        
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale 
        frame = fvs.read()

        
        total += 1
        frame_gray,frame,ret,th = pre_proc1(frame)
        
        
        g = find_cont(th,frame_gray)      

        # newFrame = np.array(threashold)
        
        pedestrian = pedstrain_tracker.detectMultiScale(frame_gray, 1.3, 2)
        
        
        cars = car_tracker.detectMultiScale(frame_gray)
        
        
    

        carinnerllist = []
        idd = 0
    #
    #   for (x,y,w,h) in cars:
    #     carinnerllist.clear()
    #     carinnerllist.append(x)
    #     carinnerllist.append(y)
    #     carinnerllist.append(w)
    #     carinnerllist.append(h)
            #carinnerllist.append(id)
    #     p.carlist.append(carinnerllist)
            
    #      #cv2.rectangle(frame,(p.carlist[count][0],p.carlist[count][1]),(p.carlist[count][0]+p.carlist[count][2],p.carlist[count][count]+p.carlist[count][3]),(0,255,0),2)
            #cv2.putText(frame, f'Car {p.carlist[idd][4]}', (p.carlist[count][0],p.carlist[count][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    #      count += 1
    #      idd += 1
    #     id +=1
    #     
        
            
        #function call 
        #person = 1
    # iddd=0
        
        pedframes = 0
        pedinnerlist = []
    
        for (x, y, w, h) in pedestrian: 
            
            pedinnerlist.clear()
            pedinnerlist.append(x)
            pedinnerlist.append(y)
            pedinnerlist.append(w)
            pedinnerlist.append(h)
            #pedinnerlist.append(idd)
            #pedinnerlist.append(iddd)
            pedframes +=1
            #pedinnerlist.append(pedframes)
            
            pp.pedlist.append(pedinnerlist)  
            # 1. Object Detection
            detections = pp.pedlist
        
        # if pp.pedlist[countped ][0] != x :
            #   pedframes = 0
            #cv2.rectangle(frame,(pp.pedlist[countped][0],pp.pedlist[countped][1]),(pp.pedlist[countped][0]+pp.pedlist[countped][2],pp.pedlist[countped][1]+pp.pedlist[countped][3]),(0,0,255),2)
            
            #cv2.putText(frame, f'Person {pp.pedlist[iddd][4]}', (pp.pedlist[countped][0],pp.pedlist[countped][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        
        # print ( "Estimated  time for ",pp.pedlist[iddd][4],"->>>", pedframes/(1.0 / (time.time() - start_time))) 
            
        # person +=1
            #countped +=1
        # iddd+=1
            roi =frame
            boxes_ids = tracker.update(detections)
            x, y, w, h, id = boxes_ids[i]
            
            cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3) 
            i+=1  
                
        
    

            
            

                
        


    
            

        # display the size of the queue on the frame
        cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)	
            
        #show number of detected persons
    # cv2.putText(frame, f'Total Persons : {person - 1}', (10,30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,0,0), 2)
        cv2.putText(frame, f'Distance : {g}', (10,70), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,0,0), 2)
        cv2.putText(frame, f'Total cars : {idd }', (10,90), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,0,0), 2)
    
        
        #cv2.putText(frame_gray, f'el asped time : {0:.2f}'.format(fps.elapsed()), (10,90), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,0,0), 2)
        #cv2.putText(frame_gray, f'FPS : {0:.2f}'.format(fps.fps()), (10,110), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,0,0), 2)
    

        # show the frame and update the FPS counter
        cv2.imshow("Frame",frame)
        fps.update()
        

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1) #wait until any key is pressed


    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps())) 
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
    fvs.stop()	

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
    window = tk.Tk()
    window.title("DetectLex")
    window.geometry('500x500')
    window.configure(bg='#856ff8')
    lbl = tk.Label(text='Press q to exit the stream',font=('Arial Bold',15),bg='#856ff8', fg='#FFFFFF')
    lbl2 = tk.Label(text='Press p to pause the stream',font=('Arial Bold',15),bg='#856ff8', fg='#FFFFFF')
    Start_btn = tk.Button(text='Start', command=project,font=('Arial Bold',30))

    lbl.place(x=120, y=120)
    lbl2.place(x=120, y=150)
    Start_btn.place(x=200, y=200)
    window.mainloop()