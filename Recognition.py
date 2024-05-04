import cv2
import numpy as np
import face_recognition as fr
import os
from datetime import datetime

path=r'D:\work\git\Face-Recognition-Attendance\Image Gallery'

image=[]
classnames=[]
mylist=os.listdir(path)
print(mylist)
for cls in mylist:
    curImg = cv2.imread(os.path.join(path, cls))
    image.append(curImg)
    classnames.append(os.path.splitext(cls)[0])


def findEncode(image):
    elist=[]
    for img in image:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=fr.face_encodings(img)[0]
        elist.append(encode)
    return elist

def markattendance(name):
    file_path = r'D:\work\git\Face-Recognition-Attendance\attendance.csv'
    with open(file_path,'r+') as f:
        myData=f.readlines()
        nameList=[]
        for line in myData:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtstring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')

known=findEncode(image)
print("encode complete")

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()

    imgs=cv2.resize(frame,(0,0),None,0.25,0.25)
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

    currfdis=fr.face_locations(imgs)
    currfencode=fr.face_encodings(imgs,currfdis)

    for encodeface,floc in zip(currfencode,currfdis):
        match=fr.compare_faces(known,encodeface)
        facedis=fr.face_distance(known,encodeface)
        matchIndex=np.argmin(facedis)

        if match[matchIndex]:
            name=classnames[matchIndex].upper()

            y1,x2,y2,x1=floc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(255,0,0),cv2.FILLED)
            cv2.putText(frame,name,(x1+6,y2-6),0,1,(255,255,255),2)
            markattendance(name)



    cv2.imshow('cam',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()

cv2.destroyAllWindows()



