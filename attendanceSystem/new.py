import tkinter as tk
from tkinter import Message , Text
import cv2
import os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

window =tk.Tk()
window.title("face recognies")
window.geometry("1280x720")
dialog_title = "QUIT"
dialog_text ="Are you Sure"
window.configure(background="green")
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

message = tk.Label(window, text = "Attendence Management System By Face Recognition" , bg="blue", fg="white" ,width=50, height=3, font=('times',30,'italic bold underline'))
message.place(x=75,y=17)
lb1=tk.Label(window,text="Enrollment no", width=20,height=1,fg="black",bg="white", font=('times',15,'bold'))
lb1.place(x=150,y=200)
txt=tk.Entry(window,width=20,fg="black",bg="white", font=('times',15,'bold'))
txt.place(x=450,y=200)

lbl2=tk.Label(window,text="Enter Name", width=20,height=1,fg="black",bg="white", font=('times',15,'bold'))
lbl2.place(x=150,y=260)
txt2=tk.Entry(window,width=20,fg="black",bg="white", font=('times',15,'bold'))
txt2.place(x=450,y=260)

lbl3=tk.Label(window,text="Notification", width=20,height=1,fg="black",bg="white", font=('times',15,'bold'))
lbl3.place(x=150,y=320)
message=tk.Label(window,text="",bg="white",fg="black",width=30,height=2, activebackground="white",font=('times',15,'bold'))
message.place(x=450,y=320)


lbl3=tk.Label(window,text="Attendence", width=20,height=1,fg="black",bg="white", font=('times',15,'bold underline'))
lbl3.place(x=150,y=600)
message2=tk.Label(window,text="",bg="white",fg="black",width=20,height=1, activebackground="yellow",font=('times',15,'bold'))
message2.place(x=450,y=600)

def clear():
    txt.delete(0, 'end')
    res =""
    message.configure(text=res)
def clear2():
    txt2.delete(0, 'end')
    res =""
    message.configure(text=res)
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def TakeImage():
    Id=(txt.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam=cv2.VideoCapture(0)
        harcascadePath="haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(cv2.data.haarcascades + harcascadePath)
        sampleNum=0
        while(True):
            ret,img=cam.read()
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
            faces= detector.detectMultiScale(gray,1.3,5)
            for(x,y,w,h)in faces:
                cv2.rectangle(img,(x,y),(x+y,y+h),(255,0,0),2)
                sampleNum=sampleNum+1
                cv2.imwrite("TrainingImages\ "+name +"." + Id + '.' + str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('fame',img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                            break
            elif sampleNum >60:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Image Saved for ID: "+ Id + "Name : "+ name
        row = [Id,name]
        with open('StudentDetails\studentDetails.csv', 'a+') as csvFile:
            writer=csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text=res)
    else:
        if(is_number(Id)):
            res=" Enter Alphabetical Name"
            message.configure(text=res)
        if(name.isalpha()):
            res=" Enter numeric Id"
            message.configure(text=res)
                            


def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade frontalface default.xml"
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + harcascadePath)
    cv2.data.haarcascades
    faces, Id = getImagesAndLabels("TrainingImages")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res ="Image Trained"#+", ".join(str(f) for f in Id)
    message.configure(text=res)

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 

    faces=[]
    Ids=[]

    for imagePath in imagePaths:
        pilImage= Image.open(imagePath).convert('L')
        imageNp = np.array (pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)

    return faces,Ids
def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel"+os.sep+"Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+harcascadePath)
    df = pd.read_csv("StudentDetails"+os.sep+"StudentDetails.csv")
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
 
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height
    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
 
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5,minSize = (int(minW), int(minH)),flags = cv2.CASCADE_SCALE_IMAGE)
        for(x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x+w, y+h), (10, 159, 255), 2)
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
 
            if conf < 100:
 
                aa = df.loc[df['Id'] == Id]['Name'].values
                confstr = "  {0}%".format(round(100 - conf))
                tt = str(Id)+"-"+aa
 
 
            else:
                Id = '  Unknown  '
                tt = str(Id)
                confstr = "  {0}%".format(round(100 - conf))
 
            if (100-conf) > 67:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = str(aa)[2:-2]
                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
 
            tt = str(tt)[2:-2]
            if(100-conf) > 67:
                tt = tt + " [Pass]"
                cv2.putText(im, str(tt), (x+5,y-5), font, 1, (255, 255, 255), 2)
            else:
                cv2.putText(im, str(tt), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
 
            if (100-conf) > 67:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font,1, (0, 255, 0),1 )
            elif (100-conf) > 50:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 255), 1)
            else:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 0, 255), 1)
 
 
 
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('Attendance', im)
        if (cv2.waitKey(1) == ord('q')):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = "Attendance"+os.sep+"Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName, index=False)
    cam.release()
    cv2.destroyAllWindows()
    res =attendance
    message2.configure(text =res)

        
clearButton =tk.Button(window,text="clear",command=clear,fg="black",bg="white",width=20,height=1,activebackground="red",font=('times',15,'bold'))
clearButton.place(x=800,y=200)

clearButton2 =tk.Button(window,text="clear",command=clear2,fg="black",bg="white",width=20,height=1,activebackground="red",font=('times',15,'bold'))
clearButton2.place(x=800,y=260)

takeImg=tk.Button(window, text="Take Image",command=TakeImage ,fg="black",bg="white",width=20,height=3,activebackground="red",font=('times',15,'bold'))
takeImg.place(x=80,y=440)
trainImg=tk.Button(window, text="Train Image",command=TrainImages,fg="black",bg="white",width=20,height=3,activebackground="red",font=('times',15,'bold'))
trainImg.place(x=380,y=440)
trackImg=tk.Button(window, text="Track Image",command=TrackImages ,fg="black",bg="white",width=20,height=3,activebackground="red",font=('times',15,'bold'))
trackImg.place(x=680,y=440)

quitWindow=tk.Button(window, text="Quit",command=window.destroy,fg="black",bg="white",width=20,height=3,activebackground="red",font=('times',15,'bold'))
quitWindow.place(x=980,y=440)

window.mainloop()

