import pandas as pd
import cv2
import urllib.request
import numpy as np
import os
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
import joblib
import glob

################################### PATH #################################################
# ESP 32 cam url
url = 'http://192.168.31.227/cam-hi.jpg'
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
###########################################################################################


# Create directory if not present 
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if f'Attendance_sheet.xlsx' not in os.listdir('Attendance'):
    columns = ['Surname', 'Name', 'M/F', 'Class']
    # Create a DataFrame with the specified columns
    df = pd.DataFrame(columns=columns)
    # Save the DataFrame to an Excel file
    df.to_excel('Attendance/Attendance_sheet.xlsx', index=False)

attendance_sheet = r'Attendance/Attendance_sheet.xlsx'
nimgs = 10

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)
    

def add_attendance(name):
    # Read existing data from Excel sheet
        df = pd.read_excel(attendance_sheet)
        now = datetime.now()
        date_string = now.strftime('%Y-%m-%d 00:00:00')
        # Add the current date column if it doesn't exist
        if date_string not in df.columns:
            df[date_string] = ""

        # Find the student in the dataframe and mark the attendance
        for index, row in df.iloc[:].iterrows():
            student_name = f"{row['Name']} {row['Surname']}"
            if student_name.upper() == name.upper():
                df.at[index, date_string] = "Present"
                print(f"Attendance marked for {name} on {date_string}")
                break
        else:
            print(f"Student '{name}' not found in the attendance sheet.")
        df.to_excel(attendance_sheet, index=False)
         

def mark_attendance():

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        print('There is no trained model in the static folder. Please add a new face to continue.')

    else:
        while True:
            # Continuously capture frames from ESP32 camera
            img_resp = urllib.request.urlopen(url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(imgnp, -1)

            # Resize and convert the frame to RGB
            imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            facesCurFrame = extract_faces(imgS) 
            for (x, y, w, h) in facesCurFrame:
                face = imgS[y:y+h, x:x+w]
                face = cv2.resize(face, (50, 50))
                face_flatten = face.flatten().reshape(1, -1)
                identified_person = identify_face(face_flatten)[0]
                add_attendance(identified_person)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
                cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
                cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
            cv2.imshow('Attendance', frame)
            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()
        cv2.imread


def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []


def add_student():
    first_name = input("Enter first name of student: ").title()
    surname = input("Enter surname of student: ").title()
    std = input("Enter standard: ").upper()
    gender = input("Gender (Male/Female): ").title()

    df = pd.read_excel(attendance_sheet)
    for index, row in df.iloc[:].iterrows():
        if (row['Name'] == first_name) and (row['Surname'] == surname):
            print(f"Student is already in the list")
            break
    else:
        # cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
        print(f"Adding new student '{first_name} {surname}' into attendance sheet.")
        name_drive = 'static/faces/' + first_name + " " + surname
        if not os.path.isdir(name_drive):
            os.makedirs(name_drive)
        i, j = 0, 0
        df = pd.read_excel(attendance_sheet)
        new_student_data = ({'Surname': surname, 'Name': first_name, 'M/F': gender, 'Class': std})
        df = df._append(new_student_data, ignore_index=True)
        # Write the updated DataFrame back to the Excel sheet
        df.to_excel(attendance_sheet, index=False)

        print("To quite press 'esc'")
        while True:
            # Capture frame from ESP32 camera
            img_resp = urllib.request.urlopen(url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(imgnp, -1)

            # Extract faces from the frame
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 5 == 0:
                    name = first_name + "_" + surname + '_' + str(i) + '.jpg'
                    cv2.imwrite(name_drive + '/' + name, frame[y:y+h, x:x+w])
                    i += 1
                j += 1
            if j == nimgs * 5:
                break
            
            cv2.imshow('Adding new Student', frame)
            if cv2.waitKey(1) == 27:  # Press 'ESC' to break
                break
    cv2.destroyAllWindows()           
    train_model()

def update_student_profile():
    first_name = input("Enter first name of student: ").title()
    surname = input("Enter surname of student: ").title()
    df = pd.read_excel(attendance_sheet)                   
    for index, row in df.iloc[:].iterrows():
        if (row['Name'] == first_name) and (row['Surname'] == surname):
            print(f"Person is already in the list")
            print("##############################")
            print("1. Update class")
            print("2. Update photo")
            print("3. Exit")
            ch = int(input("Choose to update(1/2): "))
            if(ch == 1):
                Class = input("Enter class: ").upper()
                df.at[index, 'Class'] = Class
                df.to_excel(attendance_sheet, index=False)
                print(f"Class updated")
                break
            elif(ch == 2):
                cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
                name_drive = 'static/faces/' + first_name + " " + surname
                if not os.path.isdir(name_drive):
                    os.makedirs(name_drive)
                else:
                    files = glob.glob(name_drive + '/*')
                    for f in files:
                        os.remove(f)
                i, j = 0, 0
                print("Press 'Spacebar' to capture image or to quite press 'esc'")
                while True:
                    # Capture frame from ESP32 camera
                    img_resp = urllib.request.urlopen(url)
                    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
                    frame = cv2.imdecode(imgnp, -1)

                    # Extract faces from the frame
                    faces = extract_faces(frame)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                        cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                        if j % 5 == 0:
                            name = first_name + "_" + surname + '_' + str(i) + '.jpg'
                            cv2.imwrite(name_drive + '/' + name, frame[y:y+h, x:x+w])
                            i += 1
                        j += 1
                    if j == nimgs * 5:
                        break
                    cv2.imshow('Adding new User', frame)
                    if cv2.waitKey(1) == 27:  # Press 'ESC' to break
                        break
                cv2.destroyAllWindows()
                train_model()

            elif(ch == 3):
                print("Cancelling update profile......")
                break

    else:
        print("Student not found")


while True:
    print("################### MENU ######################################")
    print("Select operations")
    print("1. Add new student")
    print("2. Mark attendance")
    print("3. Find and update student")
    print("4. Exit")
    print("##############################################################")

    choice = input("Enter choice(1/2/3/4): ")
    if choice == '1':
        add_student()
    
    elif choice == '2':
        mark_attendance()

    elif choice == '3':
        update_student_profile()
    
    elif choice == '4':
        print("Thank you")
        break
