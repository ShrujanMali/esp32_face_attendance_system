# Project Title: Face Recognition Based Attendance System

## Overview

This project demonstrates a face recognition-based attendance system using an ESP32 CAM (IoT device) combined with Python and various data processing libraries. The system can register new students, mark attendance by recognizing faces, and update student information or photos.
Features

1. Add New Student: Register a new student by providing name, surname, gender, and class. The system captures 10 photos of the student, stores the data, and trains the model.
2. Mark Attendance: Automatically marks attendance by recognizing faces through the camera. Only students in the database are marked present.
3. Update Student Information: Update student details such as class or update their photos in the database.
4. Exit: Exits the program.

Technology Stack

<p align="center">
  <a href="https://skillicons.dev">
    <img src="https://skillicons.dev/icons?i=python,opencv,cmake,github,arduino,sklearn,vscode,git" />
    
  </a>
</p>
    Python: Core programming language used for scripting.
    Numpy: Used for numerical operations.
    Pandas: Utilized for data manipulation and handling student data in Excel.
    OpenCV: Employed for image processing and face recognition.
    Scikit-learn: Used for training and managing machine learning models.
    ESP32 CAM: IoT device for capturing and processing images.

## Detailed Functionality

1. Add New Student
        Prompts for name, surname, gender, and class.
        Checks if the student already exists in the Excel sheet.
        If not, adds the student data to the Excel sheet, captures 10 photos, and trains the model.

 2.  Mark Attendance
        Initiates the camera.
        Recognizes faces and matches them with the database.
        Marks attendance for matched faces.

  3.  Update Student Information
        Prompts for the student's first name and surname.
        Checks if the student exists in the database.
        Options to update the class or photos:
            Update Class: Prompts for new class and updates it.
            Update Photo: Captures new photos, replaces old ones, and retrains the model.

  4.  Exit
        Exits the program.

## Installation and Setup

1. Clone the repository:

        git clone https://github.com/yourusername/face-recognition-attendance.git

2. Navigate to the project directory:

        cd face-recognition-attendance

3. Create and activate virtual environment

       >>> python -m venv <virtual_env_name>
       >>> env/Scripts/activate

4. Install the required libraries:

       >>> pip install -r requirements.txt

5. Install arduino ide from following website

        https://www.arduino.cc/

6. Open esp32cam.ino code in arduino ide and set the ssid(wifi router name) and password 

        const char* WIFI_SSID = "wifi name";
        const char* WIFI_PASS = "wifi password";

7. To upload the code into esp32 cam following the link for tutorial

        https://www.youtube.com/watch?v=vcBTt5KcdQw

8. Open serial monitor and copy the url and paste into python_esp32_attendance_system.py
9. Run the code
