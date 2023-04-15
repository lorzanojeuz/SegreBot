import csv
import numpy as np
import cv2
import time
import Adafruit_PCA9685
import RPi.GPIO as GPIO
import gpiozero

# Define the servo motor configurations
servo_min = [150, 150, 150, 150, 150, 150]
servo_max = [600, 600, 600, 600, 600, 600]

# Define the PCA9685 servo driver
pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(50)

# Define the infrared sensor pin
ir_sensor_pin = 18

# Set up the GPIO mode
GPIO.setmode(GPIO.BCM)
GPIO.setup(ir_sensor_pin, GPIO.IN)

# Define the image processing function
def process_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect the edges in the image using Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Find the contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours and draw a bounding box around each object
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Return the image with the bounding boxes drawn around the objects
    return image

# Define the function that maps the position of the object to the servo motor angles
def map_position_to_angles(x, y):
    # Calculate the angle for the base servo motor
    angle1 = np.interp(x, [0, 640], [servo_min[0], servo_max[0]])

    # Calculate the angle for the shoulder servo motor
    angle2 = np.interp(y, [0, 480], [servo_min[1], servo_max[1]])

    # Calculate the angle for the elbow servo motor
    angle3 = np.interp(x, [0, 640], [servo_min[2], servo_max[2]])

    # Calculate the angle for the wrist servo motor
    angle4 = np.interp(y, [0, 480], [servo_min[3], servo_max[3]])

    # Calculate the angle for the gripper servo motor
    angle5 = np.interp(x, [0, 640], [servo_min[4], servo_max[4]])

    # Calculate the angle for the rotation servo motor
    angle6 = np.interp(y, [0, 480], [servo_min[5], servo_max[5]])

    # Return the angles for the servo motors
    return angle1, angle2, angle3, angle4, angle5, angle6

# Define the main function
def main():
    # Set up the infrared sensor
    infrared_sensor = gpiozero.DigitalInputDevice(ir_sensor_pin)

    # Wait for the infrared sensor to detect something
    print("Waiting for detection...")
    while not infrared_sensor.is_active:
        time.sleep(1)

    # Set up the camera
    camera = cv2.VideoCapture(0)
    time.sleep(2)

    # Set the initial position of the servo motors
    initial_position = [375, 240, 375, 240, 375, 240]
    for i in range(6):
        pwm.set_pwm(i, 0, initial_position[i])

    # Initialize the timer
    last_detection_time = time.time()

    # Loop through the images from the camera
    while True:
        # Check if the infrared sensor is still active
        if not infrared_sensor.is_active:
            # Check if the timer has exceeded 30 seconds since the last detection
            if time.time() - last_detection_time > 30:
                print("No detection for 30 seconds. Turning off...")
                break
            else:
                time.sleep(1)
                continue

        # Update the timer
        last_detection_time = time.time()

        # Read an image from the camera
        _, image = camera.read()

        # Process the image to detect the objects
        processed_image = process_image(image)

        # Display the processed image
        cv2.imshow('Processed Image', processed_image)

        # Find the position of the object in the image
        x, y, _, _ = cv2.boundingRect(cv2.findContours(cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0), 50, 150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])

        # Map the position of the object to the servo motor angles
        angle1, angle2, angle3, angle4, angle5, angle6 = map_position_to_angles(x, y)

        # Set the angles of the servo motors
        pwm.set_pwm(0, 0, int(angle1))
        pwm.set_pwm(1, 0, int(angle2))
        pwm.set_pwm(2, 0, int(angle3))
        pwm.set_pwm(3, 0, int(angle4))
        pwm.set_pwm(4, 0, int(angle5))
        pwm.set_pwm(5, 0, int(angle6))

        # Wait for a key press to exit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    camera.release()
    cv2.destroyAllWindows()

