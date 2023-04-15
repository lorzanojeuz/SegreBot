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

    # Initialize the previous frame
    _, prev_frame = camera.read()

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

        # Process the image to detect moving objects
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Check if the previous frame is available
        if prev_frame is not None:
            # Compute the absolute difference between the current and previous frames
            frame_delta = cv2.absdiff(prev_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

            # Dilate the thresholded image to fill in holes
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Iterate through the contours and draw a bounding box around each moving object
            for contour in contours:
                if cv2.contourArea(contour) < 5000:  # Set a minimum size threshold to filter out small objects
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Update the previous frame
        prev_frame = gray

        # Display the processed image
        cv2.imshow('Processed Image', image)

        # Find the center of the moving object in the image
        object_center = None
        if len(contours) > 0:
            # Find the contour with the largest area
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) >= 5000:  # Set a minimum size threshold to filter out small objects
                # Compute the center of the contour
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    object_center = (cx, cy)
        # Adjust the position of the servo motors based on the location of the object
        if object_center is not None:
            x, y = object_center
            # Calculate the error between the current position of the object and the center of the image
            error_x = x - 320
            error_y = y - 240

            # Calculate the PWM values to adjust the servo motors
            pwm_values = []
            for i in range(6):
                if i % 2 == 0:  # Servos 0, 2, 4 control the pan angle
                    pwm_values.append(initial_position[i] + error_x)
                else:  # Servos 1, 3, 5 control the tilt angle
                    pwm_values.append(initial_position[i] + error_y)

            # Set the PWM values for the servo motors
            for i in range(6):
                pwm.set_pwm(i, 0, pwm_values[i])

        # Wait for a key press to exit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up the resources
    camera.release()
    cv2.destroyAllWindows()
    pwm.software_reset()

# Call the main function
if name == 'main':
    main()
