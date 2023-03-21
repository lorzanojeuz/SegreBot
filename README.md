# SegreBot

### Description for the Main Code


### Description for the Conveyor Code
This code assumes that you have connected the **infrared sensor** to GPIO 4 and the **DC motor** to GPIO 17 and GPIO 27 using an **H-bridge motor driver**. 
![image](https://user-images.githubusercontent.com/119225795/226761648-fa73d744-2829-400b-87db-c138a24ee29a.png)

In this code, we first import the required libraries and set up the GPIO pins for the infrared sensor and DC motor. Then, we define two functions, clockwise() and counterclockwise(), that control the direction of the motor.

In the main loop, we continuously read the value of the infrared sensor using GPIO.input(4). If the sensor detects something, we call the clockwise() function to rotate the motor clockwise. If the sensor does not detect anything, we call the counterclockwise() function to rotate the motor counterclockwise. We then wait for a short time using time.sleep(0.1) to avoid excessive loop frequency.

Finally, we clean up the GPIO pins using GPIO.cleanup() to ensure that they are returned to their default state when the program exits.
