# SegreBot

### Description for the Main Code


The code provided is a good start, but it will need some modifications to achieve the objective you described.

First, the process_image function detects the edges and draws bounding boxes around them but doesn't identify moving objects. To identify moving objects, you will need to compare the current frame to the previous frame and detect the differences using background subtraction or frame differencing techniques.

Second, the map_position_to_angles function maps the position of the object to the servo motor angles, but it doesn't take into account the location of the object relative to the arm. To move the arm to the left or right depending on the location of the object, you will need to compare the x-coordinate of the object to the center of the image and adjust the angles of the appropriate servo motors accordingly.

Third, the code doesn't include any logic to pick up and place the object. You will need to add code to control the gripper servo motor and the movements of the arm to pick up and place the object in the desired location.

Finally, the code waits for the infrared sensor to detect something before starting the object detection and tracking process. This may not be the most efficient way to start the process as it requires a physical trigger. Instead, you may want to consider starting the process automatically when the program is run and have it continuously detect and track objects until a specified condition is met.

In summary, the code provided is a good starting point, but it will need modifications and additional logic to achieve the objective you described.
