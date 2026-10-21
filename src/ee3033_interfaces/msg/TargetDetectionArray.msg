# All detections from one camera image.
# header.stamp is the image capture time and header.frame_id is the camera frame,
# so the observation can be placed in the map with TF at the right time.

std_msgs/Header header
uint32 image_width
uint32 image_height
TargetDetection[] detections
