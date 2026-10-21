# Fused estimate of the target position.

std_msgs/Header header          # frame_id is the map frame
geometry_msgs/Point position    # target centre (z unused)
string class_name
uint32 num_observations         # observations supporting this estimate
uint32 total_observations       # all observations received so far
float32 spread                  # median distance of supporting observations to the estimate (m)
bool confirmed                  # enough consistent observations to report
