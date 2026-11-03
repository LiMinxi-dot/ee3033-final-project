"""Small ROS helpers shared by the nodes (no Nav2 dependency, so perception can use them too)."""

from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)


def marker(frame, ns, mid, mtype, scale, rgba, xy=None, z=0.0, stamp=None):
    """RViz marker; scale is one number for all axes or (x, y, z)."""
    mk = Marker()
    mk.header.frame_id = frame
    if stamp is not None:
        mk.header.stamp = stamp
    mk.ns, mk.id, mk.type = ns, mid, mtype
    if xy is not None:
        mk.pose.position.x, mk.pose.position.y = float(xy[0]), float(xy[1])
    mk.pose.position.z = float(z)
    mk.pose.orientation.w = 1.0
    mk.scale.x, mk.scale.y, mk.scale.z = scale if isinstance(scale, tuple) else (scale,) * 3
    mk.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
    return mk
