#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
import zmq

def talker():
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5555")
    socket.setsockopt(zmq.SUBSCRIBE, "")
    while not rospy.is_shutdown():
        message = socket.recv_string()
        print("Received:", message)
        hello_str = "Received: %s" % message
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass