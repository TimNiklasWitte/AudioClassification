#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
import zmq
import signal

def SpeechClassifier():
    pub = rospy.Publisher('movement', String, queue_size=10)
    rospy.init_node('SpeechClassifier', anonymous=True)
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5555")
    socket.setsockopt(zmq.SUBSCRIBE, "")
    while not rospy.is_shutdown():
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        message = socket.recv_string()
        print("Received:", message)
        msg_str = "Received: %s" % message
        rospy.loginfo(msg_str)
        pub.publish(message)

if __name__ == '__main__':
    try:
        SpeechClassifier()
    except rospy.ROSInterruptException:
        pass