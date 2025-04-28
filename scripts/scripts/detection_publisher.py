#!/usr/bin/env python3

import rospy
import yaml
import argparse
from os.path import expanduser
from tb3_semantic_mapping.srv import GetObjectLocation, GetObjectLocationResponse
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Pose, Point

class DetectionPublisher:
    def __init__(self, filename):
        try:
            with open(filename, 'r') as infile:
                self.detections = yaml.safe_load(infile)
        except Exception as e:
            rospy.logerr(f"Failed to load detection file: {e}")
            self.detections = {}
            
        self.marker_id = 0
        self.rate = rospy.Rate(2)
        self.map_objects = self.make_marker_array(self.detections)
        self.map_publisher = rospy.Publisher('~map_objects', MarkerArray, queue_size=10)
        self.map_server = rospy.Service('~object_location', GetObjectLocation, self.object_location)
        
        rospy.loginfo("Detection publisher initialized")
        self.process()

    def make_marker(self, name, x, y, z):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()

        marker.ns = "semantic_objects"
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.id = self.marker_id
        self.marker_id += 1

        marker.text = name
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z + 0.3  # 文本显示在物体上方

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.scale.z = 0.3  # 文本大小

        return marker

    def make_marker_array(self, detections):
        markers = []
        for name, positions in detections.items():
            for pos in positions:
                marker = self.make_marker(name, *pos)
                markers.append(marker)
        return MarkerArray(markers)

    def object_location(self, req):
        pose_array = PoseArray()
        pose_array.header.frame_id = "map"
        pose_array.header.stamp = rospy.Time.now()

        if req.object_name in self.detections:
            poses = []
            for point in self.detections[req.object_name]:
                pose = Pose()
                pose.position = Point(*point)
                poses.append(pose)
            pose_array.poses = poses
        return GetObjectLocationResponse(pose_array)

    def process(self):
        rospy.loginfo("Starting detection publisher")
        while not rospy.is_shutdown():
            self.map_publisher.publish(self.map_objects)
            self.rate.sleep()

if __name__ == '__main__':
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, default='/tmp/detections_dbscan.yaml')
    args, _ = parser.parse_known_args()

    # 初始化
    rospy.init_node('detection_publisher')
    DetectionPublisher(args.input_file)