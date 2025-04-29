#!/usr/bin/env python3

import rospy
import yaml
import numpy as np
np.float = float  # 修复兼容性
np.int = int
from geometry_msgs.msg import PoseArray, Pose
from sklearn.cluster import DBSCAN
from visualization_msgs.msg import Marker
from std_srvs.srv import Trigger, TriggerResponse  

class DetectionCollector:
    def __init__(self):
        # self.marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
        self.detected = {}
        self.detection_names = rospy.get_param('/yolov8_detector/detection_classes/names')
        rospy.Subscriber('/cluster_decomposer/centroid_pose_array', PoseArray, self.collect)
        self.rate = rospy.Rate(1)  # 1Hz

        self.save_service = rospy.Service('~save_now', Trigger, self.save_service_callback)

        rospy.loginfo('Detection collector initialized, searching for objects...')
        
        # 周期性保存检测结果
        self.timer = rospy.Timer(rospy.Duration(30), self.save_detections)
    
    def save_service_callback(self, req):
        try:
            self.save_detections()
            return TriggerResponse(success=True, message="Detections saved successfully")
        except Exception as e:
            return TriggerResponse(success=False, message=f"Error saving detections: {e}")

    def save_detections(self, event=None):
        if not self.detected:
            rospy.loginfo("No detections to save, skipping...")
            return
            
        rospy.loginfo("Saving raw detections...")
        try:
            with open('/tmp/detections_raw.yaml', 'w') as outfile:
                yaml.dump(self.detected, outfile)
            
            rospy.loginfo("Processing detections with DBSCAN...")
            clusters = self.cluster_detections(self.detected)
            
            with open('/tmp/detections_dbscan.yaml', 'w') as outfile:
                yaml.dump(clusters, outfile)
                
            rospy.loginfo(f"Saved detection data. Found {len(clusters)} object types.")
        except Exception as e:
            rospy.logerr(f"Error saving detections: {e}")
    
    def cluster_detections(self, detections, eps=0.3, min_samples=10):
        """使用DBSCAN聚类检测结果"""
        clusters = {}
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        
        for name, pos_list in detections.items():
            if len(pos_list) < min_samples:
                continue
                
            X = np.array(pos_list)
            db = dbscan.fit(X)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            
            # 计算每个聚类的中心点
            label_types = set(labels)
            if -1 in label_types:
                label_types.remove(-1)
            
            result = []
            for k in label_types:
                class_member_mask = (labels == k)
                points = X[class_member_mask & core_samples_mask]
                if len(points) > 0:
                    mean = points.mean(axis=0)
                    result.append(mean.tolist())
            
            if result:
                clusters[name] = result
                
        return clusters

    def update_key(self, key, val):
        if key in self.detected:
            self.detected[key].append(val)
        else:
            self.detected[key] = [val]

    def collect(self, msg):
        # rospy.loginfo(f"Received centroid pose array with {len(msg.poses)} poses")
        for i, pose in enumerate(msg.poses):
            # 检查pose是否有效，跳过空的pose
            if pose.position.x == 0 and pose.position.y == 0 and pose.position.z == 0:
                continue
                
            pos = pose.position
            val = [pos.x, pos.y, pos.z]
            
            # 确保检测序号在有效范围内
            if i < len(self.detection_names):
                key = self.detection_names[i]
                rospy.loginfo(f'Found a {key} at {val}')
                self.update_key(key, val)

    # def publish_marker(self, class_name, position, id):
    #     marker = Marker()
    #     marker.header.frame_id = "map"  # 或者是你的全局frame，比如odom或者camera_link
    #     marker.header.stamp = rospy.Time.now()

    #     marker.ns = "detections"
    #     marker.id = id  # 每个marker一个唯一ID
    #     marker.type = Marker.TEXT_VIEW_FACING
    #     marker.action = Marker.ADD

    #     marker.pose.position.x = position.x
    #     marker.pose.position.y = position.y
    #     marker.pose.position.z = position.z + 0.3  # 字放在物体上面一点
    #     marker.pose.orientation.w = 1.0

    #     marker.scale.z = 0.2  # 字体大小
    #     marker.color.a = 1.0  # 不透明
    #     marker.color.r = 1.0
    #     marker.color.g = 1.0
    #     marker.color.b = 1.0

    #     marker.text = class_name  # 这里写物体类别！

    #     self.marker_pub.publish(marker)

if __name__ == '__main__':
    rospy.init_node('detection_collector')
    collector = DetectionCollector()
    rospy.spin()