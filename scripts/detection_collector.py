#!/usr/bin/env python3

import rospy
import yaml
import numpy as np
np.float = float  # 修复兼容性
np.int = int
from geometry_msgs.msg import PoseArray, Pose
from sklearn.cluster import DBSCAN

class DetectionCollector:
    def __init__(self):
        self.detected = {}
        self.detection_names = rospy.get_param('/yolov8_detector/detection_classes/names')
        rospy.Subscriber('/cluster_decomposer/centroid_pose_array', PoseArray, self.collect)
        self.rate = rospy.Rate(1)  # 1Hz
        
        rospy.loginfo('Detection collector initialized, searching for objects...')
        
        # 周期性保存检测结果
        self.timer = rospy.Timer(rospy.Duration(30), self.save_detections)
        
    def save_detections(self, event=None):
        if not self.detected:
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

if __name__ == '__main__':
    rospy.init_node('detection_collector')
    collector = DetectionCollector()
    rospy.spin()