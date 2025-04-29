#!/usr/bin/env python3
import os
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Int32MultiArray, MultiArrayDimension
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
from ultralytics import YOLO
import torch
class YoloV8Detector:
    def __init__(self):
        rospy.init_node('yolov8_detector', anonymous=True)
        pkg_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(pkg_path, "weights", "yolov8n.pt")
        # 加载YOLOv8模型 - 使用小型模型以适应CPU
        self.model = YOLO(model_path).to('cpu')  # 'n' for nano version (smallest)
        
        # 创建CV bridge
        self.bridge = CvBridge()
        
        # 订阅相机图像
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        
        # 发布检测结果
        self.detection_pub = rospy.Publisher('/yolov8_detector/detections', BoundingBoxes, queue_size=1)
        self.detection_image_pub = rospy.Publisher('/yolov8_detector/detection_image', Image, queue_size=1)
        # self.label_pub = rospy.Publisher('/yolov8_detector/label_image', Int32MultiArray, queue_size=1)
        self.label_pub = rospy.Publisher('/yolov8_detector/label_image', Image, queue_size=1)

        # 存储类别名称
        self.classes = self.model.names
        rospy.set_param('/yolov8_detector/detection_classes/names', list(self.classes.values()))
        
        rospy.loginfo("YOLOv8 detector initialized")
        
    def image_callback(self, data):
        try:
            # 将ROS图像转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # 运行YOLOv8检测
            results = self.model(cv_image)
            
            # 创建标签图像（用于语义映射）
            label_image = np.zeros((cv_image.shape[0], cv_image.shape[1]), dtype=np.int32)
            
            # 创建检测结果消息
            detections_msg = BoundingBoxes()
            detections_msg.header = data.header
            detections_msg.image_header = data.header
            
            # 处理检测结果
            for i, det in enumerate(results[0].boxes):
                # 获取边界框
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                confidence = float(det.conf[0])
                class_id = int(det.cls[0])
                class_name = self.classes[class_id]
                
                # 如果置信度太低，跳过
                if confidence < 0.5:
                    continue
                    
                # 创建边界框消息
                box = BoundingBox()
                box.probability = confidence
                box.xmin = x1
                box.ymin = y1
                box.xmax = x2
                box.ymax = y2
                box.id = class_id
                box.Class = class_name
                detections_msg.bounding_boxes.append(box)
                
                # 在图像上绘制框和标签
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cv_image, f"{class_name}: {confidence:.2f}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 填充标签图像（用于后续的点云聚类）
                label_image[y1:y2, x1:x2] = class_id + 1  # +1 避免与背景混淆
            
            # 发布检测结果
            self.detection_pub.publish(detections_msg)
            
            # 发布带有边界框的图像
            self.detection_image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
            
            # 发布标签图像（改为sensor_msgs/Image）
            label_image_msg = self.bridge.cv2_to_imgmsg(label_image.astype(np.int32), encoding="32SC1")
            label_image_msg.header = data.header  # 记得加上header同步时间戳
            # label_image_msg.header.stamp = data.header.stamp  # 继承原始图的时间戳
            # label_image_msg.header.frame_id = data.header.frame_id# frame_id 也要继承，最好保持一致
            self.label_pub.publish(label_image_msg)

            # 添加调试输出
            # rospy.loginfo(f"Publishing label image with shape {label_image.shape} and {np.count_nonzero(label_image)} non-zero pixels")

            # 发布标签图像
            # label_msg = Int32MultiArray()
            # label_msg.layout.dim = [
            #     MultiArrayDimension(label='height', size=label_image.shape[0], stride=label_image.shape[0] * label_image.shape[1]),
            #     MultiArrayDimension(label='width', size=label_image.shape[1], stride=label_image.shape[1])
            # ]
            # label_msg.data = label_image.flatten().tolist()
            # self.label_pub.publish(label_msg)
            
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Error in detection: {e}")

if __name__ == '__main__':
    try:
        detector = YoloV8Detector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass