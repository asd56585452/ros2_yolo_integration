import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import os
from ament_index_python.packages import get_package_share_directory
import torch
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
import tf2_ros
import tf2_geometry_msgs  # 用來做點座標的 TF 轉換
from image_geometry import PinholeCameraModel


class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__("yolo_detection_node")

        # 初始化 cv_bridge
        self.bridge = CvBridge()

        self.latest_depth_image_raw = None
        self.latest_depth_image_compressed = None

        # 使用 yolo model 位置
        model_path = os.path.join(
            get_package_share_directory("yolo_example_pkg"), "models", "tennis_v2.pt"
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device : ", device)
        self.model = YOLO(model_path)
        self.model.to(device)

        # 訂閱影像 Topic
        self.image_sub = self.create_subscription(
            CompressedImage, "/camera/image/compressed", self.image_callback, 1
        )

        # 訂閱 **無壓縮** 深度圖 Topic
        self.depth_sub_raw = self.create_subscription(
            Image, "/camera/depth/image_raw", self.depth_callback_raw, 1
        )

        # 訂閱 **壓縮** 深度圖 Topic
        self.depth_sub_compressed = self.create_subscription(
            CompressedImage,
            "/camera/depth/compressed",
            self.depth_callback_compressed,
            1,
        )

        # 發佈處理後的影像 Topic
        self.image_pub = self.create_publisher(
            CompressedImage, "/yolo/detection/compressed", 10
        )

        # 發布 目標檢測數據 (是否找到目標 + 距離)
        self.target_pub = self.create_publisher(
            Float32MultiArray, "/yolo/target_info", 10
        )

        self.x_multi_depth_pub = self.create_publisher(
            Float32MultiArray, "/camera/x_multi_depth_values", 10
        )

        # 設定要過濾標籤 (如果為空，那就不過濾)
        self.allowed_labels = {"tennis"}

        # 設定 YOLO 可信度閾值
        self.conf_threshold = 0.5  # 可以修改這個值來調整可信度

        # 相機畫面中央高度上切成 n 個等距水平點。
        self.x_num_splits = 20

        # 🌟 新增 1：訂閱相機內參 (Camera Info)
        self.camera_model = PinholeCameraModel()
        self.camera_info_received = False

        self.camera_info_sub = self.create_subscription(
            CameraInfo, "/camera/color/camera_info", self.camera_info_callback, 1
        )

        # 🌟 新增 2：初始化 TF2 監聽器 (用來讀取 URDF 的相對位置)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 🌟 新增 3：發布 Marker 供 Foxglove 顯示
        self.marker_pub = self.create_publisher(Marker, "/yolo/target_marker", 1)
    
    def camera_info_callback(self, msg):
        """接收相機內參，並更新給 image_geometry 模型"""
        if not self.camera_info_received:
            # PinholeCameraModel 會自動解析 K, D, R, P 矩陣
            self.camera_model.fromCameraInfo(msg)
            self.camera_info_received = True

    def publish_map_coordinate(self, u, v, depth):
        """使用 image_geometry 將像素與深度精確轉換為 3D 座標"""
        if not self.camera_info_received:
            self.get_logger().warning("尚未收到 camera_info，無法計算絕對座標！")
            return

        # 1. 將 YOLO 找到的原始像素 (u, v) 進行「去畸變 (Rectification)」
        # 這會考慮 CameraInfo 中的 D 矩陣，消除鏡頭邊緣的魚眼變形
        rectified_uv = self.camera_model.rectifyPoint((u, v))

        # 2. 獲取指向該像素的 3D 射線 (Ray)
        # 這會回傳一個向量 (x, y, z)，通常 z = 1.0
        ray = self.camera_model.projectPixelTo3dRay(rectified_uv)

        # 3. 根據實際測量到的 depth，將射線按比例延伸，得到精確的 3D 座標
        # (除以 ray[2] 是為了確保正規化，避免不同版本底層實作差異)
        scale = depth / ray[2]
        x_cam = ray[0] * scale
        y_cam = ray[1] * scale
        z_cam = depth

        # --- 以下完全與前一篇相同，將相機座標轉換到 Map 座標 ---
        point_cam = PointStamped()
        point_cam.header.frame_id = self.camera_model.tfFrame() # 自動取得相機 frame 名稱
        point_cam.header.stamp = self.get_clock().now().to_msg()
        point_cam.point.x = float(x_cam)
        point_cam.point.y = float(y_cam)
        point_cam.point.z = float(z_cam)

        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 
                point_cam.header.frame_id, 
                rclpy.time.Time()
            )
            point_map = tf2_geometry_msgs.do_transform_point(point_cam, transform)

            # 4. 發布 Foxglove 視覺化 Marker (畫一顆球)
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'yolo_target'
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = point_map.point
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.08  # 網球大小約 8 公分
            marker.scale.y = 0.08
            marker.scale.z = 0.08
            marker.color.a = 1.0   # 不透明度
            marker.color.r = 0.8   # 螢光黃/綠色
            marker.color.g = 1.0
            marker.color.b = 0.0

            self.marker_pub.publish(marker)
            
            # (可選) 印出絕對座標供除錯
            # self.get_logger().info(f"網球絕對位置: X={point_map.point.x:.2f}, Y={point_map.point.y:.2f}, Z={point_map.point.z:.2f}")

        except Exception as e:
            pass # 偶爾 TF 延遲是正常的，直接忽略不印出錯誤避免洗畫面

    def depth_callback_raw(self, msg):
        """接收 **無壓縮** 深度圖"""
        try:
            self.latest_depth_image_raw = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
        except Exception as e:
            self.get_logger().error(f"Could not convert raw depth image: {e}")

    def depth_callback_compressed(self, msg):
        """接收 **壓縮** 深度圖（當無壓縮深度圖不可用時使用）"""
        try:
            self.latest_depth_image_compressed = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
        except Exception as e:
            self.get_logger().error(f"Could not convert compressed depth image: {e}")

    def image_callback(self, msg):
        """接收影像並進行物體檢測"""
        # 將 ROS 影像消息轉換為 OpenCV 格式
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        # 使用 YOLO 模型檢測物體
        try:
            results = self.model(cv_image, conf=self.conf_threshold, verbose=False)
        except Exception as e:
            self.get_logger().error(f"Error during YOLO detection: {e}")
            return

        # 繪製 Bounding Box
        processed_image = self.draw_bounding_boxes(cv_image, results)

        # 取得影像中心深度並發布
        self.publish_x_multi_depths(processed_image)

        # 發佈處理後的影像
        self.publish_image(processed_image)

    def draw_cross(self, image):
        # 回傳繪製十字架的影像和畫面正中間的像素座標
        height, width = image.shape[:2]
        cx_center = width // 2
        cy_center = height // 2
        # 繪製橫線
        cv2.line(image, (0, cy_center), (width, cy_center), (0, 0, 255), 2)

        # 繪製直線
        cv2.line(
            image,
            (cx_center, cy_center - 10),
            (cx_center, cy_center + 10),
            (0, 0, 255),
            2,
        )

        cv2.line(
            image,
            (cx_center, cy_center - 10),
            (cx_center, cy_center + 10),
            (0, 0, 255),
            2,
        )

        # 計算橫線上的 n 個等分點
        segment_length = width // self.x_num_splits
        points = [
            (i * segment_length, cy_center) for i in range(self.x_num_splits + 1)
        ]  # 11 個點表示 10 段區間的端點

        # 在每個等分點繪製垂直的短黑線
        for x, y in points:
            cv2.line(image, (x, y - 10), (x, y + 10), (0, 0, 0), 2)  # 黑色垂直線

        return image, points

    def draw_bounding_boxes(self, image, results):
        """在影像上繪製 YOLO 檢測到的 Bounding Box，並挑選最近的目標"""
        found_target = 0
        best_distance = float('inf')  # 尋找最小值用的初始無限大
        best_delta_x = 0.0

        image, points = self.draw_cross(image)
        
        # 取得畫面的精確正中心 X 座標
        height, width = image.shape[:2]
        cx_center = width // 2

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                # 只保留設定內的標籤 (例如 tennis)
                if self.allowed_labels and class_name not in self.allowed_labels:
                    continue

                # 計算 Bounding Box 正中心點
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # 優先使用無壓縮的深度圖
                depth_value = self.get_depth_at(cx, cy)
                depth_text = f"{depth_value:.2f}m" if depth_value > 0 else "N/A"

                # 繪製框和標籤 (所有看到的網球都畫框)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f} Depth: {depth_text}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 🌟 目標過濾邏輯：只記錄「有效深度」且「距離最近」的那顆球
                if depth_value > 0.1 and depth_value < best_distance:
                    best_cx = cx  # 🌟 紀錄最佳目標的像素 X
                    best_cy = cy  # 🌟 紀錄最佳目標的像素 Y
                    best_distance = depth_value
                    best_delta_x = float(cx - cx_center) # 修正偏移量算法
                    found_target = 1

        if found_target == 1 and best_distance < 99.0:
            self.publish_map_coordinate(best_cx, best_cy, best_distance)
        # 迴圈結束後，只發布那顆「最完美」的目標資訊
        final_distance = best_distance if found_target == 1 else 0.0
        self.publish_target_info(found_target, final_distance, best_delta_x)
        
        return image

    def get_depth_at(self, x, y):
        """
        取得指定像素的深度值，轉換為米 (m)
        若深度出問題，回傳 -1
        """
        # **優先使用無壓縮的深度圖**
        depth_image = (
            self.latest_depth_image_raw
            if self.latest_depth_image_raw is not None
            else self.latest_depth_image_compressed
        )

        if depth_image is None:
            return -1.0

        # 如果深度影像為三通道，那只取第一個數值
        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0]

        try:
            depth_value = depth_image[y, x]
            if depth_value < 0.0001 or depth_value == 0.0:  # 無效深度
                return -1.0
            return depth_value / 1000.0  # 16-bit 深度圖通常單位為 mm，轉換為 m
        except IndexError:
            return -1.0

    def publish_image(self, image):
        """將處理後的影像轉換並發佈到 ROS"""
        try:
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(image)
            self.image_pub.publish(compressed_msg)
        except Exception as e:
            self.get_logger().error(f"Could not publish image: {e}")

    def publish_target_info(self, found, distance, delta_x):
        """發佈目標資訊 (找到目標, 距離)"""
        msg = Float32MultiArray()
        msg.data = [float(found), float(distance), float(delta_x)]
        self.target_pub.publish(msg)

    def publish_x_multi_depths(self, image):
        """
        取得畫面 n 個等分點的深度並發布
        """
        height, width = image.shape[:2]
        cy_center = height // 2  # 固定 Y 座標在畫面中心
        segment_length = width // self.x_num_splits

        # 計算 10 個等分點的 X 座標
        points = [(i * segment_length, cy_center) for i in range(self.x_num_splits)]

        # 取得每個等分點的深度值
        depth_values = [self.get_depth_at(x, cy_center) for x, _ in points]

        # 以 Float32MultiArray 發布
        depth_msg = Float32MultiArray()
        depth_msg.data = depth_values
        self.x_multi_depth_pub.publish(depth_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
