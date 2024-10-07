import struct
import cv2
import numpy as np
import pandas as pd
import os
from collections import defaultdict

# 设置 TS 文件路径和输出目录
ts_file_path = '20240711152253.ts'  # 在当前目录读取该文件
output_dir = os.getcwd()  # 将生成文件输出到当前目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定义 TS 包解析类
class TSPacket:
    def __init__(self, packet_data):
        self.packet_data = packet_data

    @property
    def packet_type(self):
        return self.packet_data[1] & 0x1F

    @property
    def payload(self):
        # 提取有效载荷
        adaptation_field_control = (self.packet_data[3] >> 4) & 0x03
        if adaptation_field_control == 0x00:
            # 保留，不包含适配字段和有效载荷
            return None
        elif adaptation_field_control == 0x01:
            # 仅包含有效载荷
            payload_start_index = 4
            return self.packet_data[payload_start_index:]
        elif adaptation_field_control == 0x02:
            # 仅包含适配字段
            return None
        elif adaptation_field_control == 0x03:
            # 包含适配字段和有效载荷
            adaptation_field_length = self.packet_data[4]
            payload_start_index = 5 + adaptation_field_length
            return self.packet_data[payload_start_index:]
        else:
            return None

# 定义 PES 包解析类
class PESPacket:
    def __init__(self, ts_packet):
        self.packet_data = ts_packet.payload

    @property
    def stream_id(self):
        if self.packet_data is not None and len(self.packet_data) >= 3:
            return self.packet_data[3]
        return None

    @property
    def data(self):
        if self.packet_data is not None and len(self.packet_data) >= 9:
            pes_header_data_length = self.packet_data[8]
            pes_data_start = 9 + pes_header_data_length
            return self.packet_data[pes_data_start:]
        return None

# 解析 SEI 数据
def parse_sei_data(sei_data):
    idx = 0
    while idx < len(sei_data):
        # 读取 payload_type
        payload_type = 0
        while sei_data[idx] == 0xFF:
            payload_type += 255
            idx += 1
        payload_type += sei_data[idx]
        idx += 1

        # 读取 payload_size
        payload_size = 0
        while sei_data[idx] == 0xFF:
            payload_size += 255
            idx += 1
        payload_size += sei_data[idx]
        idx += 1

        # 获取 SEI payload 数据
        sei_payload_data = sei_data[idx:idx + payload_size]
        idx += payload_size

        # 输出 SEI payload 类型和数据
        print(f"SEI Payload Type: {payload_type}")
        print(f"SEI Payload Data: {sei_payload_data.hex()}")

        # 具体解析 SEI 数据
        if payload_type == 5:  # 用户未注册数据
            parse_unregistered_sei_message(sei_payload_data)
        elif payload_type == 4:  # 用户注册数据
            parse_user_data_registered_id(sei_payload_data)

        # 检查是否是 RBSP trailing bits
        if idx < len(sei_data) and sei_data[idx] == 0x80:
            idx += 1
            break

# 解析注册的 SEI 消息
def parse_user_data_registered_id(data):
    if len(data) < 4:
        return
    registered_id = data[:4]
    remaining_data = data[4:]
    print(f"Registered ID: {registered_id.hex()}")
    print(f"Remaining Data: {remaining_data.hex()}")

# 解析未注册的 SEI 消息
def parse_unregistered_sei_message(data):
    if len(data) < 16:
        return
    uuid_iso_iec_11578 = data[:16]
    user_data_payload_byte = data[16:]
    print(f"UUID: {uuid_iso_iec_11578.hex()}")
    print(f"User Data: {user_data_payload_byte.hex()}")

# 解析 TS 文件并提取 SEI 信息
def parse_ts_file(ts_file_name, output_file_path):
    ts_file_path = os.path.join(os.getcwd(), ts_file_name)  # 在当前目录读取 TS 文件
    # 打开 TS 文件
    with open(ts_file_path, 'rb') as file:
        sei_info_list = []
        while True:
            # 读取 188 字节的 TS 包
            packet_data = file.read(188)
            if not packet_data or len(packet_data) < 188:
                break

            # 同步字节检查
            if packet_data[0] != 0x47:
                continue

            # 解析 TS 包
            ts_packet = TSPacket(packet_data)

            # 如果是 PES 包，尝试读取包含 SEI 信息的 PES 包
            if ts_packet.packet_type == 0x00:  # 根据实际的 PID 修改条件
                pes_packet = PESPacket(ts_packet)
                if pes_packet.stream_id == 0xE0:  # 视频流的 stream_id 通常为 0xE0
                    sei_info = pes_packet.data
                    if sei_info:
                        sei_info_list.append(sei_info)

    # 将 SEI 信息保存到文本文件
    with open(output_file_path, 'w') as output_file:
        for sei_info in sei_info_list:
            output_file.write(f"{sei_info.hex()}\n")

    # 解析并打印 SEI 信息
 ts_file_name = "20240711152253.ts"  # 在当前目录中查找该文件

# 调用函数读取 TS 文件并将 SEI 信息保存到文本文件
sei_output_path = os.path.join(os.getcwd(), "sei_info.txt")
parse_ts_file(ts_file_name, sei_output_path)

# 假设 sei_info_list 在 parse_ts_file 函数中生成
for sei_info in sei_info_list:
    parse_sei_data(sei_info)
# ----------------------- 目标检测与跟踪部分 -----------------------

# 加载 YOLOv4-tiny 模型
weights_path = os.path.join(os.getcwd(), 'yolov4-tiny.weights.bin')
config_path = os.path.join(os.getcwd(), 'yolov4-tiny.cfg')

net = cv2.dnn.readNet(weights_path, config_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# 读取类别文件
class_list = []
with open(os.path.join(os.getcwd(), 'classes.txt'), 'r') as txt:
    class_list = [line.strip() for line in txt.readlines()]

# 颜色映射表：为不同类别分配不同的颜色（RGB格式）
colors = {
    "person": (0, 255, 0),        # 绿色
    "bicycle": (255, 0, 255),     # 紫色
    "motorcycle": (255, 128, 0),  # 橙色
    "car": (0, 0, 255),           # 红色
    "truck": (255, 0, 0),         # 蓝色
    "bus": (255, 255, 0)          # 黄色
}

# 获取 YOLOv4-tiny 的输出层
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 初始化跟踪器相关的参数
object_id = 0  # 初始ID
previous_objects = []  # 记录上一帧中的物体（ID, 坐标, 尺寸, 类别）
max_distance = 150  # 距离阈值
size_threshold = 0.2  # 尺寸变化阈值
person_bike_distance = 80  # 人和非机动车之间的距离阈值

# 读取视频文件
video = cv2.VideoCapture(ts_file_path)

# 检查是否成功加载视频
if not video.isOpened():
    print("无法打开视频文件，检查路径或视频格式是否正确")
    exit()

# 获取视频宽度和高度
fps = video.get(cv2.CAP_PROP_FPS)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_video_path = os.path.join(output_dir, "检测结果.avi")
out = cv2.VideoWriter(output_video_path,
                      cv2.VideoWriter_fourcc(*'XVID'),
                      fps,
                      (frame_width, frame_height))

# 创建可调整大小的窗口
cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Object Detection", frame_width, frame_height)

# 存储检测结果的列表
detection_results = []
frame_count = 0

# 类别稳定性缓冲：用于记录每个ID的类别
object_class_buffer = defaultdict(lambda: {"class": None, "count": 0})

while True:
    ret, frame = video.read()
    if not ret:
        print("无法读取帧，可能视频已结束或格式不支持")
        break

    # 创建模型输入 (Blob)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    class_ids, confidences, boxes, current_objects = [], [], [], []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # 置信度阈值
                box = obj[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                (centerX, centerY, width, height) = box.astype("int")
                x = max(0, int(centerX - (width / 2)))
                y = max(0, int(centerY - (height / 2)))
                w = min(frame_width - x, int(width))
                h = min(frame_height - y, int(height))

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

                # 跟踪器部分：分配物体 ID
                new_id = None
                new_class = class_list[class_id]

                # 仅对特定类别进行跟踪
                if new_class in ["person", "bicycle", "motorcycle", "car", "truck", "bus"]:
                    for prev_obj in previous_objects:
                        prev_id, prev_x, prev_y, prev_w, prev_h, prev_class = prev_obj
                        distance = np.linalg.norm(np.array([x, y]) - np.array([prev_x, prev_y]))
                        size_diff = abs(w - prev_w) / max(prev_w, 1) + abs(h - prev_h) / max(prev_h, 1)

                        # 使用距离和尺寸变化来判断是否为同一目标
                        if distance < max_distance and size_diff < size_threshold:
                            # 当检测到“人”和“非机动车”接近时，标记为“非机动车”
                            if (new_class == "person" and prev_class in ["bicycle", "motorcycle"] and distance < person_bike_distance) or \
                               (prev_class == "person" and new_class in ["bicycle", "motorcycle"] and distance < person_bike_distance):
                                new_class = "bicycle" if prev_class == "bicycle" or new_class == "bicycle" else "motorcycle"

                            new_id = prev_id
                            break

                    if new_id is None:
                        new_id = object_id  # 新物体，分配新 ID
                        object_id += 1

                    current_objects.append((new_id, x, y, w, h, new_class))

                    # 类别稳定性逻辑
                    if object_class_buffer[new_id]["class"] == new_class:
                        object_class_buffer[new_id]["count"] += 1
                    else:
                        object_class_buffer[new_id]["class"] = new_class
                        object_class_buffer[new_id]["count"] = 1

                    # 如果类别在连续3帧中相同，则确认该类别
                    stable_class = object_class_buffer[new_id]["class"] if object_class_buffer[new_id]["count"] >= 3 else None

                    # 保存检测结果
                    if stable_class is not None:
                        detection_results.append({
                            "Frame": frame_count,
                            "ID": new_id,
                            "Class": stable_class,
                            "Confidence": confidence,
                            "X": x,
                            "Y": y,
                            "Width": w,
                            "Height": h
                        })
                        print(f"保存检测结果: ID {new_id}, 类别 {stable_class}, 置信度 {confidence}, 坐标 ({x}, {y}), 宽: {w}, 高: {h}")

    # 更新上一帧的物体信息
    previous_objects = current_objects

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
    print(f"保留了 {len(indices)} 个检测框")

    # 绘制边界框并标注
    if len(indices) > 0:
        for idx in indices.flatten():
            if idx < len(current_objects):
                x, y, w, h = boxes[idx]
                detected_class = object_class_buffer[current_objects[idx][0]]["class"] if object_class_buffer[current_objects[idx][0]]["count"] >= 3 else None
                if detected_class is not None:
                    label = f"ID {current_objects[idx][0]}: {detected_class}: {confidences[idx]:.2f}"

                    # 根据类别选择颜色
                    color = colors.get(detected_class, (255, 255, 255))

                    # 画出检测到的目标边界框
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    # 在边界框上方显示检测到的目标类别名称和置信度
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 显示检测结果的每一帧
    cv2.imshow("Object Detection", frame)

    # 使用 30ms 延迟确保窗口刷新
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    out.write(frame)  # 将每一帧写入输出视频
    frame_count += 1

# 释放视频文件并关闭所有窗口
video.release()
out.release()
cv2.destroyAllWindows()

# 保存检测结果到 CSV 文件
output_csv_path = os.path.join(output_dir, "检测结果.csv")
if detection_results:
    df = pd.DataFrame(detection_results)
    df.to_csv(output_csv_path, index=False)
    print(f"检测结果已保存到 {output_csv_path}")
else:
    print("没有检测结果保存")
