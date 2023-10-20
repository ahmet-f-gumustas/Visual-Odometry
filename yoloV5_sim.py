import cv2
import numpy as np
import torch
import math

object_class = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


def load_model():
    model_path = 'yolov5s.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    model.to(device)
    return model, device

def overlay_transparent(background, foreground, angle, x, y, objSize=50):
    original_frame = background.copy()
    foreground = cv2.resize(foreground, (objSize, objSize))
    rows, cols, channels = foreground.shape
    center_x = int(cols / 2)
    center_y = int(rows / 2)
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
    foreground = cv2.warpAffine(foreground, M, (cols, rows))
    for row in range(rows):
        for col in range(cols):
            if x + row < background.shape[0] and y + col < background.shape[1]:
                alpha = foreground[row, col, 3] / 255.0
                background[x + row, y + col] = alpha * foreground[row, col, :3] + (1 - alpha) * background[x + row, y + col]
    result = background
    return result

def simulate_object(background, object_class, x, y):
    object_img = cv2.imread(f'assets/{object_class}.png', cv2.IMREAD_UNCHANGED)
    if object_img is None:
        return background
    background[y:y+100, x:x+100] = overlay_transparent(background[y:y+100, x:x+100], object_img, 0, 0, 0)
    return background

def add_myCar_overlay(background):
    overlay_img = cv2.imread('assets/MyCar.png', cv2.IMREAD_UNCHANGED)
    rows, cols, _ = overlay_img.shape
    x = 550
    y = background.shape[0] - 200
    overlay_img = overlay_transparent(background[y:y+rows, x:x+cols], overlay_img, 0, 0, 0, objSize=250)
    background[y:y+rows, x:x+cols] = overlay_img
    return background

def plot_object_bev(transformed_image_with_centroids, src_points, dst_points, objs_):
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    persObjs = []
    for obj_ in objs_:
        if obj_:
            centroid_coords = np.array([list(obj_[0])], dtype=np.float32)
            transformed_coords = cv2.perspectiveTransform(centroid_coords.reshape(-1, 1, 2), M)
            transformed_coords_ = tuple(transformed_coords[0][0].astype(int)
            cv2.circle(transformed_image_with_centroids, transformed_coords_, radius=3, color=(0, 255, 0), thickness=-1)
            cv2.circle(transformed_image_with_centroids, transformed_coords_, radius=12, color=(255, 255, 255), thickness=1)
            class_text = f"Class: {obj_[1]}"
            cv2.putText(transformed_image_with_centroids, class_text, (transformed_coords_[0] + 10, transformed_coords_[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            persObjs.append([transformed_coords_, obj_[1])
    return transformed_image_with_centroids, persObjs

def main():
    model, device = load_model()
    video = cv2.VideoCapture('videos/test4.mp4')
    output_filename = 'output_video2.mp4'
    width, height = 1280, 720
    videoOut = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

    frame_count = 0
    centroid_prev_frame = []
    tracking_objects = {}
    tracking_id = 0

    while True:
        success, frame = video.read()
        frame = cv2.resize(frame, (width, height))
        frame_count += 1
        if not success:
            break

        results = model(frame, size=320)
        detections = results.pred[0]
        image_ = np.zeros((height, width, 3), dtype=np.uint8)
        simulated_image = image_.copy()
        transformed_image_with_centroids = image_.copy()
        transformed_image_to_sim = image_.copy()
        simObjs = image_.copy()
        objs = []
        centroid_curr_frame = []

        for detection in detections:
            xmin = detection[0]
            ymin = detection[1]
            xmax = detection[2]
            ymax = detection[3]
            score = detection[4]
            class_id = detection[5]
            centroid_x = int(xmin + xmax) // 2
            centroid_y = int(ymin + ymax) // 2

            if int(class_id) in [0, 1, 2, 3, 5, 7] and score >= 0.3:
                color = (0, 0, 255)
                object_label = f"{class_id}: {score:.2f}"
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(frame, object_label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
                centroid_curr_frame.append([(centroid_x, centroid_y), yolo_classes[int(class_id)])
                if track:
                    objs.append([(centroid_x, centroid_y), yolo_classes[int(class_id)])

        if track:
            if frame_count <= 2:
                for pt1, class_id in centroid_curr_frame:
                    for pt2, class_id in centroid_prev_frame:
                        dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
                        if dist < 50:
                            tracking_objects[tracking_id] = pt1, class_id
                            tracking_id += 1
            else:
                tracking_objects_copy = tracking_objects.copy()
                for obj_id, pt2 in tracking_objects_copy.items():
                    objects_exists = False
                    for pt1, class_id in centroid_curr_frame:
                        dist = math.hypot(pt2[0][0] - pt1[0], pt2[0][1] - pt1[1])
                        if dist < 20:
                            tracking_objects[obj_id] = pt1, class_id
                            objects_exists = True
                            continue
                    if not objects_exists:
                        tracking_objects.pop(obj_id)

            for obj_id, pt1 in tracking_objects.items():
                cv2.circle(frame, pt1[0], 3, (0, 255, 255), -1)
                objs.append([pt1[0], pt1[1])

            centroid_prev_frame = centroid_curr_frame.copy()

        x1, y1 = 10, 720
        x2, y2 = 530, 400
        x3, y3 = 840, 400
        x4, y4 = 1270, 720
        src_points = np.float32([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
        u1, v1 = 370, 720
        u2, v2 = 0+150, 0
        u3, v3 = 1280-150, 0
        u4, v4 = 900, 720
        dst_points = np.float32([[u1, v1], [u2, v2], [u3, v3], [u4, v4])

        transformed_image_with_centroids, persObjs_ = plot_object_bev(transformed_image_with_centroids, src_points, dst_points, objs)

        for persObj_ in persObjs_:
            simObjs = simulate_object(transformed_image_to_sim, persObj_[1], persObj_[0][0], persObj_[0][1])

        simulated_image = add_myCar_overlay(simObjs)

        videoOut.write(simulated_image)
        cv2.imshow("Video", frame)
        cv2.imshow("Simulated Objects", simulated_image)
        cv2.imshow('Transformed Frame', transformed_image_with_centroids)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    videoOut.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
