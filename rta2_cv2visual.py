import os
import cv2
import json
import yaml
from radar_points import RadarData, StaticPoints
from preprocess import load_data_sensorhost, rot_mtx_entry, rot_mtx_exit
from radar_clustering import *

from ultralytics import YOLO
import time
from pathlib import Path
import sys
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
import random
from utils.torch_utils import select_device
import tracking_tools
from utils.general import xywh2xyxy


# ------------------ LOAD YOLO MODEL ------------------#
model = YOLO('configs/yolov8m.pt')
# yolo_results = model.track(source=config["Files"]["video_file"], conf=0.3, iou=0.5, classes=0, show=True,
# save_txt=True)
names, = model.names,

# ------------------ CONFIGURE SS++----------------------#
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH

strong_sort_weights = WEIGHTS / 'osnet_x0_25_msmt17.pt'  # model.pt path
device = '0'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
half = False,  # use FP16 half-precision inference

# initialize StrongSORT
cfg = get_config()
cfg.merge_from_file('strong_sort/configs/strong_sort.yaml')
nr_sources = 3

# Load model
device = select_device(device)

# Create as many strong sort instances as there are tracking streams
strongsort_list = []

for i in range(nr_sources):
    if i == 0:  # Set config for YOLO Tracker
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                half,
                max_dist=cfg.STRONGSORT_Yolo.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT_Yolo.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT_Yolo.MAX_AGE,
                n_init=cfg.STRONGSORT_Yolo.N_INIT,
                nn_budget=cfg.STRONGSORT_Yolo.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT_Yolo.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT_Yolo.EMA_ALPHA,

            )

        )
    elif i == 1:
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                half,
                max_dist=cfg.STRONGSORT_Radar.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT_Radar.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT_Radar.MAX_AGE,
                n_init=cfg.STRONGSORT_Radar.N_INIT,
                nn_budget=cfg.STRONGSORT_Radar.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT_Radar.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT_Radar.EMA_ALPHA,

            )

        )
    elif i == 2:
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                half,
                max_dist=cfg.STRONGSORT_Fusion.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT_Fusion.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT_Fusion.MAX_AGE,
                n_init=cfg.STRONGSORT_Fusion.N_INIT,
                nn_budget=cfg.STRONGSORT_Fusion.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT_Fusion.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT_Fusion.EMA_ALPHA,

            )

        )

    strongsort_list[i].model.warmup()
outputs = [None] * nr_sources

colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

dt, seen = [0.0, 0.0, 0.0, 0.0], 0

curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources



# ------------------- DATA PREPROCESS ------------------ #
# load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# check if video file exists
if not os.path.isfile(config["Files"]["video_file"]):
    raise FileNotFoundError(f"Video file does not exist.")

# load json data
radar_data_file = config["Files"]["radar_data_file"]
with open(radar_data_file) as json_file:
    data = json.load(json_file)

# use sensorhost format
radar_data = load_data_sensorhost(data)  # Original coordinates
print(f"Radar data loaded.\n{radar_data}\n")
TOTAL_DATA_S = (radar_data.ts[-1] - radar_data.ts[0]) / 1000  # total seconds of data, before removing points

# Apply transformation
alpha = config["SensorAngles"]["alpha"]
beta = config["SensorAngles"]["beta"]
# distance of sensor from gate centre, positive in mm
offsetx = config["SensorOffsets"]["offsetx"]
offsety = config["SensorOffsets"]["offsety"]
offsetz = config["SensorOffsets"]["offsetz"]
print(f"{alpha = }")
print(f"{beta = }")
print(f"{offsetx = }")
print(f"{offsety = }")
print(f"{offsetz = }")
s1_rotz, s1_rotx = rot_mtx_entry(alpha, beta)
s2_rotz, s2_rotx = rot_mtx_exit(alpha, beta)

radar_data.transform_coord(
    s1_rotz, s1_rotx, s2_rotz, s2_rotx, offsetx, offsety, offsetz
)
print(f"Radar data transformed.\n{radar_data}\n")

# ------------------ VISUALIZATION PARAMS ------------------ #
rad_cam_offset = config["rad_cam_offset"]
scalemm2px = config["scalemm2px"]
wait_ms = config["wait_ms"]
slider_xoffset = config["TrackbarDefaults"]["slider_xoffset"]
slider_yoffset = config["TrackbarDefaults"]["slider_yoffset"]
xy_trackbar_scale = config["TrackbarDefaults"]["xy_trackbar_scale"]

print(f"{rad_cam_offset = }")
print(f"{scalemm2px = }")
print(f"{wait_ms = }")
print(f"{slider_xoffset = }")
print(f"{slider_yoffset = }")
print(f"{xy_trackbar_scale = }")

# ------------------ CV2 SUPPORT FUNCTIONS ------------------ #

# BGR colours for drawing points on frame (OpenCV)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)


def washout(color, factor=0.2):
    # create washed out color
    return (int(color[0] * factor), int(color[1] * factor), int(color[2] * factor))


def x_trackbar_callback(*args):
    # updates global offsets by trackbar value
    global slider_xoffset
    slider_xoffset = cv2.getTrackbarPos("x offset", "Radar Visualization")


def y_trackbar_callback(*args):
    # updates global offsets by trackbar value
    global slider_yoffset
    slider_yoffset = cv2.getTrackbarPos("y offset", "Radar Visualization")


def scale_callback(*args):
    # multiplies x and y by scale value from trackbar
    global xy_trackbar_scale
    xy_trackbar_scale = cv2.getTrackbarPos("scale %", "Radar Visualization") / 100


# draw gate at top left of window, with width and height of gate.
# Scale to match gate location with trackbar - returns valid display region
def draw_gate_topleft():
    # initial coords at top left corner (0,0)
    rect_start = (
        (slider_xoffset),
        (slider_yoffset)
    )
    # rect end initial coords are based on the physical width and height of the gate
    rect_end = (
        (int(offsetx * 2 * scalemm2px * xy_trackbar_scale) + slider_xoffset),
        (int(offsety * 2 * scalemm2px * xy_trackbar_scale) + slider_yoffset)
    )
    cv2.rectangle(frame, rect_start, rect_end, BLUE, 2)
    return rect_start, rect_end


def remove_points_outside_gate(points, rect_start, rect_end) -> list:
    """Remove points that are outside the gate area. 
    Returns a list of points that are inside the gate area."""
    points_in_gate = []
    for coord in points:
        x = int((coord[0] + offsetx) * scalemm2px)
        y = int((-coord[1] + offsety) * scalemm2px)
        x = int(x * xy_trackbar_scale)
        y = int(y * xy_trackbar_scale)
        x += slider_xoffset
        y += slider_yoffset
        if x < rect_start[0] or x > rect_end[0]:
            continue
        if y < rect_start[1] or y > rect_end[1]:
            continue
        points_in_gate.append(coord)
    return points_in_gate


def draw_radar_points(points, sensor_id):
    if sensor_id == 1:
        color = GREEN
    elif sensor_id == 2:
        color = YELLOW
    else:
        raise
    for coord in points:
        x = int((coord[0] + offsetx) * scalemm2px)
        y = int((-coord[1] + offsety) * scalemm2px)  # y axis is flipped
        z = int(coord[2] * scalemm2px)  # z is not used
        static = coord[3]

        # xy modifications from trackbar controls
        x = int(x * xy_trackbar_scale)
        y = int(y * xy_trackbar_scale)
        x += slider_xoffset
        y += slider_yoffset
        if static:
            cv2.circle(frame, (x, y), 4, washout(color), -1)
        else:
            cv2.circle(frame, (x, y), 4, color, -1)


def draw_clustered_points(processed_centroids, color=RED):
    for cluster in processed_centroids:
        x = int((int(cluster['x'] + offsetx) * scalemm2px))
        y = int((int(-cluster['y'] + offsety) * scalemm2px))  # y axis is flipped
        # z = int(coord[2] * scalemm2px)  # z is not used
        # static = coord[3]

        # xy modifications from trackbar controls
        x = int(x * xy_trackbar_scale)
        y = int(y * xy_trackbar_scale)
        x += slider_xoffset
        y += slider_yoffset
        cv2.circle(frame, (x, y), 10, color, -1)


def draw_bbox(centroids, cluster_point_cloud):
    for i in enumerate(centroids):
        x1, y1, x2, y2 = cluster_bbox(cluster_point_cloud, i[0])
        # convert mm to px 
        x1, y1, x2, y2 = int(x1 + offsetx) * scalemm2px, int(-y1 + offsety) * scalemm2px, int(
            x2 + offsetx) * scalemm2px, int(-y2 + offsety) * scalemm2px
        # modify based on trackbar
        x1, y1, x2, y2 = int(x1 * xy_trackbar_scale) + slider_xoffset, int(
            y1 * xy_trackbar_scale) + slider_yoffset, int(x2 * xy_trackbar_scale) + slider_xoffset, int(
            y2 * xy_trackbar_scale) + slider_yoffset
        object_size, object_height = obj_height(cluster_point_cloud, i[0])
        rect = cv2.rectangle(frame, (x1, y1), (x2, y2), ORANGE, 1)
        size, _ = cv2.getTextSize(f"{object_height:.1f} mm", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_width, text_height = size
        cv2.putText(rect, f"{object_height:.1f} mm", (x1, y1 - text_height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ORANGE,
                    2)


def display_frame_info(radar_frame: RadarData, width, height):
    """Display video info on frame. width and height are the dimensions of the window."""
    # Time remaining
    cv2.putText(frame,
                f"{0 if not radar_data.ts else (radar_data.ts[-1] - radar_data.ts[0]) / 1000:.2f} s remaining",
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
                )
    # Number of points in frame
    cv2.putText(frame,
                f"nPoints (frame): {len(radar_frame.x)}",
                (10, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
                )
    # Number of points in gate
    cv2.putText(
        frame,
        f"Points in gate -- s1:{len(s1_display_points)} s2: {len(s2_display_points)}",
        (10, height - 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
    )
    # Video config info, time elapsed, total time of data
    cv2.putText(
        frame,
        f"Replay 1.0x, {config['playback_fps']} fps Time Elapsed (s): {radar_data._RadarData__time_elapsed / 1000:.2f} / {TOTAL_DATA_S:.2f}",
        (10, height - 100),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
    )
    # Legend: green: s1, yellow: s2, orange: bbox, washed: static. With colour coded text, top left
    cv2.putText(
        frame,
        "Legend: ",
        (0, 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
    )
    cv2.putText(
        frame,
        "s1",
        (0, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2
    )
    cv2.putText(
        frame,
        "s2",
        (0, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 2
    )
    cv2.putText(
        frame,
        "Bounding box",
        (0, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 2
    )
    cv2.putText(
        frame,
        "Static",
        (0, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, washout(GREEN), 2
    )


def display_control_info():
    cv2.putText(
        frame,
        "Controls - 'q': quit  'p': pause",
        (width - 175, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 0, 150), 1
    )
    cv2.putText(
        frame,
        "scale/offset gate region with trackbar",
        (width - 217, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 0, 150), 1
    )


# ------------------ VISUALIZATION ------------------ #

# video frame buffer
video_file = config["Files"]["video_file"]
cap = cv2.VideoCapture(video_file)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Number of frames: {num_frames}")

cv2.namedWindow("Radar Visualization")
cv2.createTrackbar(
    "x offset", "Radar Visualization", slider_xoffset, 600, x_trackbar_callback
)
cv2.createTrackbar(
    "y offset", "Radar Visualization", slider_yoffset, 600, y_trackbar_callback
)
cv2.createTrackbar(
    "scale %", "Radar Visualization", int(xy_trackbar_scale * 100), 200, scale_callback
)  # *100 and /100 to account for floating point usuability to downscale

# static points buffer
s1_static = StaticPoints(cnt_thres=5)
s2_static = StaticPoints(cnt_thres=5)

# previous frame buffer
s1_display_points_prev = []
s2_display_points_prev = []

# frame interval, set to the same as video
incr = 1000 / config["playback_fps"]  # frame ts increment, in ms

# radar camera synchronization
rad_cam_offset = rad_cam_offset - rad_cam_offset % (
    incr
)  # make sure it's multiples of video frame interval
print(f"Radar is set to be ahead of video by {rad_cam_offset:.1f}ms.")

# Prepare for main loop: remove radar points, if radar is ahead
all_increments = 0
ts_start = radar_data.ts[0]  # initial timestamp of radar points at start of program
if round(rad_cam_offset) > 0:
    print("rad_cam_offset is set positive, removing radar points while waiting for video.")
while round(rad_cam_offset) > 0:
    all_increments += incr
    while radar_data.ts[0] < ts_start + all_increments:
        # print(f"Point being removed at timestamp {radar_data.ts[0]}")
        radar_data.sid.pop(0)
        radar_data.x.pop(0)
        radar_data.y.pop(0)
        radar_data.z.pop(0)
        radar_data.ts.pop(0)
    rad_cam_offset -= incr
    # print(f"rad_cam_offset is now: {0 if rad_cam_offset < 1 else rad_cam_offset}")
    t_rad = radar_data.ts[0]  # timestamp of the first point in frame

# Prepare for main loop: skip video frames, if video is ahead
if round(rad_cam_offset) < 0:
    print("rad_cam_offset is set negative, waiting radar points while playing video.")
while round(rad_cam_offset) < 0:
    rad_cam_offset += incr
    ret, frame = cap.read()
    if not ret:
        break

# main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (round(width), round(height)))  # reduce frame size
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    height, width = frame.shape[:2]

    s = ''

    # YOLO Inference
    yolo_results = model.predict(source=frame, conf=0.3, iou=0.5, classes=0, show=False)
    # Track Yolo Detections
    yolo_track_start = time.time()
    yolo_tracks = tracking_tools.ss_tracking(frame, yolo_results, seen, strongsort_list, outputs, names, colors, s,
                                             display=False)
    print(yolo_tracks)
    yolo_track_end = time.time()
    print('Yolo Track Time: ' + str(yolo_track_end - yolo_track_start) + 's')
    # draw gate area and get gate area coordinates
    gate_tl, gate_br = draw_gate_topleft()

    # take points in current RADAR frame
    radar_frame = radar_data.take_next_frame(interval=incr)

    # update static points, prepare for display
    radar_process_1 = time.time()
    s1_display_points = []
    s2_display_points = []
    if not radar_frame.is_empty(target_sensor_id=1):
        s1_static.update(radar_frame.get_xyz_coord(sensor_id=1))
        radar_frame.set_static_points(s1_static.get_static_points())
        s1_display_points = radar_frame.get_points_for_display(sensor_id=1)

    if not radar_frame.is_empty(target_sensor_id=2):
        s2_static.update(radar_frame.get_xyz_coord(sensor_id=2))
        radar_frame.set_static_points(s2_static.get_static_points())
        s2_display_points = radar_frame.get_points_for_display(sensor_id=2)

    # remove points that are out of gate area, if configured
    if config["remove_noise"]:
        s1_display_points = remove_points_outside_gate(s1_display_points, gate_tl, gate_br)
        s2_display_points = remove_points_outside_gate(s2_display_points, gate_tl, gate_br)

    # retain previous frame if no new points
    if not s1_display_points:
        s1_display_points = s1_display_points_prev
    else:
        s1_display_points_prev = s1_display_points
    if not s2_display_points:
        s2_display_points = s2_display_points_prev
    else:
        s2_display_points_prev = s2_display_points

    radar_process_2 = time.time()

    print("Radar Pre-Processing Time: " + str(radar_process_2 - radar_process_1) + "s")

    # get all non-static radar points and cluster

    s1_s2_combined = [values[:-1] for values in s1_display_points + s2_display_points if values[-1] == 0]
    if len(s1_s2_combined) > 1:
        cluster_time_1 = time.time()
        processor = ClusterProcessor(eps=350, min_samples=4)  # default: eps=400, min_samples=5 --> eps is in mm
        centroids, cluster_point_cloud = processor.cluster_points(s1_s2_combined)  # get the centroids of each
        cluster_time_2 = time.time()
        # cluster and their associated point cloud
        draw_clustered_points(centroids)  # may not be in the abs center of bbox --> "center of mass", not area
        # centroid.
        draw_clustered_points(cluster_point_cloud, color=BLUE)  # highlight the points that belong to the detected
        # obj
        draw_bbox(centroids, cluster_point_cloud)  # draw the bounding box of each cluster
        print("Cluster Processing Time: " + str(cluster_time_2 - cluster_time_1) + "s")
        r_bboxes = centroid2xywh(centroids, cluster_point_cloud, offsetx, offsety, scalemm2px, slider_xoffset,
                                 slider_yoffset, xy_trackbar_scale)

    else:
        centroids = []
        r_bboxes = []
        print('No Objects Detected by Radar')

    # Track Clustered Radar Objects - SS++
    radar_track_start = time.time()
    rad_tracks = tracking_tools.ss_tracking(frame, r_bboxes, seen, strongsort_list, outputs, names, colors, s,
                                            method='radar', display=False)
    radar_track_end = time.time()
    print(rad_tracks)
    print('Radar Track Time: ' + str(radar_track_end - radar_track_start) + 's')


    # Merging Points
    matched_tracks = tracking_tools.match_tracks(yolo_tracks, rad_tracks)
    if len(matched_tracks) > 0:
        matched_bboxes = [torch.Tensor([pair[:4]]) for pair in matched_tracks[0]]
        matched_xyxy = []
        # Convert from xyxy2xywh
        for det in matched_bboxes:
            det_xyxy = tracking_tools.xyxy2xywh(det)
            matched_xyxy.append(det_xyxy)
        # Track Fused Points
        fusion_track_start = time.time()
        fusion_tracks = tracking_tools.ss_tracking(frame, matched_xyxy, seen, strongsort_list, outputs, names, colors,
                                                   s,
                                                   method='fusion', display=True)
        fusion_track_end = time.time()
        print('Fusion Track Time: ' + str(fusion_track_end - fusion_track_start) + 's')

    # draw points on frame
    if s1_display_points:
        draw_radar_points(s1_display_points, sensor_id=1)
    if s2_display_points:
        draw_radar_points(s2_display_points, sensor_id=2)

    display_frame_info(radar_frame, width, height)
    display_control_info()

    # after drawing points on frames, imshow the frames
    cv2.imshow("Radar Visualization", frame)

    # Key controls
    key = cv2.waitKey(wait_ms) & 0xFF
    if key == ord("q"):  # quit program if 'q' is pressed
        break
    elif key == ord("p"):  # pause/unpause program if 'p' is pressed
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
total_frames = 60



# ------------------ SAVE CONFIG ------------------ #
def yaml_update():
    while True:
        choice = input("Update final trackbar values in yaml? (y/n): ").lower()
        if choice == "y":
            config["TrackbarDefaults"]["slider_xoffset"] = slider_xoffset
            config["TrackbarDefaults"]["slider_yoffset"] = slider_yoffset
            config["TrackbarDefaults"]["xy_trackbar_scale"] = xy_trackbar_scale
            with open("config.yaml", "w") as file:
                yaml.dump(config, file)
            print("Trackbar values updated in yaml.")
            break
        elif choice == "n":
            print("Trackbar values not updated in yaml.")
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

# yaml_update()
