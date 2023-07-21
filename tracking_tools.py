import cv2
from pathlib import Path

import numpy

from utils.general import scale_coords, xyxy2xywh
from utils.torch_utils import time_synchronized
from utils.plots import plot_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from scipy.optimize import linear_sum_assignment

VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video


def ss_tracking(frame, detections, seen, strongsort_list, outputs, names, colors, s, method='yolo', display=False):
    for i, det in enumerate(detections):  # Detections per image
        seen += 1
        if det is not None and len(det) and method == "yolo":
            # Print Results
            class_labels = det.boxes.cls.cpu().numpy().astype(int)
            unique_classes = np.unique(class_labels)
            for c in unique_classes:
                class_name = names[c]
                n = (class_labels == c).sum()
                s += f"{n} {class_name}{'s' * (n > 1)} "

            # Extract Object Data
            boxes = det.boxes
            xywhs = boxes.xywh
            confs = boxes.conf
            clss = boxes.cls
            # Pass detections to strongsort
            outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame)

            if len(outputs[i]) > 0 and display is True:
                for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                    # Extract track info from yolo/ss++ output
                    bboxes = output[0:4]
                    id = int(output[4])
                    cls = output[5]

                    c = int(cls)  # integer class
                    id = int(id)  # integer id

                    label = f'{id} {names[c]} {conf:.2f}'
                    plot_one_box(bboxes, frame, label=label, color=colors[int(cls)], line_thickness=2)
                    print(s)
            return outputs[i]

        elif method == "radar":
            # Get Radar Data
            xywhs = det
            confs = torch.tensor(np.ones(len(det)) * 0.9)  # Assume all detections are 90%
            clss = torch.tensor(np.ones(
                len(det)))  # set to any class other than 0 so tht its a diff color (for now, will need to update)
            # Pass detections to strongsort
            outputs[i + 1] = strongsort_list[i + 1].update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame, strong=False,
                                                           radar=True)

            if len(outputs[i + 1]) > 0 and display is True:
                for j, (output, conf) in enumerate(zip(outputs[i + 1], confs)):
                    # Extract track info from radar/ss++ output
                    bboxes = output[0:4]
                    id = int(output[4])
                    cls = output[5]

                    c = 'RADAR'
                    id = int(id)  # integer id

                    label = f'{id} {c} {conf:.2f}'
                    plot_one_box(bboxes, frame, label=label, color=colors[int(cls)], line_thickness=2)
                    print(s)

            return outputs[i + 1]

        elif method == "fusion":
            # Get Radar Data
            xywhs = det
            confs = torch.tensor(
                np.ones(len(det)) * 0.9)  # Assume all detections are 90% --> Needs updating to get fused conf
            clss = torch.tensor(np.ones(len(det) * 5))  # need to update to get the fused class
            # Pass detections to strongsort
            outputs[i + 2] = strongsort_list[i + 2].update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame, strong=False)

            if len(outputs[i + 2]) > 0 and display is True:
                for j, (output, conf) in enumerate(zip(outputs[i + 2], confs)):
                    # Extract track info from radar/ss++ output
                    bboxes = output[0:4]
                    print('FUSION BBOX FORMAT')
                    print(bboxes)
                    id = int(output[4])
                    cls = output[5]

                    c = 'FUSION'
                    id = int(id)  # integer id

                    label = f'{id} {c} {conf:.2f}'
                    plot_one_box(bboxes, frame, label=label, color=colors[int(cls)], line_thickness=2)
                    print(s)

            return outputs[i + 2]

        else:
            if method == "yolo":
                strongsort_list[i].increment_ages()
            elif method == 'radar':
                strongsort_list[i + 1].increment_ages()
            else:
                strongsort_list[i + 2].increment_ages()
            print('No detections')

# TODO for match_tracks function: Make sure yolo/yolo and radar/radar matches
#  do not occur so we dont lose data

def match_tracks(yolo_tracks, rad_tracks):
    # Find the lengths of both lists
    rad_tracks = rad_tracks if rad_tracks is not None else []
    yolo_tracks = yolo_tracks if yolo_tracks is not None else []

    yolo_len = len(yolo_tracks) if yolo_tracks is not None else 0
    rad_len = len(rad_tracks) if rad_tracks is not None else 0

    if yolo_len != 0 or rad_len != 0:
        # Add dummy tracks to the shorter list to make lengths equal
        if yolo_len > rad_len:
            print('RADAR EMPTY')
            dummy_tracks = np.full((yolo_len - rad_len, 9), -1)
            rad_tracks_copy = np.vstack((rad_tracks, dummy_tracks)) if rad_len > 0 else dummy_tracks
            yolo_tracks_copy = yolo_tracks
        elif rad_len > yolo_len:
            print('YOLO EMPTY')
            dummy_tracks = np.full((rad_len - yolo_len, 9), -1)
            yolo_tracks_copy = np.vstack((yolo_tracks, dummy_tracks)) if yolo_len > 0 else dummy_tracks
            rad_tracks_copy = rad_tracks
        else:
            yolo_tracks_copy = yolo_tracks
            rad_tracks_copy = rad_tracks

        # Extract the bounding box coordinates from the modified lists
        yolo_boxes = np.array([track[:4] for track in yolo_tracks_copy])
        rad_boxes = np.array([track[:4] for track in rad_tracks_copy])

        # Check if both yolo_boxes and rad_boxes are not empty
        if yolo_boxes.shape[0] > 0 and rad_boxes.shape[0] > 0:
            # Calculate the pairwise distances between yolo_boxes and rad_boxes
            distances = np.linalg.norm(yolo_boxes[:, None] - rad_boxes[None], axis=-1)

            # Use the Hungarian algorithm to find the optimal assignment of pairs
            yolo_indices, rad_indices = linear_sum_assignment(distances)

            # Initialize a set to keep track of used indices
            used_indices = set()

            # Initialize a list to store the matched pairs and unmatched real points
            matched_and_unmatched = []

            # Iterate through the matched indices and form pairs
            for yolo_idx, rad_idx in zip(yolo_indices, rad_indices):
                # Check if the pair is not a dummy track
                if yolo_tracks_copy[yolo_idx][0] != -1 and rad_tracks_copy[rad_idx][0] != -1:
                    # Get the matched pair's coordinates
                    yolo_coords = yolo_tracks_copy[yolo_idx][:4]
                    rad_coords = rad_tracks_copy[rad_idx][:4]

                    # Calculate new coordinates (average) for matched tracks
                    new_coords = (yolo_coords + rad_coords) / 2.0

                    # Update the matched tracks' coordinates
                    yolo_tracks_copy[yolo_idx][:4] = new_coords
                    rad_tracks_copy[rad_idx][:4] = new_coords

                    # Append the updated tracks to the matched list
                    matched_pair = [yolo_tracks_copy[yolo_idx], rad_tracks_copy[rad_idx]]
                    matched_and_unmatched.append(matched_pair)
                else:
                    # If one of the tracks is a dummy track, store the real point in the list
                    if yolo_tracks_copy[yolo_idx][0] != -1:
                        unmatched_track = yolo_tracks_copy[yolo_idx]
                    elif rad_tracks_copy[rad_idx][0] != -1:
                        unmatched_track = rad_tracks_copy[rad_idx]
                    matched_and_unmatched.append([unmatched_track])
            return matched_and_unmatched
    else:
        # If both yolo_tracks AND rad_tracks are empty, return an empty list
        return []
