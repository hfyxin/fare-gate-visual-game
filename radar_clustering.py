import torch
from sklearn.cluster import DBSCAN
import numpy as np
from utils.general import xyxy2xywh, xywh2xyxy


def cluster_bbox(cluster_point_cloud, cluster_label):
    x_values = [point['x'] for point in cluster_point_cloud if point['index'] == cluster_label]
    y_values = [point['y'] for point in cluster_point_cloud if point['index'] == cluster_label]
    x1 = min(x_values)
    y1 = min(y_values)
    x2 = max(x_values)
    y2 = max(y_values)
    return x1, y1, x2, y2


def obj_height(cluster_point_cloud, cluster_label):
    z_values = [point['z'] for point in cluster_point_cloud if point['index'] == cluster_label]
    # height/size
    z1 = min(z_values)
    z2 = max(z_values)
    size = z2 - z1
    height = z2
    return size, height


def centroid2xywh(centroids, cluster_point_cloud, offsetx, offsety, scalemm2px, slider_xoffset, slider_yoffset,
                  xy_trackbar_scale):
    r_bboxes = []
    for i in enumerate(centroids):
        x1, y1, x2, y2 = cluster_bbox(cluster_point_cloud, i[0])
        # convert mm to px
        x1, y1, x2, y2 = int(x1 + offsetx) * scalemm2px, int(-y1 + offsety) * scalemm2px, int(
            x2 + offsetx) * scalemm2px, int(-y2 + offsety) * scalemm2px
        # modify based on trackbar
        x1, y1, x2, y2 = int(x1 * xy_trackbar_scale) + slider_xoffset, int(
            y1 * xy_trackbar_scale) + slider_yoffset, int(x2 * xy_trackbar_scale) + slider_xoffset, int(
            y2 * xy_trackbar_scale) + slider_yoffset

        bboxes = [[x1, y1, x2, y2]]
        tensor_bbox = torch.Tensor(bboxes)
        print('TENSOR CORRECT FORMAT FOR XYXY2XYWH')
        print(tensor_bbox)
        xywh_bbox = xyxy2xywh(tensor_bbox)
        r_bboxes.append(torch.Tensor(xywh_bbox))

    return r_bboxes


class ClusterProcessor:
    def __init__(self, eps=400, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def cluster_points(self, radar_points):
        '''
        :param radar_points: Any
        :return: centroids -- {'x': Xc, 'y': Yc, 'z': Zc, 'numPoints': n}
                 cluster_point_cloud -- -- {'x': Xn, 'y': Yn, 'z': Zn, 'Cluster Label': k}
        '''
        cluster_obj = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="euclidean", n_jobs=-1).fit(
            radar_points)  # eps is trial and error val  --> 400 will probably be ok for denser pointcloud
        clusters = set(cluster_obj.labels_)  # Set of labels (no repeats)
        pointLabels = cluster_obj.labels_  # List of labels by points (with repeated labels)

        # Initialize lists for centroid, clustered points and cluster idxs
        centroids = []
        cluster_point_cloud = []

        # Iterate thru clusters
        for clusterIdx, cluster in enumerate(clusters):  # No repeats in a cluster
            if cluster >= 0:  # Discard the -1 cluster which is the unassociated points
                centroids.append({'x': 0, 'y': 0, 'z': 0, 'numPoints': 0})
                # Go thru points/labels
                for labelIdx, label in enumerate(pointLabels):
                    if label == cluster:
                        # Accumulate coords values for centroid calc
                        centroids[clusterIdx]['x'] = centroids[clusterIdx]['x'] + radar_points[labelIdx][0]  # X
                        centroids[clusterIdx]['y'] = centroids[clusterIdx]['y'] + radar_points[labelIdx][1]  # Y
                        centroids[clusterIdx]['z'] = centroids[clusterIdx]['z'] + radar_points[labelIdx][2]  # Z
                        centroids[clusterIdx]['numPoints'] = centroids[clusterIdx][
                                                                 'numPoints'] + 1  # store the count to divide later
                        # Accumulate each radar point associated with cluster label
                        cluster_point_cloud.append({'x': radar_points[labelIdx][0], 'y': radar_points[labelIdx][1],
                                                    'z': radar_points[labelIdx][2], 'index': cluster})

                # Compute the centroid of the cluster, store the number of points
                centroids[clusterIdx]['x'] = centroids[clusterIdx]['x'] / centroids[clusterIdx]['numPoints']
                centroids[clusterIdx]['y'] = centroids[clusterIdx]['y'] / centroids[clusterIdx]['numPoints']
                centroids[clusterIdx]['z'] = centroids[clusterIdx]['z'] / centroids[clusterIdx]['numPoints']

        return centroids, cluster_point_cloud
