from open3d import linux as open3d
from os.path import join
import numpy as np
import colorsys, random, os, sys
import pandas as pd
import open3d


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors


class ConfigSemanticKITTI:
    k_n = 16  # KNN
    num_layers = 4  # Number of layers
    num_points = 4096 * 11  # Number of input points
    num_classes = 19  # Number of valid classes
    sub_grid_size = 0.06  # preprocess_parameter

    batch_size = 6  # batch_size during training
    val_batch_size = 20  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256]  # feature dimension
    num_sub_points = [num_points // 4, num_points // 16, num_points // 64, num_points // 256]

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None


class ConfigS3DIS:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 40960  # Number of input points
    num_classes = 13  # Number of valid classes
    sub_grid_size = 0.04  # preprocess_parameter

    batch_size = 6  # batch_size during training
    val_batch_size = 20  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None


class ConfigDales:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 65536  # Number of input points
    num_classes = 8  # Number of valid classes
    sub_grid_size = 0.06  # preprocess_parameter

    batch_size = 4  # batch_size during training
    val_batch_size = 16  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None

    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_occlusion = 'none'
    augment_color = 0.0


import numpy as np
import os
import pandas as pd
from os.path import join
import random

class DataProcessing:
    @staticmethod
    def load_pc_semantic3d(filename):
        """
        Loads the Semantic3D point cloud and labels.
        :param filename: Path to the Semantic3D dataset file.
        :return: points (xyz + intensity), labels
        """
        pc_pd = pd.read_csv(filename, header=None, delim_whitespace=True)
        pc_np = pc_pd.values
        # Return XYZ + intensity as input, sem_class separately
        points = pc_np[:, :4]        # x y z intensity
        labels = pc_np[:, 4].astype(np.uint8)
        return points[:, :3], labels  # Only return XYZ, drop intensity

    @staticmethod
    def load_label_semantic3d(filename):
        """
        Loads the Semantic3D labels.
        :param filename: Path to the Semantic3D label file.
        :return: cloud_labels (semantic labels)
        """
        label_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.uint8)
        cloud_labels = label_pd.values
        return cloud_labels

    @staticmethod
    def load_pc_kitti(pc_path):
        """
        Loads KITTI point cloud data.
        :param pc_path: Path to the KITTI point cloud file.
        :return: points (xyz)
        """
        scan = np.fromfile(pc_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))  # Assuming 4 columns (x, y, z, intensity)
        points = scan[:, 0:3]  # get xyz only, ignore intensity
        return points

    @staticmethod
    def load_label_kitti(label_path, remap_lut):
        """
        Loads KITTI labels and applies remapping.
        :param label_path: Path to KITTI label file.
        :param remap_lut: Remapping lookup table for semantic labels.
        :return: remapped semantic labels
        """
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1,))
        sem_label = label & 0xFFFF  # Extract the semantic label (lower 16 bits)
        inst_label = label >> 16  # Extract the instance id (upper 16 bits)
        assert ((sem_label + (inst_label << 16) == label).all())
        sem_label = remap_lut[sem_label]
        return sem_label.astype(np.int32)

    @staticmethod
    def get_file_list(dataset_path, test_scan_num):
        """
        Retrieves the list of files for train, validation, and test sets.
        :param dataset_path: Path to the dataset directory.
        :param test_scan_num: The scan number to be used for the test set.
        :return: train_file_list, val_file_list, test_file_list
        """
        seq_list = np.sort(os.listdir(dataset_path))

        train_file_list = []
        test_file_list = []
        val_file_list = []
        for seq_id in seq_list:
            seq_path = join(dataset_path, seq_id)
            pc_path = join(seq_path, 'velodyne')
            if seq_id == '08':
                val_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
                if seq_id == test_scan_num:
                    test_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
            elif int(seq_id) >= 11 and seq_id == test_scan_num:
                test_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
            elif seq_id in ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']:
                train_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])

        train_file_list = np.concatenate(train_file_list, axis=0)
        val_file_list = np.concatenate(val_file_list, axis=0)
        test_file_list = np.concatenate(test_file_list, axis=0)
        return train_file_list, val_file_list, test_file_list

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """
        Perform k-NN search.
        :param support_pts: Points to search from.
        :param query_pts: Points to search for neighbors.
        :param k: Number of neighbors.
        :return: Indices of the nearest neighbors.
        """
        neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
        return neighbor_idx.astype(np.int32)

    @staticmethod
    def data_aug(xyz, labels, idx, num_out):
        """
        Data augmentation by duplicating points.
        :param xyz: Points data (N, 3)
        :param labels: Labels for the points (N,)
        :param idx: Indices of the points.
        :param num_out: Target number of points after augmentation.
        :return: Augmented points, indices, and labels
        """
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):
        """
        Shuffle the indices of a given dataset.
        :param x: Dataset to shuffle.
        :return: Shuffled dataset.
        """
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_list(data_list):
        """
        Shuffle a list of data.
        :param data_list: List of data to shuffle.
        :return: Shuffled list.
        """
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list

    @staticmethod
    def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        """
        Grid-based subsampling of point cloud data.
        :param points: Point cloud data (N, 3)
        :param features: Optional features (N, d).
        :param labels: Optional labels (N,)
        :param grid_size: Size of grid voxels.
        :param verbose: If 1, verbose output.
        :return: Subsampled points, with features and/or labels depending on the input.
        """
        if (features is None) and (labels is None):
            return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
        elif labels is None:
            return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
        elif features is None:
            return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
        else:
            return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size, verbose=verbose)

    @staticmethod
    def IoU_from_confusions(confusions):
        """
        Compute Intersection over Union (IoU) from confusion matrices.
        :param confusions: Confusion matrices.
        :return: IoU scores.
        """
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)

        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        IoU += mask * mIoU
        return IoU

    @staticmethod
    def get_class_weights(dataset_name):
        num_per_class = []
        if dataset_name == 'S3DIS':
            num_per_class = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                                    650464, 791496, 88727, 1284130, 229758, 2272837], dtype=np.int32)
        elif dataset_name == 'Semantic3D':
            num_per_class = np.array([5181602, 5012952, 6830086, 1311528, 10476365, 946982, 334860, 269353],
                                    dtype=np.int32)
        elif dataset_name == 'SemanticKITTI':
            num_per_class = np.array([55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                                    240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                                    9833174, 129609852, 4506626, 1168181])
        elif dataset_name == 'DALES':
            # DALES class counts you provided
            num_per_class = np.array([0, 104967287, 71308488, 1288806, 587638, 573361, 742902, 144580], dtype=np.int32)

        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        return np.expand_dims(ce_label_weight, axis=0)







class Plot:
    @staticmethod
    def random_colors(N, bright=True, seed=0):
        """
        Generate random colors for N labels using HSV to RGB conversion.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc(pc_xyz):
        """
        Draw point cloud with XYZ coordinates only. 
        Assumes no RGB values are provided.
        """
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(pc_xyz)

        open3d.visualization.draw_geometries([pc])
        return 0

    @staticmethod
    def draw_pc_sem_ins(pc_xyz, pc_sem_ins, plot_colors=None):
        """
        Draw point cloud with semantic or instance labels. 
        Generates random colors for each unique label.
        
        Parameters:
        - pc_xyz: 3D coordinates of the point cloud (Nx3 array).
        - pc_sem_ins: Semantic or instance labels (Nx1 array).
        - plot_colors: Custom color list for labels (optional).
        
        Returns:
        - Y_semins: Point cloud with assigned colors for labels.
        """
        if plot_colors is not None:
            ins_colors = plot_colors
        else:
            ins_colors = Plot.random_colors(len(np.unique(pc_sem_ins)) + 1, seed=2)

        sem_ins_labels = np.unique(pc_sem_ins)
        Y_colors = np.zeros((pc_sem_ins.shape[0], 3))

        for id, semins in enumerate(sem_ins_labels):
            valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]
            if semins <= -1:
                tp = [0, 0, 0]  # For invalid or background label
            else:
                tp = ins_colors[semins] if plot_colors is not None else ins_colors[id]

            Y_colors[valid_ind] = tp

        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        Plot.draw_pc(Y_semins)
        return Y_semins

