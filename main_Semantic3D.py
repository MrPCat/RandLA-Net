from os.path import join, exists
from RandLANet import Network
from tester_Semantic3D import ModelTester
from helper_ply import read_ply
from helper_tool import Plot
from helper_tool import DataProcessing as DP
from helper_tool import ConfigDales as cfg
import tensorflow as tf
import numpy as np
import pickle, argparse, os


class Dales:
    def __init__(self):
        self.name = 'Dales'
        
        # Define paths to your pre-split data folders - update these with your actual paths
        self.path = '/content/drive/MyDrive/Dales_PLY_Format'  # Base path
        self.train_data_path = join(self.path, 'Train')
        self.val_data_path = join(self.path, 'Valid')
        self.test_data_path = join(self.path, 'Test')
        
        # Make sure these folders exist
        os.makedirs(self.train_data_path, exist_ok=True)
        os.makedirs(self.val_data_path, exist_ok=True)
        os.makedirs(self.test_data_path, exist_ok=True)
        
        # Make sure the label mappings match your Dales dataset
        self.label_to_names = {
            1: 'Ground',
            2: 'Vegetation',
            3: 'Cars',
            4: 'Trucks',
            5: 'Power lines',
            6: 'Poles',
            7: 'Fences',
            8: 'Buildings'
        }
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([0])  # Set 'unlabeled' as ignored

        # Create directories for processed data
        self.sub_pc_folder = join(self.path, f'input_{cfg.sub_grid_size:.3f}')
        os.makedirs(self.sub_pc_folder, exist_ok=True)

        # Keep these for compatibility with the rest of the code
        self.original_folder = join(self.path, 'original_data')
        self.full_pc_folder = join(self.path, 'original_ply')
        self.sub_pc_folder = join(self.path, 'input_{:.3f}'.format(cfg.sub_grid_size))

        # Initialize empty file lists
        self.train_files = []
        self.val_files = []
        self.test_files = []
        
        # Collect training files
        for file_name in os.listdir(self.train_data_path):
            if file_name.endswith('.ply'):
                self.train_files.append(join(self.train_data_path, file_name))
        
        # Collect validation files
        for file_name in os.listdir(self.val_data_path):
            if file_name.endswith('.ply'):
                self.val_files.append(join(self.val_data_path, file_name))
        
        # Collect test files
        for file_name in os.listdir(self.test_data_path):
            if file_name.endswith('.ply'):
                self.test_files.append(join(self.test_data_path, file_name))
        
        # Sort the file lists
        self.train_files = np.sort(self.train_files)
        self.val_files = np.sort(self.val_files)
        self.test_files = np.sort(self.test_files)

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.test_proj = []
        self.test_labels = []

        self.possibility = {}
        self.min_possibility = {}
        self.class_weight = {}
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_colors = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': []}

        # Ascii files dict for testing
        self.ascii_files = {
            'MarketplaceFeldkirch_Station4_rgb_intensity-reduced.ply': 'marketsquarefeldkirch4-reduced.labels',
            'sg27_station10_rgb_intensity-reduced.ply': 'sg27_10-reduced.labels',
            'sg28_Station2_rgb_intensity-reduced.ply': 'sg28_2-reduced.labels',
            'StGallenCathedral_station6_rgb_intensity-reduced.ply': 'stgallencathedral6-reduced.labels',
            'birdfountain_station1_xyz_intensity_rgb.ply': 'birdfountain1.labels',
            'castleblatten_station1_intensity_rgb.ply': 'castleblatten1.labels',
            'castleblatten_station5_xyz_intensity_rgb.ply': 'castleblatten5.labels',
            'marketplacefeldkirch_station1_intensity_rgb.ply': 'marketsquarefeldkirch1.labels',
            'marketplacefeldkirch_station4_intensity_rgb.ply': 'marketsquarefeldkirch4.labels',
            'marketplacefeldkirch_station7_intensity_rgb.ply': 'marketsquarefeldkirch7.labels',
            'sg27_station10_intensity_rgb.ply': 'sg27_10.labels',
            'sg27_station3_intensity_rgb.ply': 'sg27_3.labels',
            'sg27_station6_intensity_rgb.ply': 'sg27_6.labels',
            'sg27_station8_intensity_rgb.ply': 'sg27_8.labels',
            'sg28_station2_intensity_rgb.ply': 'sg28_2.labels',
            'sg28_station5_xyz_intensity_rgb.ply': 'sg28_5.labels',
            'stgallencathedral_station1_intensity_rgb.ply': 'stgallencathedral1.labels',
            'stgallencathedral_station3_intensity_rgb.ply': 'stgallencathedral3.labels',
            'stgallencathedral_station6_intensity_rgb.ply': 'stgallencathedral6.labels'}

        self.load_sub_sampled_clouds(cfg.sub_grid_size)                         

    def load_sub_sampled_clouds(self, sub_grid_size):
        files = np.hstack((self.train_files, self.val_files, self.test_files))

        for i, file_path in enumerate(files):
            cloud_name = file_path.split('/')[-1][:-4]
            print('Load_pc_' + str(i) + ': ' + cloud_name)
            
            # Determine which split this file belongs to
            if file_path in self.val_files:
                cloud_split = 'validation'
                # Get directory for this split
                split_dir = self.val_data_path
            elif file_path in self.train_files:
                cloud_split = 'training'
                split_dir = self.train_data_path
            else:
                cloud_split = 'test'
                split_dir = self.test_data_path

            # Update paths to look in the appropriate split directory
            kd_tree_file = join(split_dir, '{:s}_KDTree.pkl'.format(cloud_name))
            
            # Use the actual PLY file path directly
            sub_ply_file = file_path

            # Read ply with data
            data = read_ply(sub_ply_file)
            # Here, we only use x, y, z, and intensity (no RGB)
            sub_points = np.vstack((data['x'], data['y'], data['z'])).T
            
            # Check if 'intensity' field exists, otherwise use a default value
            if 'intensity' in data.dtype.names:
                sub_intensity = data['intensity']
            else:
                print(f"Warning: 'intensity' not found in {sub_ply_file}, using default values")
                sub_intensity = np.ones(len(data))
            
            if cloud_split == 'test':
                sub_labels = None
            else:
                # Check if 'class' field exists
                if 'class' in data.dtype.names:
                    sub_labels = data['class']
                else:
                    print(f"Warning: 'class' not found in {sub_ply_file}")
                    sub_labels = np.zeros(len(data))

            # Read pkl with search tree or create one if it doesn't exist
            if exists(kd_tree_file):
                with open(kd_tree_file, 'rb') as f:
                    search_tree = pickle.load(f)
            else:
                print(f"KDTree file not found: {kd_tree_file}, creating new one")
                search_tree = DP.build_kdtree(sub_points)
                # Optionally save the tree for future use
                with open(kd_tree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_intensity]
            if cloud_split in ['training', 'validation']:
                self.input_labels[cloud_split] += [sub_labels]

        # Modified projection handling for validation and test sets
        print('\nPreparing reprojection indices for validation and test')

        for i, file_path in enumerate(files):
            cloud_name = file_path.split('/')[-1][:-4]
            
            # Determine directory based on which split this file belongs to
            if file_path in self.val_files:
                split_dir = self.val_data_path
                proj_file = join(split_dir, '{:s}_proj.pkl'.format(cloud_name))
                
                # Check if projection file exists, create one if not
                if exists(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_idx, labels = pickle.load(f)
                else:
                    print(f"Projection file not found: {proj_file}, creating default projection")
                    # Create a simple identity projection (or implement your own logic)
                    data = read_ply(file_path)
                    if 'class' in data.dtype.names:
                        labels = data['class']
                    else:
                        labels = np.zeros(len(data))
                    proj_idx = np.arange(len(data))
                    # Save for future use
                    with open(proj_file, 'wb') as f:
                        pickle.dump((proj_idx, labels), f)
                        
                self.val_proj += [proj_idx]
                self.val_labels += [labels]

            # Test projection
            elif file_path in self.test_files:
                split_dir = self.test_data_path
                proj_file = join(split_dir, '{:s}_proj.pkl'.format(cloud_name))
                
                # Similar handling for test files
                if exists(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_idx, labels = pickle.load(f)
                else:
                    print(f"Projection file not found: {proj_file}, creating default projection")
                    data = read_ply(file_path)
                    # For test data, we might not have labels
                    labels = np.zeros(len(data))
                    proj_idx = np.arange(len(data))
                    with open(proj_file, 'wb') as f:
                        pickle.dump((proj_idx, labels), f)
                        
                self.test_proj += [proj_idx]
                self.test_labels += [labels]
        
        print('finished')
        return

    # Generate the input data flow
    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size
        elif split == 'test':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size

        # Reset possibility
        self.possibility[split] = []
        self.min_possibility[split] = []
        self.class_weight[split] = []

        # Random initialize
        for i, tree in enumerate(self.input_trees[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        if split != 'test':
            _, num_class_total = np.unique(np.hstack(self.input_labels[split]), return_counts=True)
            self.class_weight[split] += [np.squeeze([num_class_total / np.sum(num_class_total)], axis=0)]

        def spatially_regular_gen():
            # Generator loop
            for i in range(num_per_epoch):  # num_per_epoch

                # Choose the cloud with the lowest probability
                cloud_idx = int(np.argmin(self.min_possibility[split]))

                # choose the point with the minimum of possibility in the cloud as query point
                point_ind = np.argmin(self.possibility[split][cloud_idx])

                # Get all points within the cloud from tree structure
                points = np.array(self.input_trees[split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)
                query_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                # Shuffle index
                query_idx = DP.shuffle_idx(query_idx)

                # Get corresponding points and features based on the index
                queried_pc_xyz = points[query_idx]
                queried_pc_xyz[:, 0:2] = queried_pc_xyz[:, 0:2] - pick_point[:, 0:2]
                queried_pc_features = self.input_colors[split][cloud_idx][query_idx]  # Intensity as features

                if split == 'test':
                    queried_pc_labels = np.zeros(queried_pc_xyz.shape[0])
                    queried_pt_weight = 1
                else:
                    queried_pc_labels = self.input_labels[split][cloud_idx][query_idx]
                    queried_pc_labels = np.array([self.label_to_idx[l] for l in queried_pc_labels])
                    queried_pt_weight = np.array([self.class_weight[split][0][n] for n in queried_pc_labels])

                # Update the possibility of the selected points
                dists = np.sum(np.square((points[query_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists)) * queried_pt_weight
                self.possibility[split][cloud_idx][query_idx] += delta
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

                yield (queried_pc_xyz,
                    queried_pc_features.astype(np.float32),
                    queried_pc_labels,
                    query_idx.astype(np.int32),
                    np.array([cloud_idx], dtype=np.int32))


        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None])
        return gen_func, gen_types, gen_shapes

    def get_tf_mapping(self):
        # Collect flat inputs
        def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
            batch_features = tf.map_fn(self.tf_augment_input, [batch_xyz, batch_features], dtype=tf.float32)
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            for i in range(cfg.num_layers):
                neigh_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                pool_i = neigh_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
                input_points.append(batch_xyz)
                input_neighbors.append(neigh_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points

            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]

            return input_list

        return tf_map

    # data augmentation
    @staticmethod
    def tf_augment_input(inputs):
        xyz = inputs[0]  # (x, y, z)
        features = inputs[1]  # features (including intensity)
        
        # Randomly rotate around z-axis (vertical axis for aerial data)
        theta = tf.random_uniform((1,), minval=0, maxval=2 * np.pi)
        c, s = tf.cos(theta), tf.sin(theta)
        cs0 = tf.zeros_like(c)
        cs1 = tf.ones_like(c)
        
        # Only rotate around z-axis for aerial data
        R = tf.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
        stacked_rots = tf.reshape(R, (3, 3))
        transformed_xyz = tf.reshape(tf.matmul(xyz, stacked_rots), [-1, 3])
        
        # Scale augmentation - adjust min/max based on your dataset
        min_s = cfg.augment_scale_min
        max_s = cfg.augment_scale_max
        if cfg.augment_scale_anisotropic:
            s = tf.random_uniform((1, 3), minval=min_s, maxval=max_s)
        else:
            s = tf.random_uniform((1, 1), minval=min_s, maxval=max_s)
        
        # Apply symmetries if configured
        symmetries = []
        for i in range(3):
            if cfg.augment_symmetries[i]:
                symmetries.append(tf.round(tf.random_uniform((1, 1))) * 2 - 1)
            else:
                symmetries.append(tf.ones([1, 1], dtype=tf.float32))
        s *= tf.concat(symmetries, 1)
        
        # Apply scales
        stacked_scales = tf.tile(s, [tf.shape(transformed_xyz)[0], 1])
        transformed_xyz = transformed_xyz * stacked_scales
        
        # Add random noise
        if cfg.augment_noise > 0:
            noise = tf.random_normal(tf.shape(transformed_xyz), stddev=cfg.augment_noise)
            transformed_xyz = transformed_xyz + noise
        
        # Combine transformed coordinates with features
        if tf.shape(features)[1] > 1:
            # If there are multiple features
            transformed_features = tf.concat([transformed_xyz, features[:, 1:]], axis=1)
        else:
            # If only one feature (intensity)
            transformed_features = tf.concat([transformed_xyz, features], axis=1)
        
        return transformed_features



    def init_input_pipeline(self):
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        gen_function_test, _, _ = self.get_batch_gen('test')
        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)
        self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)

        self.batch_train_data = self.train_data.batch(cfg.batch_size)
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        self.batch_test_data = self.test_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping()

        self.batch_train_data = self.batch_train_data.map(map_func=map_func)
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)
        self.batch_test_data = self.batch_test_data.map(map_func=map_func)

        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)
        self.batch_test_data = self.batch_test_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)
        self.test_init_op = iter.make_initializer(self.batch_test_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--mode', type=str, default='train', help='options: train, test, vis')
    parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
    FLAGS = parser.parse_args()

    GPU_ID = FLAGS.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    Mode = FLAGS.mode
    dataset = Dales()
    dataset.init_input_pipeline()

    if Mode == 'train':
        model = Network(dataset, cfg)
        model.train(dataset)
    elif Mode == 'test':
        cfg.saving = False
        model = Network(dataset, cfg)
        if FLAGS.model_path is not 'None':
            chosen_snap = FLAGS.model_path
        else:
            chosen_snapshot = -1
            logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])
            chosen_folder = logs[-1]
            snap_path = join(chosen_folder, 'snapshots')
            snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
            chosen_step = np.sort(snap_steps)[-1]
            chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
        tester = ModelTester(model, dataset, restore_snap=chosen_snap)
        tester.test(model, dataset)

    else:
        ##################
        # Visualize data #
        ##################

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(dataset.train_init_op)
            while True:
                flat_inputs = sess.run(dataset.flat_inputs)
                pc_xyz = flat_inputs[0]
                sub_pc_xyz = flat_inputs[1]
                labels = flat_inputs[21]
                Plot.draw_pc_sem_ins(pc_xyz[0, :, :], labels[0, :])
                Plot.draw_pc_sem_ins(sub_pc_xyz[0, :, :], labels[0, 0:np.shape(sub_pc_xyz)[1]])
