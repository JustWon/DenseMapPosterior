import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from glob import glob


class DenseMapPosterior:
    def __init__(self, seq_id, intrinsic_matrix):
        self.img_size = (50,400,3)
        self.seq_id = seq_id
        self.intrinsic_matrix = intrinsic_matrix
    
    def projection(self, _points, _A, _Rt, range_limit=30.0):
        projected_img =   np.ones(self.img_size,dtype=float)*np.inf
        agumented_points = np.c_[_points,np.ones(_points.shape[0])]
        transformed_points = np.linalg.inv(_Rt)@np.transpose(agumented_points)
        projected_points = _A@transformed_points[:3,:]
        hnormalized_points = projected_points/projected_points[2,:]

        # transpose
        hnormalized_points = np.transpose(hnormalized_points.astype(int))
        transformed_points = np.transpose(transformed_points)

        # in front of cameras only
        condition1 = transformed_points[:,2] > 0
        hnormalized_points = hnormalized_points[condition1]
        transformed_points = transformed_points[condition1]

        condition_ = np.linalg.norm(transformed_points, axis=1) < range_limit
        hnormalized_points = hnormalized_points[condition_]
        transformed_points = transformed_points[condition_]

        # inside of the image region
        condition2 = ((0 <= hnormalized_points[:,0]) & (hnormalized_points[:,0] < self.img_size[1]) & (0 <= hnormalized_points[:,1]) & (hnormalized_points[:,1] < self.img_size[0]))
        hnormalized_points = hnormalized_points[condition2]
        transformed_points = transformed_points[condition2]

        for i, hnormalized_point in enumerate(hnormalized_points):
            t_y = hnormalized_point[1]
            t_x = hnormalized_point[0]

            # r = np.sqrt((t_x-cx)**2 + (t_y-cy)**2)
            # distorted_x = int(t_x + r/(1 + k1*r**2 + k2*r**4 + k3*r**6))
            # distorted_y = int(t_y + r/(1 + k1*r**2 + k2*r**4 + k3*r**6))

            new_val = transformed_points[i]
            pre_val = projected_img[t_y, t_x]
            if (new_val[2] < pre_val[2]):
                projected_img[t_y, t_x] = transformed_points[i,:3]

        return projected_img

    def GlobalMapProjection(self, global_map, seq_id, frame_idx):
        # global map projection
        Rt = np.eye(4)
        transform = []
        with open("./dump/%s/dump/%06d/data" % (seq_id, frame_idx)) as fp:
            lines = fp.readlines()
            for idx, line in enumerate(lines):
                if ("estimate" in line):
                    transform.append([float(item) for item in lines[idx+1].split()])
                    transform.append([float(item) for item in lines[idx+2].split()])
                    transform.append([float(item) for item in lines[idx+3].split()])
                    transform.append([float(item) for item in lines[idx+4].split()])
                    break


        # Rotation coordinates change
        Rt = np.asarray(transform)
        rot_mat = Rt[:3,:3]
        rot_class = R.from_matrix(rot_mat)
        euler = rot_class.as_euler('xyz')
        rot_class = R.from_euler('xyz', np.array([-euler[1], -euler[2], euler[0]]))
        Rt[:3,:3] = rot_class.as_matrix()

        # Translation coordinates change
        Rt[:3,3] = np.array([-Rt[1,3], -Rt[2,3], Rt[0,3]])

        global_points = np.asarray(global_map.points)
        new_global_points = np.asarray([-global_points[:,1], -global_points[:,2], global_points[:,0]])
        global_points = np.transpose(new_global_points)

        projected_global_img = self.projection(global_points, self.intrinsic_matrix, Rt)
        # plt.imshow(projected_global_img[:,:,2])

        return projected_global_img

    def LidarFrameProjection(self, lidar_pcd):
        lidar_points = np.asarray(lidar_pcd.points)
        new_lidar_points = np.asarray([-lidar_points[:,1], -lidar_points[:,2], lidar_points[:,0]])
        lidar_points = np.transpose(new_lidar_points)

        projected_lidar_img = self.projection(lidar_points, self.intrinsic_matrix, np.eye(4))
        # plt.imshow(projected_lidar_img[:,:,2])

        return projected_lidar_img

    def compute(self):

        n_frames = len(sorted(glob('./dump/%s/dump/0*' % self.seq_id)))
        global_map = o3d.io.read_point_cloud("./dump/%s/map.ply" % self.seq_id)
        # o3d.visualization.draw_geometries([global_map])

        log_dmp_list = []
        for frame_idx in range(n_frames):
            if (frame_idx%10) == 0 : 
                print(frame_idx, '/', n_frames)

            # global map projection
            projected_global_img = self.GlobalMapProjection(global_map, self.seq_id, frame_idx)

            # lidar frame projection
            lidar_pcd = o3d.io.read_point_cloud("./dump/%s/dump/%06d/cloud.pcd" % (self.seq_id, frame_idx))
            projected_lidar_img = self.LidarFrameProjection(lidar_pcd);

            # dmp calculation
            flatten_lidar_xyz = projected_lidar_img.reshape((-1,3))
            flatten_map_xyz = projected_global_img.reshape((-1,3))

            dmp = 0
            for lidar_val, map_val in zip(flatten_lidar_xyz, flatten_map_xyz):
                if (lidar_val[0] != np.inf and map_val[0] != np.inf):
                    dmp = dmp + np.linalg.norm(lidar_val - map_val)
            log_dmp_list.append(np.log(dmp))

        return np.mean(log_dmp_list)