import numpy as np
from mmdet.datasets.builder import PIPELINES
from shapely.geometry import LineString
from numpy.typing import NDArray
from typing import List, Tuple, Union, Dict
from IPython import embed
import cv2
import av2.geometry.interpolate as interp_utils
import copy
def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords

@PIPELINES.register_module(force=True) 
class VectorizeMap_ins(object):
    """Generate vectoized map and put into `semantic_mask` key.
    Concretely, shapely geometry objects are converted into sample points (ndarray).
    We use args `sample_num`, `sample_dist`, `simplify` to specify sampling method.

    Args:
        roi_size (tuple or list): bev range .
        normalize (bool): whether to normalize points to range (0, 1).
        coords_dim (int): dimension of point coordinates.
        simplify (bool): whether to use simpily function. If true, `sample_num` \
            and `sample_dist` will be ignored.
        sample_num (int): number of points to interpolate from a polyline. Set to -1 to ignore.
        sample_dist (float): interpolate distance. Set to -1 to ignore.
    """

    def __init__(self, 
                 roi_size: Union[Tuple, List], 
                 normalize: bool,
                 coords_dim: int,
                 simplify: bool=False, 
                 sample_num: int=-1, 
                 sample_dist: float=-1, 
                 permute: bool=False
        ):
        self.coords_dim = coords_dim
        self.sample_num = sample_num
        self.sample_dist = sample_dist
        self.roi_size = np.array(roi_size)
        self.normalize = normalize
        self.simplify = simplify
        self.permute = permute

        if sample_dist > 0:
            assert sample_num < 0 and not simplify
            self.sample_fn = self.interp_fixed_dist
        elif sample_num > 0:
            assert sample_dist < 0 and not simplify
            self.sample_fn = self.interp_fixed_num
        else:
            assert simplify

    def interp_fixed_num(self, line: LineString) -> NDArray:
        ''' Interpolate a line to fixed number of points.
        
        Args:
            line (LineString): line
        
        Returns:
            points (array): interpolated points, shape (N, 2)
        '''

        distances = np.linspace(0, line.length, self.sample_num)
        sampled_points = np.array([list(line.interpolate(distance).coords) 
            for distance in distances]).squeeze()

        return sampled_points

    def interp_fixed_dist(self, line: LineString) -> NDArray:
        ''' Interpolate a line at fixed interval.
        
        Args:
            line (LineString): line
        
        Returns:
            points (array): interpolated points, shape (N, 2)
        '''

        distances = list(np.arange(self.sample_dist, line.length, self.sample_dist))
        # make sure to sample at least two points when sample_dist > line.length
        distances = [0,] + distances + [line.length,] 
        
        sampled_points = np.array([list(line.interpolate(distance).coords)
                                for distance in distances]).squeeze()
        
        return sampled_points
    
    def get_vectorized_lines(self, map_geoms: Dict) -> Dict:
        ''' Vectorize map elements. Iterate over the input dict and apply the 
        specified sample funcion.
        
        Args:
            line (LineString): line
        
        Returns:
            vectors (array): dict of vectorized map elements.
        '''

        vectors = {}
        pure_geoms = {}
        for label, geom_list in map_geoms.items():
            vectors[label] = []
            pure_geoms[label] = []
            
            for geom in geom_list:
                if geom.geom_type == 'LineString':
                    if self.simplify:
                        line = geom.simplify(0.2, preserve_topology=True)
                        line = np.array(line.coords)
                    else:
                        line = self.sample_fn(geom)
                    line_simp = geom.simplify(0.2, preserve_topology=True)
                    line_simp = np.array(line_simp.coords)
                    line_simp = line_simp[:, :self.coords_dim]
                    pure_geoms[label].append(line_simp)

                    line = line[:, :self.coords_dim]
                    if self.normalize:
                        line = self.normalize_line(line)
                    if self.permute:
                        line = self.permute_line(line)
                    vectors[label].append(line)
                    

                elif geom.geom_type == 'Polygon':
                    # polygon objects will not be vectorized
                    continue
                
                else:
                    raise ValueError('map geoms must be either LineString or Polygon!')
        return vectors, pure_geoms
    
    def normalize_line(self, line: NDArray) -> NDArray:
        ''' Convert points to range (0, 1).
        
        Args:
            line (LineString): line
        
        Returns:
            normalized (array): normalized points.
        '''

        origin = -np.array([self.roi_size[0]/2, self.roi_size[1]/2])

        line[:, :2] = line[:, :2] - origin

        # transform from range [0, 1] to (0, 1)
        eps = 1e-5
        line[:, :2] = line[:, :2] / (self.roi_size + eps)

        return line
    
    def permute_line(self, line: np.ndarray, padding=1e5):
        '''
        (num_pts, 2) -> (num_permute, num_pts, 2)
        where num_permute = 2 * (num_pts - 1)
        '''
        is_closed = np.allclose(line[0], line[-1], atol=1e-3)
        num_points = len(line)
        permute_num = num_points - 1
        permute_lines_list = []
        if is_closed:
            pts_to_permute = line[:-1, :] # throw away replicate start end pts
            for shift_i in range(permute_num):
                permute_lines_list.append(np.roll(pts_to_permute, shift_i, axis=0))
            flip_pts_to_permute = np.flip(pts_to_permute, axis=0)
            for shift_i in range(permute_num):
                permute_lines_list.append(np.roll(flip_pts_to_permute, shift_i, axis=0))
        else:
            permute_lines_list.append(line)
            permute_lines_list.append(np.flip(line, axis=0))

        permute_lines_array = np.stack(permute_lines_list, axis=0)

        if is_closed:
            tmp = np.zeros((permute_num * 2, num_points, self.coords_dim))
            tmp[:, :-1, :] = permute_lines_array
            tmp[:, -1, :] = permute_lines_array[:, 0, :] # add replicate start end pts
            permute_lines_array = tmp

        else:
            # padding
            padding = np.full([permute_num * 2 - 2, num_points, self.coords_dim], padding)
            permute_lines_array = np.concatenate((permute_lines_array, padding), axis=0)
        
        return permute_lines_array

    def remove_nan_values(self,uv):
        is_u_valid = np.logical_not(np.isnan(uv[:, 0]))
        is_v_valid = np.logical_not(np.isnan(uv[:, 1]))
        is_uv_valid = np.logical_and(is_u_valid, is_v_valid)

        uv_valid = uv[is_uv_valid]
        return uv_valid

    def points_ego2img(self,pts_ego, extrinsics, intrinsics):
        pts_ego_4d = np.concatenate([pts_ego, np.ones([len(pts_ego), 1])], axis=-1)
        pts_cam_4d = extrinsics @ pts_ego_4d.T
        
        uv = (intrinsics @ pts_cam_4d[:3, :]).T
        uv = self.remove_nan_values(uv)
        depth = uv[:, 2]
        uv = uv[:, :2] / uv[:, 2].reshape(-1, 1)

        return uv, depth


    def draw_visible_polyline_cv2(self,line, valid_pts_bool, image, color, thickness_px):
        """Draw a polyline onto an image using given line segments.

        Args:
            line: Array of shape (K, 2) representing the coordinates of line.
            valid_pts_bool: Array of shape (K,) representing which polyline coordinates are valid for rendering.
                For example, if the coordinate is occluded, a user might specify that it is invalid.
                Line segments touching an invalid vertex will not be rendered.
            image: Array of shape (H, W, 3), representing a 3-channel BGR image
            color: Tuple of shape (3,) with a BGR format color
            thickness_px: thickness (in pixels) to use when rendering the polyline.
        """
        line = np.round(line).astype(int)  # type: ignore
        for i in range(len(line) - 1):

            if (not valid_pts_bool[i]) or (not valid_pts_bool[i + 1]):
                continue
            # np.round(uv[is_valid_points]).astype(np.int32)
            scale_size = 16
            x1 = np.round(line[i][0]/scale_size).astype(np.int32)
            y1 = np.round(line[i][1]/scale_size).astype(np.int32)
            x2 = np.round(line[i + 1][0]/scale_size).astype(np.int32)
            y2 =  np.round(line[i + 1][1]/scale_size).astype(np.int32)

            # Use anti-aliasing (AA) for curves
            image = cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=1)
            # if np.any(image != 0):
            #     print("画上了")

    def line_ego_to_pvmask(self,
                          polyline_ego, 
                          mask, 
                        #   lidar2feat,
                          extrinsics, 
                          intrinsics, 
                          color=1, 
                          thickness=1,
                          z=-1.6):
        # distances = np.linspace(0, line_ego.length, 200)
        # coords = np.array([list(line_ego.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
        # print("coords",coords)
        # pts_num = coords.shape[0]
        # zeros = np.zeros((pts_num,1))
        # zeros[:] = z
        # ones = np.ones((pts_num,1))
        # #pts_ego_4d
        # lidar_coords = np.concatenate([coords,zeros,ones], axis=1).transpose(1,0)
        # print("lidar_coords",lidar_coords)
        # print("lidar2feat",lidar2feat)
        # pix_coords = perspective(lidar_coords, lidar2feat)
        # print("pix_coords",pix_coords)
        # cv2.polylines(mask, np.int32([pix_coords]), False, color=color, thickness=thickness)
        if polyline_ego.shape[1] == 2:
            zeros = np.zeros((polyline_ego.shape[0], 1))
            polyline_ego = np.concatenate([polyline_ego, zeros], axis=1)

        polyline_ego = interp_utils.interp_arc(t=200, points=polyline_ego)
        
        uv, depth = self.points_ego2img(polyline_ego, extrinsics, intrinsics)

        # h, w, c = img_bgr.shape
        h=480
        w=800
        is_valid_x = np.logical_and(0 <= uv[:, 0], uv[:, 0] < w - 1)
        is_valid_y = np.logical_and(0 <= uv[:, 1], uv[:, 1] < h - 1)
        is_valid_z = depth > 0
        is_valid_points = np.logical_and.reduce([is_valid_x, is_valid_y, is_valid_z])

        if is_valid_points.sum() == 0:
            return
        
        uv = np.round(uv[is_valid_points]).astype(np.int32)

        self.draw_visible_polyline_cv2(
            copy.deepcopy(uv),
            valid_pts_bool=np.ones((len(uv), 1), dtype=bool),
            color=color,
            image=mask,
            thickness_px=thickness,
        )


    def get_semantic_mask(self, input_dict):
        num_cam=len(input_dict['img_filenames'])
        gt_pv_semantic_mask = np.zeros((num_cam, 1,30, 50), dtype=np.uint8)
        gt_pv_ins_mask = np.zeros((num_cam, 100,30, 50), dtype=np.uint8)
        # scale_factor = np.eye(4)
        # scale_factor[0, 0] *= 1/32
        # scale_factor[1, 1] *= 1/32
        # ego2feat = [scale_factor @ l2i for l2i in input_dict['ego2img']]
        scaleW=0.5
        scaleH=480/900
        rot_resize_matrix = np.array([
            [scaleW, 0,      0,    0],
            [0,      scaleH, 0,    0],
            [0,      0,      1,    0],
            [0,      0,      0,    1]])
        # post_intrinsic = rot_resize_matrix[:3, :3] @ cam_intrinsic
        
        cam_extrinsics = input_dict['cam_extrinsics']
        cam_intrinsics = input_dict['cam_intrinsics']
        cam_intrinsics= [rot_resize_matrix[:3, :3] @ l2i for l2i in cam_intrinsics]
        # category configs
        cat2id = {
            'ped_crossing': 0,
            'divider': 1,
            'boundary': 2,
        }
        id2cat = {v: k for k, v in cat2id.items()}

        # for line_instance in input_dict['pure_lines']['all']:
        line_count = 0
        for label, vector_list in input_dict['pure_lines'].items():
           # ins_cls = cat2id[label]
            # print("ins_cls",ins_cls)
            # color = COLOR_MAPS_PLT[cat]
            
            for vector in vector_list:
                if isinstance(vector, list):
                    vector = np.array(vector)
                
                for cam_index in range(num_cam):
                    self.line_ego_to_pvmask(vector, gt_pv_semantic_mask[cam_index][0], cam_extrinsics[cam_index], cam_intrinsics[cam_index],color=1, thickness=1)
                    self.line_ego_to_pvmask(vector, gt_pv_ins_mask[cam_index][line_count], cam_extrinsics[cam_index], cam_intrinsics[cam_index],color=1, thickness=1)
                line_count+=1    
                    # print(gt_pv_semantic_mask[cam_index][0])
        #input_dict['gt_pv_semantic_mask'] = gt_pv_semantic_mask
        return gt_pv_semantic_mask, gt_pv_ins_mask
    
    def ego2pv_line(self, polyline_ego, extrinsics, intrinsics):
        if polyline_ego.shape[1] == 2:
            zeros = np.zeros((polyline_ego.shape[0], 1))
            polyline_ego = np.concatenate([polyline_ego, zeros], axis=1)

        polyline_ego = interp_utils.interp_arc(t=200, points=polyline_ego)
        
        uv, depth = self.points_ego2img(polyline_ego, extrinsics, intrinsics)

        # h, w, c = img_bgr.shape
        h=480
        w=800
        is_valid_x = np.logical_and(0 <= uv[:, 0], uv[:, 0] < w - 1)
        is_valid_y = np.logical_and(0 <= uv[:, 1], uv[:, 1] < h - 1)
        is_valid_z = depth > 0
        is_valid_points = np.logical_and.reduce([is_valid_x, is_valid_y, is_valid_z])

        if is_valid_points.sum() == 0:
            return 
        
        uv = np.round(uv[is_valid_points]).astype(np.int32)
        return uv

    def get_pv_vectorized_lines(self, input_dict):
        num_cam=len(input_dict['img_filenames'])
        gt_pv_semantic_mask = np.zeros((num_cam, 1,30, 50), dtype=np.uint8)
        gt_pv_ins_mask = np.zeros((num_cam, 100,30, 50), dtype=np.uint8)
        # scale_factor = np.eye(4)
        # scale_factor[0, 0] *= 1/32
        # scale_factor[1, 1] *= 1/32
        # ego2feat = [scale_factor @ l2i for l2i in input_dict['ego2img']]
        scaleW=0.5
        scaleH=480/900
        rot_resize_matrix = np.array([
            [scaleW, 0,      0,    0],
            [0,      scaleH, 0,    0],
            [0,      0,      1,    0],
            [0,      0,      0,    1]])
        # post_intrinsic = rot_resize_matrix[:3, :3] @ cam_intrinsic
        
        cam_extrinsics = input_dict['cam_extrinsics']
        cam_intrinsics = input_dict['cam_intrinsics']
        # print("Before_ input_dict['cam_intrinsics']", input_dict['cam_intrinsics'])
        cam_intrinsics= [rot_resize_matrix[:3, :3] @ l2i for l2i in cam_intrinsics]
        # print("vis_ex",cam_extrinsics)

        # print("vis_ins",cam_intrinsics)
        # print("After_ input_dict['cam_intrinsics']", input_dict['cam_intrinsics'])
        # category configs
        cat2id = {
            'ped_crossing': 0,
            'divider': 1,
            'boundary': 2,
        }
        id2cat = {v: k for k, v in cat2id.items()}
        all_pv_vectors = []
        scale_size = 16   #
        for cam_index in range(num_cam):
            pv_vec={}
            for label, vector_list in input_dict['pure_lines'].items():
                pv_vec[label]=[]
                for vector in vector_list:
                    if isinstance(vector, list):
                        vector = np.array(vector)
                    pv_vector = self.ego2pv_line(vector, cam_extrinsics[cam_index], cam_intrinsics[cam_index])
                    if isinstance(pv_vector, np.ndarray) and pv_vector.shape[0]>=2:
                        #pv_vector = interp_utils.interp_arc(t=20, points=pv_vector) #800 480
                        #pv_vector = self.remove_nan_values(pv_vector)
                        pv_linestring= LineString(pv_vector)
                        
                        pv_vector = self.sample_fn(pv_linestring)
                        #print(pv_vector)
                        normed_pv_vector = pv_vector/(800, 480) ### 直接归一化
                        permuted_normed_pv_vector = self.permute_line(normed_pv_vector)  ### 然后permute
                        pv_vec[label].append(permuted_normed_pv_vector)
                        # pv_vec[label].append(normed_pv_vector)
            all_pv_vectors.append(pv_vec)
         
        return all_pv_vectors
        
    def __call__(self, input_dict):
        map_geoms = input_dict['map_geoms']

        input_dict['vectors'], input_dict['pure_lines'] = self.get_vectorized_lines(map_geoms)
        #input_dict['gt_pv_semantic_mask'], input_dict['gt_pv_ins_mask'] = self.get_semantic_mask(input_dict)
        input_dict['pv_vectors'] = self.get_pv_vectorized_lines(input_dict)

        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(simplify={self.simplify}, '
        repr_str += f'sample_num={self.sample_num}), '
        repr_str += f'sample_dist={self.sample_dist}), ' 
        repr_str += f'roi_size={self.roi_size})'
        repr_str += f'normalize={self.normalize})'
        repr_str += f'coords_dim={self.coords_dim})'

        return repr_str