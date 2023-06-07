import numpy as np
import torch


def get_dists(points1, points2):
    '''
    Calculate dists between two group points
    :param cur_point: shape=(B, M, C)
    :param points: shape=(B, N, C)
    :return:
    '''
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists) # Very Important for dist = 0.
    return torch.sqrt(dists).float()


def gather_points(points, inds):
    '''
    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]
def three_nn(xyz1, xyz2):
    '''
    :param xyz1: shape=(B, N1, 3)
    :param xyz2: shape=(B, N2, 3)
    :return: dists: shape=(B, N1, 3), inds: shape=(B, N1, 3)
    '''
    dists = get_dists(xyz1, xyz2)
    dists, inds = torch.sort(dists, dim=-1)
    dists, inds = dists[:, :, :3], inds[:, :, :3]
    return dists, inds


def three_interpolate(xyz1, xyz2, points2):
    '''
    :param xyz1: shape=(B, N1, 3)
    :param xyz2: shape=(B, N2, 3)
    :param points2: shape=(B, N2, C2)
    :return: interpolated_points: shape=(B, N1, C2)
    '''
    dists, inds = three_nn(xyz1, xyz2)
    inversed_dists = 1.0 / (dists + 1e-8)
    weight = inversed_dists / torch.sum(inversed_dists, dim=-1, keepdim=True) # shape=(B, N1, 3)
    weight_xyz = torch.unsqueeze(weight, -1).repeat(1, 1, 1, 3)
    interpolated_xyzs = gather_points(xyz2, inds)
    interpolated_xyzs = torch.sum(weight_xyz * interpolated_xyzs, dim=2)
    if points2 is not None:
        _, _, C2 = points2.shape
        weight = torch.unsqueeze(weight, -1).repeat(1, 1, 1, C2)
        interpolated_points = gather_points(points2, inds)  # shape=(B, N1, 3, C2)
        interpolated_points = torch.sum(weight * interpolated_points, dim=2)
        return interpolated_xyzs, interpolated_points
    return interpolated_xyzs, None
def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()


def fps(points, num):
    cids = []
    cid = np.random.choice(points.shape[0])
    cids.append(cid)
    id_flag = np.zeros(points.shape[0])
    id_flag[cid] = 1

    dist = torch.zeros(points.shape[0]) + 1e4
    dist = dist.type_as(points)
    while np.sum(id_flag) < num:
        dist_c = torch.norm(points - points[cids[-1]], p=2, dim=1)
        dist = torch.where(dist<dist_c, dist, dist_c)
        dist[id_flag == 1] = 1e4
        new_cid = torch.argmin(dist)
        id_flag[new_cid] = 1
        cids.append(new_cid)
    cids = torch.Tensor(cids).long()
    return cids
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
# 1. inv
class PointcloudScale(object):
    def __init__(self, N, lo=0.8, hi=1.25, p=1):
        self.lo, self.hi = lo, hi
        self.p = p
        self.plist = np.random.random_sample(N)
        # self.scaler = np.random.uniform(self.lo, self.hi, N)

    def __call__(self, index, points):
        # if np.random.uniform(0, 1) > self.p:
        if self.p < self.plist[index]:
            return points

        points[:, 0:3] *= np.random.uniform(self.lo, self.hi)
        return points

# 3. inv
class PointcloudRotatePerturbation(object):
    def __init__(self, N, angle_sigma=0.06, angle_clip=0.18, p=1):
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip
        self.p = p
        self.plist = np.random.random_sample(N)
        self.to_tensor = PointcloudToTensor()

    def _get_angles(self, index):
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        )

        return angles

    def __call__(self, index, points):
        points = self.to_tensor(points)

        # if np.random.uniform(0, 1) > self.p:
        if self.p < self.plist[index]:
            return points
        angles = self._get_angles(index)
        Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)
        normals = points.shape[1] > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
            # return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points

# 4. inv
class PointcloudJitter(object):
    def __init__(self, N, std=0.01, clip=0.025, p=1, points_num=2048):
        self.std, self.clip = std, clip
        self.p = p
        self.plist = np.random.random_sample(N)
        self.points_num = points_num
    def __call__(self, index, points):
        if self.p < self.plist[index]:
            return points
        points[:, 0:3] += torch.zeros((self.points_num, 3)).normal_(mean=0.0, std=self.std).clamp_(-self.clip, self.clip).data.cpu().numpy()
        return points

# 4. inv
class PointcloudNoise(object):
    def __init__(self, N, std=0.01, clip=0.025, p=1, points_num=2048):
        self.std, self.clip = std, clip
        self.p = p
        self.plist = np.random.random_sample(N)
        self.points_num = points_num
    def __call__(self, index, points):
        if self.p < self.plist[index]:
            return points
        points[:, 0:3] += torch.zeros((self.points_num, 3)).normal_(mean=0.0, std=self.std).data.cpu().numpy()
        return points

# 5. inv
class PointcloudTranslate(object):
    def __init__(self, N, translate_range=0.1, p=1):
        self.translate_range = translate_range
        self.p = p
        self.plist = np.random.random_sample(N)
        # self.rlist = [np.random.uniform(-self.translate_range, self.translate_range, size=(3)) for _ in range(N)]
        self.rlist = [None for _ in range(N)]
    def __call__(self, index, points):
        if self.p < self.plist[index]:
            return points
        if self.rlist[index] is not None:
            points[:, 0:3] += self.rlist[index]
            return torch.from_numpy(points).float()
        points = points.numpy()
        coord_min = np.min(points[:,:3], axis=0)
        coord_max = np.max(points[:,:3], axis=0)
        coord_diff = coord_max - coord_min
        translation = np.random.uniform(-self.translate_range, self.translate_range, size=(3)) * coord_diff
        self.rlist[index] = translation
        points[:, 0:3] += translation
        return torch.from_numpy(points).float()

# to tensor
class PointcloudToTensor(object):
    def __call__(self, points):
        if isinstance(points, np.ndarray):
            return torch.from_numpy(points).float()
        else:
            return points

# 8. inv
class PointcloudNormalize(object):
    def __init__(self, radius=1):
        self.radius = radius

    def pc_normalize(self, pc):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __call__(self, points):
        if not isinstance(points, np.ndarray):
            pc = points.numpy()
        else:
            pc = points
        pc[:, 0:3] = self.pc_normalize(pc[:, 0:3])
        return torch.from_numpy(pc).float()

# 9. eqv
class PointcloudRemoveInvalid(object):
    def __init__(self, invalid_value=0):
        self.invalid_value = invalid_value

    def __call__(self, points):
        pc = points.numpy()
        valid = np.sum(pc, axis=1) != self.invalid_value
        pc = pc[valid, :]
        return torch.from_numpy(pc).float()


# 10. eqv
class PointcloudRandomCrop(object):
    def __init__(self, N, res1, res2, x_min=0.6, x_max=1.1, ar_min=0.75, ar_max=1.33, p=1, min_num_points=1024, max_try_num=100):
        self.x_min = x_min
        self.x_max = x_max

        self.ar_min = ar_min
        self.ar_max = ar_max

        self.p = p

        self.max_try_num = max_try_num
        self.min_num_points = min_num_points

        self.res1 = res1
        self.res2 = res2

        self.new_coord_mins = [np.zeros(3) for _ in range(N)]
        self.new_coord_maxs = [np.zeros(3) for _ in range(N)]
        self.indexs = [[] for _ in range(N)]
        self.isvalid = [False for _ in range(N)]

    def random_crop(self, index, points, feature=None):
        device = points.device
        points = points.data.cpu().numpy()
        if feature is not None:
            feature = feature.data.cpu().numpy()
            if self.isvalid[index]:
                new_indices = self.indexs[index]
                new_points = points[new_indices]
                new_feature = feature[new_indices]
                res_indices = [not a for a in new_indices]
                return torch.from_numpy(new_points).float().to(device), torch.from_numpy(new_feature).float().to(device), res_indices
            else:
                return torch.from_numpy(points).float().to(device), torch.from_numpy(feature).float().to(device), None
        isvalid = False
        try_num = 0
        while not isvalid:
            coord_min = np.min(points[:,:3], axis=0)
            coord_max = np.max(points[:,:3], axis=0)
            coord_diff = coord_max - coord_min
            # resampling later, so only consider crop here
            # get indices of cropped points
            new_coord_range = np.zeros(3)
            new_coord_range[0] = np.random.uniform(self.x_min, self.x_max)
            ar = np.random.uniform(self.ar_min, self.ar_max)
            new_coord_range[1] = new_coord_range[0] * ar
            new_coord_range[2] = new_coord_range[0] / ar

            new_coord_min = np.random.uniform(0, 1-new_coord_range)
            new_coord_max = new_coord_min + new_coord_range

            new_coord_min = coord_min + coord_diff * new_coord_min
            new_coord_max = coord_min + coord_diff * new_coord_max

            new_indices = (points[:,:3] > new_coord_min) & (points[:, :3] < new_coord_max)
            new_indices = np.sum(new_indices, axis=1) == 3
            new_points = points[new_indices]
            res_indices = [not a for a in new_indices]

            if new_points.shape[0] >= self.min_num_points and new_points.shape[0] < points.shape[0]:
                isvalid = True
                self.isvalid[index] = True
                self.new_coord_mins[index] = new_coord_min
                self.new_coord_maxs[index] = new_coord_max
                self.indexs[index] = new_indices
            try_num += 1
            if try_num > self.max_try_num:
                return torch.from_numpy(points).float().to(device), None, None

        return torch.from_numpy(new_points).float().to(device), None, res_indices
    def __call__(self, indices, points, features=None):
        new_points = []
        new_features = []

        if features is not None:
            for i, idx in enumerate(indices):
                object = points[i]
                feature = features[i]
                new_obj, new_feature, res_indices = self.random_crop(idx, object, feature)
                new_obj, new_feature = three_interpolate(object.unsqueeze(0), new_obj.unsqueeze(0), new_feature.unsqueeze(0))
                new_points.append(new_obj)
                new_features.append(new_feature)
            new_points, new_features = torch.cat(new_points, 0), torch.cat(new_features, 0)
            return new_points, new_features
        else:
            for i, idx in enumerate(indices):
                object = points[i]
                new_obj,_,res_indices = self.random_crop(idx, object, None)
                new_obj, _ = three_interpolate(object.unsqueeze(0), new_obj.unsqueeze(0), None)
                new_points.append(new_obj)
            new_points = torch.cat(new_points,0)
            return new_points,None


class BaseTransform(object):
    """
    Resize and center crop.
    """

    def __init__(self, res):
        self.res = res

    def __call__(self, index, image):
        image = torch.from_numpy(image)
        idx = fps(image, self.res)
        image = image[idx]
        return image

if __name__ == "__main__":
    points = np.random.uniform(0, 100, [22,4096, 3])
    points = torch.Tensor(points)
    crop = PointcloudRandomCrop()
    crop_points = crop(points)
    print(crop_points.shape)

