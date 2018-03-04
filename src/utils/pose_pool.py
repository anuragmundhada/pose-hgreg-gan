import random
import torch
from torch.autograd import Variable


class PosePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.poses = []

    def query(self, poses):
        if self.pool_size == 0:
            return Variable(poses)
        return_poses = []
        for pose in poses:
            pose = torch.unsqueeze(pose, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.poses.append(pose)
                return_poses.append(pose)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.poses[random_id].clone()
                    self.poses[random_id] = pose
                    return_poses.append(tmp)
                else:
                    return_poses.append(pose)
        return_poses = Variable(torch.cat(return_poses, 0))
        return return_poses
