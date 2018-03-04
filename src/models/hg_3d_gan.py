from .layers.Residual import Residual
import torch.nn as nn
import torch
import math
import ref
from utils.eval import getPreds

class Hourglass(nn.Module):
  def __init__(self, n, nModules, nFeats):
    super(Hourglass, self).__init__()
    self.n = n
    self.nModules = nModules
    self.nFeats = nFeats
    
    _up1_, _low1_, _low2_, _low3_ = [], [], [], []
    for j in range(self.nModules):
      _up1_.append(Residual(self.nFeats, self.nFeats))
    self.low1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
    for j in range(self.nModules):
      _low1_.append(Residual(self.nFeats, self.nFeats))
    
    if self.n > 1:
      self.low2 = Hourglass(n - 1, self.nModules, self.nFeats)
    else:
      for j in range(self.nModules):
        _low2_.append(Residual(self.nFeats, self.nFeats))
      self.low2_ = nn.ModuleList(_low2_)
    
    for j in range(self.nModules):
      _low3_.append(Residual(self.nFeats, self.nFeats))
    
    self.up1_ = nn.ModuleList(_up1_)
    self.low1_ = nn.ModuleList(_low1_)
    self.low3_ = nn.ModuleList(_low3_)
    
    self.up2 = nn.Upsample(scale_factor = 2)
    
  def forward(self, x):
    up1 = x
    for j in range(self.nModules):
      up1 = self.up1_[j](up1)
    
    # max pool
    low1 = self.low1(x)
    for j in range(self.nModules):
      low1 = self.low1_[j](low1)
    
    if self.n > 1:
      low2 = self.low2(low1)
    else:
      low2 = low1
      for j in range(self.nModules):
        low2 = self.low2_[j](low2)
    
    low3 = low2
    for j in range(self.nModules):
      low3 = self.low3_[j](low3)
    up2 = self.up2(low3)
    
    return up1 + up2

class HourglassNet3D(nn.Module):
  def __init__(self, nStack, nModules, nFeats, nRegModules):
    super(HourglassNet3D, self).__init__()
    self.nStack = nStack
    self.nModules = nModules
    self.nFeats = nFeats
    self.nRegModules = nRegModules
    self.conv1_ = nn.Conv2d(3, 64, bias = True, kernel_size = 7, stride = 2, padding = 3)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace = True)
    self.r1 = Residual(64, 128)
    self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
    self.r4 = Residual(128, 128)
    self.r5 = Residual(128, self.nFeats)
    
    _hourglass, _Residual, _lin_, _tmpOut, _ll_, _tmpOut_, _reg_ = [], [], [], [], [], [], []
    Hourglass(4, self.nModules, self.nFeats)
    for i in range(self.nStack):
      _hourglass.append(Hourglass(4, self.nModules, self.nFeats))
      for j in range(self.nModules):
        _Residual.append(Residual(self.nFeats, self.nFeats))
      lin = nn.Sequential(nn.Conv2d(self.nFeats, self.nFeats, bias = True, kernel_size = 1, stride = 1), 
                          nn.BatchNorm2d(self.nFeats), self.relu)
      _lin_.append(lin)
      _tmpOut.append(nn.Conv2d(self.nFeats, ref.nJoints, bias = True, kernel_size = 1, stride = 1))
      _ll_.append(nn.Conv2d(self.nFeats, self.nFeats, bias = True, kernel_size = 1, stride = 1))
      _tmpOut_.append(nn.Conv2d(ref.nJoints, self.nFeats, bias = True, kernel_size = 1, stride = 1))

    for i in range(4):
      for j in range(self.nRegModules):
        _reg_.append(Residual(self.nFeats, self.nFeats))
        
    self.hourglass = nn.ModuleList(_hourglass)
    self.Residual = nn.ModuleList(_Residual)
    self.lin_ = nn.ModuleList(_lin_)
    self.tmpOut = nn.ModuleList(_tmpOut)
    self.ll_ = nn.ModuleList(_ll_)
    self.tmpOut_ = nn.ModuleList(_tmpOut_)
    self.reg_ = nn.ModuleList(_reg_)
    
    self.reg = nn.Linear(4 * 4 * self.nFeats, ref.nJoints)
    
  def forward(self, x):
    x = self.conv1_(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.r1(x)
    x = self.maxpool(x)
    x = self.r4(x)
    x = self.r5(x)
    
    out = []
    
    for i in range(self.nStack):
      hg = self.hourglass[i](x)
      ll = hg
      for j in range(self.nModules):
        ll = self.Residual[i * self.nModules + j](ll)
      ll = self.lin_[i](ll)
      tmpOut = self.tmpOut[i](ll)
      out.append(tmpOut)
      
      ll_ = self.ll_[i](ll)
      tmpOut_ = self.tmpOut_[i](tmpOut)
      x = x + ll_ + tmpOut_
    
    for i in range(4):
      for j in range(self.nRegModules):
        x = self.reg_[i * self.nRegModules + j](x)
      x = self.maxpool(x)
      
    x = x.view(x.size(0), -1)
    reg = self.reg(x)
    out.append(reg)
    
    return out

class TripleJointDis(nn.Module):
    def __init__(self):
      super(TripleJointDis, self).__init__()

      sequence = [
            nn.Linear(48, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        ]

      self.model = nn.Sequential(*sequence)

    def forward(self, input):
      return self.model(input)



class Hourglass3DGAN():

  def __init__(self, opt):
    self.regWeight = opt.regWeight
    self.adWeight = opt.adWeight
    self.nStack = opt.nStack
    self.noise_real = opt.noise_real
    
    if opt.loadModel != 'none':
      self.netG = torch.load(opt.loadModel).cuda()
    else:
      self.netG = HourglassNet3D(opt.nStack, opt.nModules, opt.nFeats, opt.nRegModules).cuda()
    if opt.loadDis != 'none':
      self.netD = torch.load(opt.loadDis).cuda()
    else:
      self.netD = TripleJointDis().cuda()
    
    self.optimizer_G = torch.optim.RMSprop(self.netG.parameters(), opt.LR, 
                                  alpha = ref.alpha, 
                                  eps = ref.epsilon, 
                                  weight_decay = ref.weightDecay, 
                                  momentum = ref.momentum)

    self.optimizer_D = torch.optim.Adam(self.netD.parameters(), opt.LR/2000)

    self.criterion_G_2D = torch.nn.MSELoss().cuda()
    self.criterion_G_depth = torch.nn.MSELoss().cuda()
    self.criterion_D = torch.nn.MSELoss().cuda()

  def create_dis_label(self, input, is_real):
    if is_real:
      return torch.autograd.Variable(torch.Tensor(input.size()).fill_(1).cuda(), requires_grad=False)
    else:
      return torch.autograd.Variable(torch.Tensor(input.size()).fill_(0).cuda(), requires_grad=False)

  def save_inputs_targets(self, input, target2D, target3D):
    self.input = torch.autograd.Variable(input).float().cuda()
    self.target3D = torch.autograd.Variable(target3D).float().cuda().view(target3D.size(0), ref.nJoints, 3)
    self.target2D = torch.autograd.Variable(target2D).float().cuda()

  def forward(self):
    self.output = self.netG(self.input)
    return self.output

  def backward_G(self):
      
    reg = self.output[self.nStack]
    z = self.target3D[:, :, 2]
    batchSize = self.target3D.size(0)
    loss = torch.autograd.Variable(torch.FloatTensor(1).cuda()*0)
    loss_adv = torch.autograd.Variable(torch.FloatTensor(1).cuda()*0)
    loss_3d = torch.autograd.Variable(torch.FloatTensor(1).cuda()*0)
    loss_hm = torch.autograd.Variable(torch.FloatTensor(1).cuda()*0)
    self.fake_pool = []
    self.real_pool = []
    xy = torch.Tensor(self.target3D[:, :, :2].data.cpu())
    xy = torch.autograd.Variable(xy.float().cuda())
    for t in range(batchSize):
      s = xy[t].sum().data[0]
      if s < ref.eps and s > - ref.eps: 
        #Supervised data
        l_sup = self.criterion_G_depth(reg[t], z[t])
        loss_3d += self.regWeight * l_sup
        self.real_pool.append(torch.cat((reg[t], xy[t].view(32)), 0).view(1,48))
      else:
        # GAN loss for unsupervised data
        fake_pose = torch.cat((reg[t].view(1,16), xy[t].view(1, 32)), 1)
        dis_fake = self.netD(fake_pose)
        l_GAN = self.criterion_D(dis_fake, self.create_dis_label(dis_fake, True))
        # add sample to fake list 
        self.fake_pool.append(fake_pose)
        # print(l_GAN)
        loss_adv += self.adWeight * l_GAN

    # Adding losses for heatmaps
    for k in range(self.nStack):
      loss_hm += self.criterion_G_2D(self.output[k], self.target2D)
    
    loss = (loss_3d + loss_adv)/batchSize + loss_hm

    self.Loss3D = loss_3d.data[0]/batchSize
    self.LossGAN = loss_adv.data[0]/batchSize
    self.LossHM = loss_hm.data[0]
    self.Loss = loss.data[0]

    if self.isTrain:
      loss.backward()



  def backward_D(self):
    # Real
    if len(self.real_pool) > 0 and len(self.fake_pool) > 0:
      num_real = len(self.real_pool)
      real_poses = torch.cat(self.real_pool, 0).detach()
      if self.noise_real > 0:
        noise = torch.autograd.Variable(torch.randn(real_poses.size()).cuda() * self.noise_real)
        real_poses += noise
        # print('asa', real_poses, noise)
      # print(real_poses)
      pred_real = self.netD(real_poses)
      loss_D_real = self.criterion_D(pred_real, self.create_dis_label(pred_real, True))
      # Fake
      fake_poses = torch.cat(self.fake_pool, 0)
      pred_fake = self.netD(fake_poses.detach())
      loss_D_fake = self.criterion_D(pred_fake, self.create_dis_label(pred_fake, False))
      # print('--------', len(self.real_pool), pred_real, len(self.fake_pool), pred_fake)
      # Combined loss
      loss_D = (loss_D_real + loss_D_fake) * 0.5
      # backward
      loss_D.backward()
      del self.real_pool
      del self.fake_pool
      return loss_D
    else:
      return 0


  def optimize_parameters(self):
      # forward
      self.forward()
      # G_A and G_B
      self.optimizer_G.zero_grad()
      self.backward_G()
      self.optimizer_D.zero_grad()
      self.backward_D()
      if self.isTrain:
        self.optimizer_G.step()
        # D_A
        self.optimizer_D.step()
