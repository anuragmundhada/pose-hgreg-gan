import torch
import numpy as np
from utils.utils import AverageMeter
from utils.eval import Accuracy, getPreds, MPJPE
from utils.debugger import Debugger, Show3D
from models.layers.FusionCriterion import FusionCriterion
import cv2
import ref
from progress.bar import Bar

def step(split, epoch, opt, dataLoader, model):
  if split == 'train':
    model.isTrain = True
  else:
    model.isTrain = False
  Loss, Acc, Mpjpe, Loss3D, LossGAN, LossHM = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
  
  nIters = len(dataLoader)
  bar = Bar('==>', max=nIters)
  
  for i, (input, target2D, target3D, meta) in enumerate(dataLoader):
    model.save_inputs_targets(input, target2D, target3D)

    output = model.forward()
    reg = output[opt.nStack]
    if opt.DEBUG >= 2:
      gt = getPreds(target2D.cpu().numpy()) * 4
      pred = getPreds((output[opt.nStack - 1].data).cpu().numpy()) * 4
      debugger = Debugger()
      debugger.addImg((input[0].numpy().transpose(1, 2, 0)*256).astype(np.uint8))
      debugger.addPoint2D(pred[0], (255, 0, 0))
      debugger.addPoint2D(gt[0], (0, 0, 255))
      debugger.showImg()
      debugger.saveImg('debug/{}.png'.format(i))
      pr = torch.FloatTensor(pred).view(16,2)
      # print(pr.shape, reg.data)
      pred = torch.cat((pr, reg.data.cpu().view(16,1)),1)
      Show3D(np, pred.numpy())

    # loss = FusionCriterion(opt.regWeight, opt.varWeight)(reg, target3D_var)
    model.optimize_parameters()

    Loss3D.update(model.Loss3D, input.size(0))
    LossGAN.update(model.LossGAN, input.size(0))
    LossHM.update(model.LossHM, input.size(0))
    Loss.update(model.Loss, input.size(0))
    Acc.update(Accuracy((output[opt.nStack - 1].data).cpu().numpy(), (model.target2D.data).cpu().numpy()))
    mpjpe, num3D = MPJPE((output[opt.nStack - 1].data).cpu().numpy(), (reg.data).cpu().numpy(), meta)

    if num3D > 0:
      Mpjpe.update(mpjpe, num3D)

    Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Tot: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | HM {losshm.avg:.6f} | 3D {loss3d.avg:.6f} | GAN {lossgan.avg:.6f} | Acc {Acc.avg:.6f} | Mpjpe {Mpjpe.avg:.6f} ({Mpjpe.val:.6f})'.format(epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, Acc=Acc, split = split, Mpjpe=Mpjpe, loss3d = Loss3D, lossgan = LossGAN, losshm = LossHM)
    bar.next()

  bar.finish()
  return Loss.avg, Acc.avg, Mpjpe.avg, Loss3D.avg
  

def train(epoch, opt, train_loader, model):
  return step('train', epoch, opt, train_loader, model)
  
def val(epoch, opt, val_loader, model):
  return step('val', epoch, opt, val_loader, model)
