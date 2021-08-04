import torch
import numpy as np
from sklearn.metrics import roc_curve, auc

# adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/score.py
MAX_IMG_PER_BATCH = 256

class IoUEval:
    def __init__(self, nthresh=255):
        self.nthresh = nthresh
        self.thresh = torch.linspace(1./(nthresh + 1), 1. - 1./(nthresh + 1), nthresh).cuda()
        self.EPSILON = np.finfo(np.float).eps

        self.gt_sum = torch.zeros((nthresh,)).cuda()
        self.pred_sum = torch.zeros((nthresh,)).cuda()
        self.num_images = 0
        self.mae = 0
        self.prec = torch.zeros(self.nthresh).cuda()
        self.recall = torch.zeros(self.nthresh).cuda()
        self.iou = 0.
        self.auc_roc_list = []


    def add_batch(self, predict, gth):
#         print(torch.max(predict[0]))
#         print(torch.max(gth[0]))
#         exit()
#         print(predict.shape, gth.shape)
#         predict[predict > 0.0] = 1.0
#         p = predict.flatten().data.cpu().numpy()
#         g = gth.flatten().data.cpu().numpy()
#         p[p > 0.0] = 1.0
#         g[g > 0.0] = 1.0
        
#         (p_unique, counts) = np.unique(p, return_counts=True)
#         p_frequencies = np.asarray((p_unique, counts)).T
        
#         (g_unique, counts) = np.unique(g, return_counts=True)
#         g_frequencies = np.asarray((g_unique, counts)).T
        
#         fpr, tpr, _ = roc_curve(p,g)
#         roc_auc = auc(fpr,tpr)
#         print(p_unique, g_unique)
#         print(torch.max(predict), torch.min(predict))
        
        for i in range(predict.shape[0]):
            
            dt = predict[i]; gt = gth[i]
            self.mae += (dt-gt).abs().mean()
            dt = dt > (dt.mean() * 1)
            gt = gt > 0.5
            
            intersect = (dt*gt).sum()
#             print(dt.shape)
            iou = intersect.float() / (dt.sum() + gt.sum() - intersect).float()
            self.iou += iou
            
            p = dt.flatten().data.cpu().numpy()
            g = gt.flatten().data.cpu().numpy()
            
            fpr, tpr, _ = roc_curve(g, p)
            roc_auc = auc(fpr,tpr)
#             print(p_unique, g_unique)
#             print(roc_auc)
            self.auc_roc_list.append(roc_auc)
    
        self.num_images += predict.shape[0]
        

    def get_metric(self):
        x = self.iou / self.num_images
        y = self.mae / self.num_images
#         print(self.iou, self.mae, self.num_images)
        return x, y

    def get_auc_roc(self):
        return sum(self.auc_roc_list) / len(self.auc_roc_list)