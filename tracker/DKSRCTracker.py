import random
from time import time
import visdom
import numpy as np
from got10k.trackers import Tracker
import skfeature.function.similarity_based.trace_ratio as trace_r
from tracker.params import p
import torchvision.transforms as T
import torch
torch.cuda.current_device()
from tracker.dim32 import Encoder
from utils.bbreg import BBRegressor, BBRegressorNN
from utils.crop_img import crop_samples
from utils.traindictKernel import DKSRC, evalKKSVD
from utils.sample_geneator import SampleGenerator
import torch.nn.functional as F
from concurrent import futures

if p.debug:
    viz = visdom.Visdom()

class DKSRCTracker(Tracker):
    def __init__(self):
        super(DKSRCTracker, self).__init__(
            name='DKSRCTracker',       # tracker name
            is_deterministic=True    # stochastic (False) or deterministic (True)
        )
        self.feat = Encoder().to('cuda:0').eval()
        enc_file = 'checkpoints/512encoder79.wgt'
        self.feat.load_state_dict(torch.load(enc_file, map_location="cpu"))
        self.transf = T.Compose([
            T.ToTensor(),
        ])
        self.ex = futures.ThreadPoolExecutor()

    def init(self, image, box):
        image = np.array(image)
        # PIL: w, h
        # Opencv: h, w, c
        self.init_box = box
        h, w, _ = image.shape
        self.im_sz = (w, h)
        # Init dictionary matrix
        self.box = box

        pos_rects = self.g_pos_s(box, p.p_ini)
        pos_samples = crop_samples(image, p.sz_T, pos_rects, self.transf)

        neg_rects = self.g_neg_s(box)
        neg_samples = crop_samples(image, p.sz_T, neg_rects, self.transf)

        self.samples = torch.vstack((pos_samples, neg_samples))
        trainX = self.feat(self.samples.cuda())[0].T
        trainX = F.normalize(trainX, p=2, dim=0)
        if p.debug:
            viz.images(self.samples, padding=1, win='Samples')
        torch.cuda.empty_cache()
        self.trainY = np.hstack((np.zeros([p.p_ini], dtype=np.uint8), np.ones([p.n_ini], dtype=np.uint8)))
        self.trainY = torch.from_numpy(self.trainY).cuda()

        self.feature_idx = trace_r.trace_ratio(trainX.T.cpu().numpy(), self.trainY.cpu().numpy(), p.feat_dim)
        # self.feature_idx = self.feature_idx[feature_score > 0]
        self.trainX = trainX#[self.feature_idx, :]
        self.inifeat = self.trainX
        self.fail_flag = 0

        sigma = 1
        self.kfncs = lambda x, y: torch.exp(-sigma*torch._euclidean_dist(x.T, y.T))

        self.recSp = p.recSp               # limitation of sparsity
        self.A = DKSRC(self.trainX, self.trainY, [0, 1], [p.dict_sz,p.dict_sz], p.recSp, 1, self.kfncs)
        torch.cuda.empty_cache()
        self.importance = torch.zeros([2, p.dict_sz]).cuda()      # divided into positive importance and negative importance

        # Parameters
        self.upp = []       # online update pos samples
        self.upn = []       # online update neg samples
        if p.debug == True:
            self.ps = []
            self.ns = []

        self.count = 0
        self.bbreg = BBRegressorNN(self.im_sz)
        # self.bbreg = BBRegressor(self.im_sz)
        # reg_bbox = self.g_pos_s(self.box, num=1500)
        # reg_feat = crop_samples(image, p.sz_T, reg_bbox, self.transf).cuda()
        # reg_feat = self.feat(reg_feat)[0].T
        # reg_feat = F.normalize(reg_feat, p=2, dim=0)#[self.feature_idx, :]
        # self.bbreg.train(reg_feat.T, reg_bbox, box)

    def update(self, image):
        t = time()
        image = np.array(image)
        rects = self.g_cand_s(self.box)
        res, success = self.eval_online(image, rects)

        if success:
            self.box = res
            self.sample_online(image, self.box)

        self.count = self.count + 1

        if self.count % p.up_int == 0:
            self.update_online()

        torch.cuda.empty_cache()

        tt = time()-t
        print("Frame:{0}, Time:{1}, Success: {2}".format(self.count, round(tt, 5), success))
        return self.box

    def sample_online(self, image, res):
        pn = p.p_up_num//2+1
        nn = p.n_up_num//2+1
        pos_rects = self.g_pos_s(res, p.p_ini)[:pn, :]
        pos_samples = crop_samples(image, p.sz_T, pos_rects, self.transf)
        neg_rects = self.g_neg_s(res)[:nn, :]
        neg_samples = crop_samples(image, p.sz_T, neg_rects, self.transf)
        samples = torch.vstack((pos_samples, neg_samples))

        trainX = self.feat(samples.cuda())[0].T
        trainX = F.normalize(trainX, p=2, dim=0)#[self.feature_idx, :]
        trainY = torch.hstack((torch.zeros([pn], dtype=torch.uint8), torch.ones([nn], dtype=torch.uint8)))

        if p.debug == True:
            self.ps.append(pos_samples)
            self.ns.append(neg_samples)

        self.upp.append(trainX[:, trainY==0])
        self.upn.append(trainX[:, trainY==1])

    def g_neg_s(self, box, num=p.n_ini):
        neg_rects = np.concatenate([
            SampleGenerator('uniform', self.im_sz, 1, 1.6)(
                        box, int(num * 0.5), [0, 0.5]),
            SampleGenerator('whole', self.im_sz)(
            box, int(num * 0.5), [0, 0.5])
        ])
        return neg_rects
    def g_pos_s(self, box, num=p.p_ini):
        return SampleGenerator('gaussian', self.im_sz, 0.1, 1.3)(
            box, num, [0.7, 1])
    def g_cand_s(self, box, num=p.n_sample, trans=0.6, scale=1.1):
        return SampleGenerator('gaussian', self.im_sz, trans, scale)(
            box, num
        )

    def update_online(self):
        if self.fail_flag >= p.up_int and len(self.upp) < 10:
            self.upp.append(self.trainX[:, :p.p_ini])
            self.upn.append(self.trainX[:, p.p_ini:])
            self.fail_flag = 0

        if len(self.upp) >= 1 and len(self.upn) >= 1:
            mask_pidx = torch.topk(self.importance[0, :], p.p_up_num,
                                   largest=True).indices  # Three atoms will be updated every update_interval frames.
            upp = torch.hstack(self.upp)
            upn = torch.hstack(self.upn)
            pmm = np.random.permutation(upp.shape[1])[:p.p_up_num]
            self.trainX[:, :p.p_ini][:, mask_pidx] = upp[:, pmm]# upp[:, -p.p_up_num:]#
            mask_nidx = p.p_ini + np.random.permutation(p.n_ini)[:p.n_up_num]
            nmm = np.random.permutation(upn.shape[1])[:p.n_up_num]
            self.trainX[:, mask_nidx] = upn[:, nmm]
            self.A = DKSRC(self.trainX, self.trainY, [0, 1], [p.dict_sz, p.dict_sz], p.recSp, 1,
                            self.kfncs)

            if p.debug == True:
                pss = torch.vstack(self.ps)
                nss = torch.vstack(self.ns)
                self.samples[:p.p_ini, :, :, :][mask_pidx, :, :, :] = pss[-p.p_up_num:, :, :, :]
                self.samples[mask_nidx, :, :, :] = nss[nmm, :, :, :]
                viz.images(self.samples, padding=1, win='Samples')


        self.importance = torch.zeros([2, p.dict_sz]).cuda()
        self.upp = []
        self.upn = []

    def eval_online(self, image, rects):
        samples_ = crop_samples(image, p.sz_T, rects, self.transf)
        if p.debug:
            viz.images(samples_, padding=1, win='Candidates')
        z = self.feat(samples_.cuda())[0].T
        z = F.normalize(z, p=2, dim=0)#[self.feature_idx, :]

        r = torch.zeros([2, z.shape[1]]).cuda()
        labels = [0, 1]

        def task(k):
            Xi = self.trainX[:, self.trainY == labels[k]]
            r[k, :], self.importance[k, :, None] = evalKKSVD(z, Xi, self.A[k], self.recSp, self.kfncs,
                                                             importance=self.importance[k, :, None])
        for _ in self.ex.map(task, range(2)):
            pass

        testYhat = torch.zeros([2, r.shape[1]]).cuda()
        for i in range(r.shape[1]):
            testYhat[0][i] = torch.min(r[:, i])
            testYhat[1][i] = torch.argmin(r[:, i])

        testYhat = testYhat.detach().cpu().numpy()
        res_ = rects[testYhat[1, :] == 0, :]
        scores = testYhat[0, testYhat[1, :] == 0]
        k = min(len(scores), 5)
        idx = torch.topk(torch.from_numpy(scores), k, largest=False).indices.numpy()
        res = np.mean(res_[idx, :], axis=0)

        success = False
        if len(idx) != 0:
            success = True
            if len(idx) == 1:
                idx = idx.repeat(2)
            if p.debug == True:
                viz.images(samples_[testYhat[1, :] == 0, :, :, :][idx, :, :, :], padding=1, win='TopK')
            ## BBreg
            # bbreg_bbox = res_[idx, :]# np.concatenate([rp, rn])#
            # bbreg_feat = crop_samples(image, p.sz_T, bbreg_bbox, self.transf)
            # bbreg_feat = self.feat(bbreg_feat.cuda())[0].T#[self.feature_idx, :]
            # bbreg_feat = F.normalize(bbreg_feat, p=2, dim=0)
            # res = self.bbreg.predict(bbreg_feat.T, bbreg_bbox)
            # res = np.mean(res, axis=0)
            if res[2]/res[3]<0.2 or res[3]/res[2]<0.2:
                res = self.box
        else:
            self.fail_flag += 1
        return res, success
