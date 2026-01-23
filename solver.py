import pandas as pd

from model.RevIN import RevIN
from model.ts2image import *
import time
from utils.utils import *
from model.twoDTSAD import twoDTSAD
from data_factory.data_loader import get_loader_segment
from metrics.metrics import *
import warnings
import torch
import torch.nn as nn
import numpy as np
warnings.filterwarnings('ignore')


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        # get loaders (now NeighborSegLoader returns 'current_value')
        self.train_loader = get_loader_segment(self.index, 'dataset/' + self.data_path,
                                               batch_size=self.batch_size,
                                               win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset,
                                               local_size=self.local_size,
                                               global_size=self.global_size,
                                               global_stride=getattr(self, 'global_stride', None),
                                               num_workers=getattr(self, 'num_workers', 8))

        self.vali_loader = get_loader_segment(self.index, 'dataset/' + self.data_path,
                                             batch_size=self.batch_size,
                                             win_size=self.win_size,
                                             mode='val',
                                             dataset=self.dataset,
                                             local_size=self.local_size,
                                             global_size=self.global_size,
                                             global_stride=getattr(self, 'global_stride', None),
                                             num_workers=getattr(self, 'num_workers', 8))

        self.test_loader = get_loader_segment(self.index, 'dataset/' + self.data_path,
                                              batch_size=self.batch_size,
                                              win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset,
                                              local_size=self.local_size,
                                              global_size=self.global_size,
                                              global_stride=getattr(self, 'global_stride', None),
                                              num_workers=getattr(self, 'num_workers', 8))

        self.thre_loader = get_loader_segment(self.index, 'dataset/' + self.data_path,
                                              batch_size=self.batch_size,
                                              win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset,
                                              local_size=self.local_size,
                                              global_size=self.global_size,
                                              global_stride=getattr(self, 'global_stride', None),
                                              num_workers=getattr(self, 'num_workers', 8))
        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()
            self.criterion_keep = nn.MSELoss(reduction='none')

    def build_model(self):
        self.model = twoDTSAD(win_size=self.win_size, d_model=self.d_model,
                             local_size=self.local_size, global_size=self.global_size,
                             channel=self.input_c)

        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        print('parameters:', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def train(self):
        time_now = time.time()
        train_steps = len(self.train_loader)
        for epoch in range(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            for it, batch in enumerate(self.train_loader):
                # batch is a dict returned by our NeighborSegLoader
                # local_window: (B, W, C, L)
                # global_window:(B, W, C, G)
                # local_delay:  (B, W, C, L, L)
                # global_delay: (B, W, C, G, G)
                # local_gaf/global_gaf same dims
                # labels: (B, W)
                # current_value: (B, W, C)  <-- NEW target
                self.optimizer.zero_grad()
                iter_count += 1

                # Move to device
                local_window = batch["local_window"].float().to(self.device)    # (B, W, C, L)
                global_window = batch["global_window"].float().to(self.device)  # (B, W, C, G)
                local_delay = batch["local_delay"].float().to(self.device)      # (B, W, C, L, L)
                global_delay = batch["global_delay"].float().to(self.device)    # (B, W, C, G, G)
                local_gaf = batch["local_gaf"].float().to(self.device)
                global_gaf = batch["global_gaf"].float().to(self.device)
                labels = batch.get("labels", None)
                if labels is not None:
                    labels = labels.to(self.device)

                # NEW: get true current values (ground truth) for centers in this batch
                current_value = batch.get("current_value", None)
                if current_value is not None:
                    current_value = current_value.float().to(self.device)  # (B, W, C)

                B, W, C, L1 = local_window.shape
                _, _, _, G1 = global_window.shape

                # Flatten per-center tensors to match model expected (B*W, C, L, L) shapes
                local_delay_flat = local_delay.reshape(B * W, C, L1, L1)
                global_delay_flat = global_delay.reshape(B * W, C, G1, G1)
                local_gaf_flat = local_gaf.reshape(B * W, C, L1, L1)
                global_gaf_flat = global_gaf.reshape(B * W, C, G1, G1)

                # call model: pass B (batch size), window length W as the 'sequence length' L parameter, and M = C
                xlocal, xglobal, pred_delay_local, pred_delay_global, pred_gaf_local, pred_gaf_global = \
                    self.model(B, W, C, local_delay_flat, global_delay_flat, local_gaf_flat, global_gaf_flat)

                x_tgt = current_value  # (B, W, C)

                # compute losses:
                local_loss = self.criterion(xlocal, x_tgt)
                global_loss = self.criterion(xglobal, x_tgt)
                lo_gl_loss = self.criterion(xlocal, xglobal)

                # for predicted graphs (delay/gaf) predictions are of shape (B*W, C, L, L) etc; compare to corresponding inputs:
                pred_delay_local_loss = self.criterion(pred_delay_local, local_delay_flat)
                pred_delay_global_loss = self.criterion(pred_delay_global, global_delay_flat)
                pred_gaf_local_loss = self.criterion(pred_gaf_local, local_gaf_flat)
                pred_gaf_global_loss = self.criterion(pred_gaf_global, global_gaf_flat)

                loss = (local_loss + global_loss + lo_gl_loss + pred_delay_local_loss + pred_delay_global_loss +
                            pred_gaf_local_loss + pred_gaf_global_loss) / 7.0

                if (it + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - it)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                # gradient clipping optional
                if getattr(self, 'grad_clip', None) is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            print("Epoch: {0}, Cost time: {1:.3f}s ".format(epoch + 1, time.time() - epoch_time))
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        op = "contr"
        attens_energy = []

        # ================= (1) train stat =================
        for i, batch in enumerate(self.train_loader):
            local_window = batch["local_window"].float().to(self.device)
            global_window = batch["global_window"].float().to(self.device)
            local_delay = batch["local_delay"].float().to(self.device)
            global_delay = batch["global_delay"].float().to(self.device)
            local_gaf = batch["local_gaf"].float().to(self.device)
            global_gaf = batch["global_gaf"].float().to(self.device)

            B, W, C, L1 = local_window.shape
            _, _, _, G1 = global_window.shape

            local_delay_flat = local_delay.reshape(B * W, C, L1, L1)
            global_delay_flat = global_delay.reshape(B * W, C, G1, G1)
            local_gaf_flat = local_gaf.reshape(B * W, C, L1, L1)
            global_gaf_flat = global_gaf.reshape(B * W, C, G1, G1)

            xlocal, xglobal, pred_delay_local, pred_delay_global, pred_gaf_local, pred_gaf_global = \
                self.model(B, W, C,
                           local_delay_flat, global_delay_flat,
                           local_gaf_flat, global_gaf_flat)

            lo_gl_loss = self.criterion_keep(xlocal, xglobal)
            lo_gl_loss, _ = torch.max(lo_gl_loss, dim=-1)

            if self.Ablation == 'Time_Only':
                loss_metric = lo_gl_loss

            metric = torch.softmax(loss_metric, dim=-1)
            attens_energy.append(metric.detach().cpu().numpy())

        train_energy = np.concatenate(attens_energy, axis=0).reshape(-1)

        # ================= (2) threshold =================
        attens_energy = []
        for i, batch in enumerate(self.thre_loader):
            local_window = batch["local_window"].float().to(self.device)
            global_window = batch["global_window"].float().to(self.device)
            local_delay = batch["local_delay"].float().to(self.device)
            global_delay = batch["global_delay"].float().to(self.device)
            local_gaf = batch["local_gaf"].float().to(self.device)
            global_gaf = batch["global_gaf"].float().to(self.device)

            B, W, C, L1 = local_window.shape
            _, _, _, G1 = global_window.shape

            local_delay_flat = local_delay.reshape(B * W, C, L1, L1)
            global_delay_flat = global_delay.reshape(B * W, C, G1, G1)
            local_gaf_flat = local_gaf.reshape(B * W, C, L1, L1)
            global_gaf_flat = global_gaf.reshape(B * W, C, G1, G1)

            xlocal, xglobal, pred_delay_local, pred_delay_global, pred_gaf_local, pred_gaf_global = \
                self.model(B, W, C,
                           local_delay_flat, global_delay_flat,
                           local_gaf_flat, global_gaf_flat)

            lo_gl_loss = self.criterion_keep(xlocal, xglobal)
            topk_value, _ = torch.topk(lo_gl_loss, k=self.topk, dim=-1)
            lo_gl_loss = torch.mean(topk_value, dim=-1)

            if self.Ablation == 'Time_Only':
                loss_metric = lo_gl_loss

            metric = torch.softmax(loss_metric, dim=-1)
            attens_energy.append(metric.detach().cpu().numpy())

        test_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        thresh = np.percentile(
            np.concatenate([train_energy, test_energy]),
            100 - self.anormly_ratio)

        print("Threshold :", thresh)

        # ================= (3) evaluation =================
        import matplotlib.pyplot as plt

        test_labels_list = []
        attens_energy = []

        mid_iter = 2
        saved_flag = False

        for i, batch in enumerate(self.thre_loader):
            local_window = batch["local_window"].float().to(self.device)
            global_window = batch["global_window"].float().to(self.device)
            local_delay = batch["local_delay"].float().to(self.device)
            global_delay = batch["global_delay"].float().to(self.device)
            local_gaf = batch["local_gaf"].float().to(self.device)
            global_gaf = batch["global_gaf"].float().to(self.device)

            labels = batch.get("labels", None)
            if labels is not None:
                labels = labels.numpy()

            B, W, C, L1 = local_window.shape
            _, _, _, G1 = global_window.shape

            local_delay_flat = local_delay.reshape(B * W, C, L1, L1)
            global_delay_flat = global_delay.reshape(B * W, C, G1, G1)
            local_gaf_flat = local_gaf.reshape(B * W, C, L1, L1)
            global_gaf_flat = global_gaf.reshape(B * W, C, G1, G1)

            xlocal, xglobal, pred_delay_local, pred_delay_global, pred_gaf_local, pred_gaf_global = \
                self.model(B, W, C,
                           local_delay_flat, global_delay_flat,
                           local_gaf_flat, global_gaf_flat)

            lo_gl_loss = self.criterion_keep(xlocal, xglobal)
            lo_gl_loss, _ = torch.max(lo_gl_loss, dim=-1)

            loss_metric = lo_gl_loss

            metric = torch.softmax(loss_metric, dim=-1)
            attens_energy.append(metric.detach().cpu().numpy())

            if labels is not None:
                test_labels_list.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels_list, axis=0).reshape(-1)

        pred = (attens_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        scores_simple = combine_all_evaluation_scores(pred, gt, attens_energy)
        for k, v in scores_simple.items():
            print(f"{k:20}: {v:.4f}")

        # post-process contiguous anomaly labeling as original
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        from sklearn.metrics import precision_recall_fscore_support, accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))

        return accuracy, precision, recall, f_score
