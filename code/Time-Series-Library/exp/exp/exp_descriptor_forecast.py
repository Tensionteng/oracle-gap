from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.compute_descriptor_stats import ensure_descriptor_stats
from utils.descriptor_labels import compute_descriptor_targets
from models.HCANComponents import (fit_bin_edges, produce_labels, evidence_ce_loss,
                                   symmetric_kl)
from utils.dtw_metric import dtw, accelerated_dtw
from utils.region_focal import RegionFocalLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class FreDFLoss(nn.Module):
    """FreDF task loss (ICLR 2025, arXiv:2402.02399): MSE in the rFFT domain.

    pred / true: [B, L, C]. The series are rFFT-transformed along the time
    dimension and compared by the squared Hermitian magnitude of the complex
    difference (= summed real/imaginary squared errors), matching the paper's
    frequency loss. The official repo's one-liner uses the unsquared magnitude
    mean, (rfft(pred) - rfft(true)).abs().mean(); both are the same comparison
    up to the exponent, see github.com/Master-PLC/FreDF.
    """

    def forward(self, pred, true):
        diff = torch.fft.rfft(pred, dim=1) - torch.fft.rfft(true, dim=1)
        return (diff.real ** 2 + diff.imag ** 2).mean()


class Exp_Descriptor_Forecast(Exp_Long_Term_Forecast):
    """Long-term forecasting with an auxiliary future-descriptor head
    (--aux_head desc / valuemtp / hcan) and/or the FreDF frequency-domain task
    loss (--task_loss fredf).

    Training adds aux_weight * aux_loss to the task loss; descriptor targets
    are computed online from the last pred_len steps of batch_y with vectorized
    torch ops on the training device. Validation and test use the task loss
    only and follow the stock protocol (inherited vali; test() is a copy of the
    stock protocol plus pred_dumps saving).
    """

    # stock backbone -> hidden-exposing variant used when aux_head != 'none'
    DESC_BACKBONE = {
        'PatchTST': 'PatchSTDesc',
        'iTransformer': 'iTransformerDesc',
        'DLinear': 'DLinearDesc',
    }
    HCAN_BACKBONE = {
        'PatchTST': 'PatchSTHcan',
        'iTransformer': 'iTransformerHcan',
    }

    def __init__(self, args):
        super(Exp_Descriptor_Forecast, self).__init__(args)
        self.aux_head = getattr(args, 'aux_head', 'none')
        self.aux_weight = getattr(args, 'aux_weight', 1.0)
        self.desc_k = getattr(args, 'desc_k', 24)
        self.desc_scales = set(getattr(args, 'desc_scales', 'near,mid,far').split(','))
        assert self.desc_scales <= {'near', 'mid', 'far'}, \
            'desc_scales must be a comma-separated subset of near,mid,far'
        self.desc_stats = None
        if self.aux_head == 'desc':
            stats = ensure_descriptor_stats(args)
            self.desc_stats = {k: torch.from_numpy(np.asarray(v)).float().to(self.device)
                               for k, v in stats.items() if v.dtype != object}
        elif self.aux_head == 'hcan':
            # per-channel quantile bin edges of the scaled train values
            # (HCAN's Group_helper), computed once at init
            train_data, _ = data_provider(args, flag='train')
            values = np.asarray(train_data.data_y)
            self.hcan_edges_c = fit_bin_edges(values, args.num_coarse).to(self.device)
            self.hcan_edges_f = fit_bin_edges(values, args.num_fine).to(self.device)
            print('hcan bin edges fitted on train values {} (coarse={}, fine={})'.format(
                values.shape, args.num_coarse, args.num_fine))

    def _build_model(self):
        model_name = self.args.model
        if getattr(self.args, 'aux_head', 'none') != 'none':
            backbone_map = self.HCAN_BACKBONE if self.args.aux_head == 'hcan' else self.DESC_BACKBONE
            model_name = backbone_map.get(model_name, model_name)
            if self.args.aux_head == 'hcan' and model_name not in self.HCAN_BACKBONE.values():
                raise ValueError('aux_head=hcan is only implemented for {} (got {})'.format(
                    list(self.HCAN_BACKBONE), self.args.model))
        model = self.model_dict[model_name](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _select_criterion(self):
        task_loss = getattr(self.args, 'task_loss', 'mse')
        if task_loss == 'fredf':
            criterion = FreDFLoss()
        elif task_loss == 'regionfocal':
            criterion = RegionFocalLoss(
                au=self.args.rf_au, ao=self.args.rf_ao, aw=self.args.rf_aw,
                lambda_base=self.args.rf_lambda_base, anchor=self.args.rf_anchor,
                soft=self.args.rf_soft, detach=self.args.rf_detach,
                seq_len=self.args.seq_len, freq=self.args.freq)
        else:
            criterion = nn.MSELoss()
        if task_loss != 'mse':
            print('task criterion: {}'.format(criterion.__class__.__name__))
        return criterion

    def _select_eval_criterion(self):
        # regionfocal is a train-time reweighting objective: val/test (incl.
        # early stopping) must use plain MSE to stay on the stock protocol.
        # fredf keeps its own loss for vali (its established behavior).
        if getattr(self.args, 'task_loss', 'mse') == 'regionfocal':
            return nn.MSELoss()
        return self._select_criterion()

    def _aux_loss(self, aux, batch_y, batch_x, step=0):
        """Auxiliary loss. batch_y: [B, pred_len, C] (f_dim-sliced, on device);
        batch_x: [B, seq_len, C] raw batch (on device)."""
        if self.aux_head == 'desc':
            channel_mode = getattr(self.args, 'desc_mode', 'pooled') == 'channel'
            targets = compute_descriptor_targets(batch_y, self.desc_k, self.desc_stats,
                                                 reduce=None if channel_mode else 'mean',
                                                 vol_log=bool(getattr(self.args, 'vol_log', 0)),
                                                 vol_ms=bool(getattr(self.args, 'vol_ms', 0)),
                                                 vol_qr=bool(getattr(self.args, 'vol_qr', 0)))
            if getattr(self.args, 'desc_shuffle', 0):
                # negative control: one shared batch-dim permutation for all
                # descriptor targets, severing the sample<->label link while
                # keeping the joint descriptor distribution intact
                perm = torch.randperm(batch_y.shape[0], device=batch_y.device)
                targets = {k: v[perm] for k, v in targets.items()}
            n_bins = aux['drift'].shape[-1]
            loss = None
            if 'near' in self.desc_scales:
                loss = F.binary_cross_entropy_with_logits(aux['cp_prob'], targets['cp_prob'])
                loss = loss + F.smooth_l1_loss(aux['cp_pos'], targets['cp_pos'])
            if 'mid' in self.desc_scales:
                # cross_entropy over flattened (batch x channel) positions
                if getattr(self.args, 'vol_qr', 0):
                    vol_term = F.mse_loss(aux['vol_qr'], targets['vol_log'])
                else:
                    vol_term = F.cross_entropy(aux['vol'].reshape(-1, n_bins),
                                               targets['vol_cls'].reshape(-1))
                mid = F.cross_entropy(aux['drift'].reshape(-1, n_bins), targets['drift_cls'].reshape(-1)) \
                    + vol_term \
                    + F.cross_entropy(aux['slope'].reshape(-1, n_bins), targets['slope_cls'].reshape(-1))
                if getattr(self.args, 'vol_ms', 0):
                    mid = mid + F.cross_entropy(aux['vol_near'].reshape(-1, n_bins),
                                                targets['vol_near_cls'].reshape(-1))
                loss = mid if loss is None else loss + mid
            if 'far' in self.desc_scales:
                far = F.mse_loss(aux['spectral'], targets['spectral'])
                loss = far if loss is None else loss + far
            if loss is None:
                raise ValueError('desc_scales selects no loss terms: {}'.format(self.args.desc_scales))
            return loss
        elif self.aux_head == 'valuemtp':
            # value-level MTP control: predict the instance-normalized future
            # values, i.e. the same target the backbone head predicts before
            # de-normalization
            f_dim = -1 if self.args.features == 'MS' else 0
            x = batch_x[:, :, f_dim:]
            means = x.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            target = ((batch_y - means) / stdev).reshape(batch_y.shape[0], -1)
            return F.mse_loss(aux, target)
        elif self.aux_head == 'hcan':
            return self._hcan_loss(aux, batch_y, step)
        return None

    def _hcan_loss(self, aux, batch_y, step):
        """HCAN auxiliary loss (paper defaults: lambda_cls=0.01, lambda_reg=1,
        lambda_acl=1; the direct-path MSE is our task loss, lambda_direct=1).

        aux: HCANHead outputs, each [B, pred_len, C, K]; batch_y:
        [B, pred_len, C] (f_dim-sliced, on device). The evidential KL annealing
        mirrors the official code: coefficient min(1, step / train_epochs) with
        step the within-epoch batch index."""
        f_dim = -1 if self.args.features == 'MS' else 0
        edges_c = self.hcan_edges_c[f_dim:, :]
        edges_f = self.hcan_edges_f[f_dim:, :]
        label_c, delta_c = produce_labels(batch_y, edges_c)
        label_f, delta_f = produce_labels(batch_y, edges_f)
        nc, nf = self.args.num_coarse, self.args.num_fine

        # HCL (ACL in the official code): coarse vs pairwise-merged fine logits
        fine_merged = aux['fine_logit'].view(*aux['fine_logit'].shape[:-1], nc, 2).mean(dim=-1)
        loss_acl = symmetric_kl(F.softmax(aux['coarse_logit'], dim=-1),
                                F.softmax(fine_merged, dim=-1))

        anneal_step = max(self.args.train_epochs, 1)
        loss_edl_c = evidence_ce_loss(label_c.reshape(-1), aux['coarse_logit'].reshape(-1, nc),
                                      nc, step, anneal_step)
        mask_c = delta_c >= 0
        loss_reg_c = F.mse_loss(aux['coarse_delta'][mask_c], delta_c[mask_c]) if mask_c.any() \
            else aux['coarse_delta'].sum() * 0.0
        loss_edl_f = evidence_ce_loss(label_f.reshape(-1), aux['fine_logit'].reshape(-1, nf),
                                      nf, step, anneal_step)
        mask_f = delta_f >= 0
        loss_reg_f = F.mse_loss(aux['fine_delta'][mask_f], delta_f[mask_f]) if mask_f.any() \
            else aux['fine_delta'].sum() * 0.0

        return self.args.lambda_cls * (loss_edl_c + loss_edl_f) \
            + self.args.lambda_reg * (loss_reg_c + loss_reg_f) \
            + self.args.lambda_acl * loss_acl

    def _forward_loss(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y, criterion,
                      step=0):
        if self.aux_head != 'none':
            outputs, aux = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                      return_hidden=True)
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            aux = None

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        if getattr(self.args, 'task_loss', 'mse') == 'regionfocal':
            anchor = criterion.compute_anchor(batch_x[:, :, f_dim:], self.args.pred_len)
            gate = None
            if getattr(self.args, 'rf_desc_gate', 0) and self.aux_head == 'desc' and aux is not None:
                # descriptor-gated regionfocal: alpha_U(window) = rf_au *
                # sigmoid(cp_prob logit); the gate is detached - no gradient
                # flows back into the descriptor head through the main loss
                gate = torch.sigmoid(aux['cp_prob']).detach()
            loss = criterion(outputs, batch_y, anchor=anchor, gate=gate)
        else:
            loss = criterion(outputs, batch_y)
        if aux is not None:
            loss = loss + self.aux_weight * self._aux_loss(aux, batch_y, batch_x, step)
        return loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        eval_criterion = self._select_eval_criterion()
        if getattr(self.args, 'task_loss', 'mse') == 'regionfocal':
            print('eval criterion (val/test/early-stopping): MSELoss (forced for regionfocal)')

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        loss = self._forward_loss(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                  batch_y, criterion, step=i)
                        train_loss.append(loss.item())
                else:
                    loss = self._forward_loss(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                              batch_y, criterion, step=i)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, eval_criterion)
            test_loss = self.vali(test_data, test_loader, eval_criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        # same protocol as the stock Exp_Long_Term_Forecast.test; the only
        # addition is saving per-window predictions / truths / window indices
        # to ./pred_dumps/<setting>/ when --save_pred is on (default 1)
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # per-window dump for mechanism analyses (probe / gain localization).
        # test loader is sequential, so window_index[i] = i = start offset of
        # the input window inside the test split (future window starts at
        # seq_len + i); see data_provider/data_loader.py __getitem__.
        if getattr(self.args, 'save_pred', 1):
            dump_dir = os.path.join('./pred_dumps/', setting)
            if not os.path.exists(dump_dir):
                os.makedirs(dump_dir)
            np.save(os.path.join(dump_dir, 'pred.npy'), preds)
            np.save(os.path.join(dump_dir, 'true.npy'), trues)
            np.save(os.path.join(dump_dir, 'window_index.npy'), np.arange(preds.shape[0]))
            print('pred dump saved to {}'.format(dump_dir))

        return
