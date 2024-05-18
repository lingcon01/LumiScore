import torch as th
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from torch.distributions import Normal
from torch_scatter import scatter_add
from sklearn import metrics
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_recall_curve, auc
from scipy.stats import pearsonr, spearmanr
import tqdm
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from math import sqrt

class EarlyStopping(object):
    """Early stop tracker
	
    Save model checkpoint when observing a performance improvement on
    the validation set and early stop if improvement has not been
    observed for a particular number of epochs.
	
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
        If ``metric`` is not None, then mode will be determined
        automatically from that.
    patience : int
        The early stopping will happen if we do not observe performance
        improvement for ``patience`` consecutive epochs.
    filename : str or None
        Filename for storing the model checkpoint. If not specified,
        we will automatically generate a file starting with ``early_stop``
        based on the current time.
    metric : str or None
        A metric name that can be used to identify if a higher value is
        better, or vice versa. Default to None. Valid options include:
        ``'r2'``, ``'mae'``, ``'rmse'``, ``'roc_auc_score'``.
	
    Examples
    --------
    Below gives a demo for a fake training process.
	
    >>> import torch
    >>> import torch.nn as nn
    >>> from torch.nn import MSELoss
    >>> from torch.optim import Adam
    >>> from dgllife.utils import EarlyStopping
	
    >>> model = nn.Linear(1, 1)
    >>> criterion = MSELoss()
    >>> # For MSE, the lower, the better
    >>> stopper = EarlyStopping(mode='lower', filename='test.pth')
    >>> optimizer = Adam(params=model.parameters(), lr=1e-3)
	
    >>> for epoch in range(1000):
    >>>     x = torch.randn(1, 1) # Fake input
    >>>     y = torch.randn(1, 1) # Fake label
    >>>     pred = model(x)
    >>>     loss = criterion(y, pred)
    >>>     optimizer.zero_grad()
    >>>     loss.backward()
    >>>     optimizer.step()
    >>>     early_stop = stopper.step(loss.detach().data, model)
    >>>     if early_stop:
    >>>         break
	
    >>> # Load the final parameters saved by the model
    >>> stopper.load_checkpoint(model)
    """

    def __init__(self, mode='higher', patience=10, filename=None, metric=None):
        if filename is None:
            # dt = datetime.datetime.now()
            filename = 'early_stop.pth'

        if metric is not None:
            assert metric in ['rp', 'rs', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score'], \
                "Expect metric to be 'rp' or 'rs' or 'mae' or " \
                "'rmse' or 'roc_auc_score', got {}".format(metric)
            if metric in ['rp', 'rs', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = {"fep": 0, "derivate": 0}
        self.timestep = 0
        self.filename = filename
        self.best_score = {}
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        """Check if the new score is higher than the previous best score.
	
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
	
        Returns
        -------
        bool
            Whether the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """Check if the new score is lower than the previous best score.
	
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.

        Returns
        -------
        bool
            Whether the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score_dict, model):
        """Update based on a new score.
	
        The new score is typically model performance on the validation set
        for a new epoch.

        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.
	
        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        self.timestep += 1

        for target, score in score_dict.items():
            filename = self.filename.split('.')[0] + '_' + target + '.pth'
            if target not in self.best_score:
                self.best_score[target] = score
                self.save_checkpoint(model, filename)

            elif self._check(score, self.best_score[target]):
                self.best_score[target] = score
                self.save_checkpoint(model, filename)
                self.counter[target] = 0
            else:
                self.counter[target] += 1
                print(
                    f'EarlyStopping counter: {self.counter[target]} out of {self.patience}')
                if self.counter[target] >= self.patience:
                    self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model, ckpt_path):
        '''Saves model when the metric on the validation set gets improved.
	
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        th.save({'model_state_dict': model.state_dict(),
                 'timestep': self.timestep}, ckpt_path)

    def load_checkpoint(self, model):
        '''Load the latest checkpoint
	
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        model.load_state_dict(th.load(self.filename)['model_state_dict'])


def screen_train_epoch(epoch, model, data_loader, optimizer, mdn_weight=1.0, affi_weight=1.0, aux_weight=0.001,
                       dist_threhold=None, device='cpu'):
    # loss_fn = nn.MSELoss()
    loss_fn = th.nn.CrossEntropyLoss(label_smoothing=0.0).to(device)
    model.train()
    total_loss = []
    probs = []
    true = []
    for batch_id, batch_data in tqdm(enumerate(data_loader)):
        pdbids, bgp, bgl, labels = batch_data
        # print(f"pdbids is {pdbids}")
        # print(f"bgp is {bgp}")
        # print(f"bgl is {bgl}")
        # print(f"labels is {labels}")
        bgl, bgp = bgl.to(device), bgp.to(device)

        _, _, pred = model(bgl, bgp, train=True)

        # print(f"pred size is {pred}\tlabels size is {labels}")
        loss = loss_fn(pred, labels.long().to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs.append(pred.detach())
        true.append(labels.detach())
        # total_loss += loss.item() * batch.unique().size(0)
        total_loss.append(loss.detach())

        del loss, labels, pred, bgl, bgp, batch_data, pdbids
        th.cuda.empty_cache()

        # for name, param in model.named_parameters():
        #     print(name, param)

    return total_loss, probs, true


def screen_eval_epoch(model, data_loader, mdn_weight=1.0, affi_weight=1.0, aux_weight=0.001,
                      dist_threhold=None, device='cpu'):
    # loss_fn = nn.MSELoss()
    loss_fn = th.nn.CrossEntropyLoss(label_smoothing=0.0).to(device)
    model.eval()
    total_loss = []
    probs = []
    true = []

    with th.no_grad():
        for batch_id, batch_data in tqdm(enumerate(data_loader)):
            pdbids, bgp, bgl, labels = batch_data
            bgl, bgp = bgl.to(device), bgp.to(device)

            _, _, pred = model(bgl, bgp, train=False)

            loss = loss_fn(pred, labels.long().to(device))

            pred = th.softmax(pred, dim=-1)[:, 1]

            probs.append(pred.detach().cpu().numpy())
            true.append(labels.detach().cpu().numpy())
            # total_loss += loss.item() * batch.unique().size(0)
            total_loss.append(loss.detach())

            del loss, labels, pred, bgl, bgp, batch_data, pdbids
            th.cuda.empty_cache()

    return total_loss, probs, true


def run_augment(epoch, model, score_iter, dock_iter, cross_iter, random_iter, optimizer, device='cpu'):
    loss_fn = nn.MSELoss()
    model.train()

    probs = []
    true = []
    total_loss = 0.0
    i_batch = 0
    while True:
        sample = next(score_iter, None)
        if sample is None:
            break

        pdbids, bgp, bgl, labels = sample
        bgl, bgp = bgl.to(device), bgp.to(device)

        vdw_pred, vdw_radius, pred, mdn, mdn_pred, _ = model(bgl, bgp, train=True)
        pred = pred.squeeze(-1)
        mdn_pred = mdn_pred.squeeze(-1)

        labels = labels.float().to(device)
        vdw = labels * 0.8

        loss1 = abs(loss_fn(vdw_pred, vdw) - 0.5)
        loss2 = loss_fn(pred, labels)
        loss_mdn = th.corrcoef(th.stack([mdn_pred, labels]))[1, 0].item()

        loss_score = loss2 + (mdn - loss_mdn) * 0.5 + 0.05 * loss1

        # loss_docking
        # loss_docking = torch.zeros((1,))
        # keys_docking = []
        sample_docking = next(dock_iter, None)
        # print("docking successful")
        if sample_docking is None:
            break

        pdbids_docking, bgp_docking, bgl_docking, labels_docking = sample_docking
        bgl_docking, bgp_docking = bgl_docking.to(device), bgp_docking.to(device)

        _, vdw_radius_docking, pred_docking, mdn_docking, mdn_pred_docking, _ = model(bgl_docking, bgp_docking, train=True)
        pred_docking = pred_docking.squeeze(-1)

        labels_docking = labels_docking.float().to(device)

        loss_docking = (labels_docking - pred_docking)
        loss_docking = loss_docking.clamp(-1.0).mean()

        loss_radius_docking = vdw_radius_docking.pow(2).mean()
        loss_docking = loss_docking * 10 + loss_radius_docking * 10 + 0.5 * mdn_docking

        sample_crossing = next(cross_iter, None)
        # print("crossing successful")
        if sample_crossing is None:
            break

        pdbids_crossing, bgp_crossing, bgl_crossing, labels_crossing = sample_crossing
        bgl_crossing, bgp_crossing = bgl_crossing.to(device), bgp_crossing.to(device)

        _, vdw_radius_crossing, pred_crossing, mdn_crossing, mdn_pred_crossing, _ = model(bgl_crossing, bgp_crossing, train=True)
        pred_crossing = pred_crossing.squeeze(-1)
        labels_crossing = labels_crossing.float().to(device)

        loss_crossing = (labels_crossing - pred_crossing)
        loss_crossing = loss_crossing.clamp(0.0).mean()
        loss_radius_crossing = vdw_radius_crossing.pow(2).mean()
        loss_crossing = loss_crossing * 5 + loss_radius_crossing * 10 + mdn_crossing * 0.5

        # loss_random
        # loss_random = torch.zeros((1,))
        # keys_random = []
        sample_random = next(random_iter, None)
        if sample_random is None:
            break

        pdbids_random, bgp_random, bgl_random, labels_random = sample_random
        bgl_random, bgp_random = bgl_random.to(device), bgp_random.to(device)

        _, vdw_radius_random, pred_random, mdn_random, mdn_pred_random, _ = model(bgl_random, bgp_random, train=True)
        pred_random = pred_random.squeeze(-1)
        labels_random = labels_random.float().to(device)

        loss_random = (labels_random - pred_random)
        loss_random = loss_random.clamp(0.0).mean()
        loss_radius_random = vdw_radius_random.pow(2).mean()
        loss_random = loss_random * 5 + loss_radius_random * 10 + 0.5 * mdn_random

        loss_all = loss_random + loss_crossing + loss_docking + loss_score

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        probs.append(pred.detach())
        true.append(labels)
        # total_loss += loss.item() * batch.unique().size(0)
        total_loss += loss_all.detach()

        # print(f"loss is {loss}, loss_crossing is {loss_crossing}, loss_random is {loss_random}, "
        #       f"loss_docking is {loss_docking}, loss_pda is {loss_pda}, loss_radius is {loss_radius}")

        del loss_score, loss_crossing, loss_docking, loss_random, labels, labels_docking, labels_random, \
            labels_crossing, pred, pred_crossing, pred_docking, pred_random, bgl, \
            bgl_random, bgl_docking, bgl_crossing, bgp, bgp_random, bgp_crossing, bgp_docking, \
            sample, sample_random, sample_docking, sample_crossing, pdbids, pdbids_random, pdbids_crossing,\
            pdbids_docking

        th.cuda.empty_cache()

    ys = th.cat(probs)
    ps = th.cat(true)
    affi_pr = th.corrcoef(th.stack([ys, ps]))[1, 0].item()

    return affi_pr

def GIP_semi_label(epoch, model, data_loader1, data_loader2, optimizer, mdn_weight=1.0, affi_weight=1.0, aux_weight=0.001,
                    dist_threhold=None, device='cpu'):
    iter1, iter2 = iter(data_loader1), iter(data_loader2)
    loss_fn = nn.MSELoss()
    model.train()
    total_loss = 0
    mdn_loss = 0
    affi_loss = 0
    atom_loss = 0
    bond_loss = 0
    probs = []
    true = []

    while True:
        # sample fep or derivate data for virtual labels
        sample_vir = next(iter2, None)

        if sample_vir is None:
            break

        pdbids_vir, bgp_vir, bgl_vir, labels_vir = sample_vir

        bgl_vir, bgp_vir = bgl_vir.to(device), bgp_vir.to(device)

        vdw_pred_vir, vdw_radius_vir, pred_vir = model(bgl_vir, bgp_vir, train=True)
        pred_vir = pred_vir.squeeze(-1)

        labels_vir = labels_vir.float().to(device)
        vdw_vir = labels_vir * 0.8

        loss1_vir = abs(loss_fn(vdw_pred_vir, vdw_vir) - 0.5)
        loss2_vir = abs(loss_fn(pred_vir, labels_vir) - 0.1)

        loss_vir = loss2_vir + 0.01 * loss1_vir

        # sample pdbbind2020 with real label

        sample_real = next(iter1, None)

        pdbids_real, bgp_real, bgl_real, labels_real = sample_real

        bgl_real, bgp_real = bgl_real.to(device), bgp_real.to(device)

        vdw_pred_real, vdw_radius_real, pred_real = model(bgl_real, bgp_real, train=True)
        pred_real = pred_real.squeeze(-1)

        labels_real = labels_real.float().to(device)
        vdw_real = labels_real * 0.8

        loss1_real = abs(loss_fn(vdw_pred_real, vdw_real) - 0.5)
        loss2_real = loss_fn(pred_real, labels_real)

        loss_real = loss2_real + 0.01 * loss1_real

        loss = loss_real + 0.5 * loss_vir

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs.append([pred_real.detach(), pred_vir.detach()])
        true.append([labels_real, labels_vir])
        # total_loss += loss.item() * batch.unique().size(0)
        total_loss += loss.detach()

        del loss, labels_real, labels_vir, pred_real, pred_vir, bgl_real, bgl_vir, bgp_real, bgp_vir, \
            sample_real, sample_vir, pdbids_real, pdbids_vir, vdw_pred_real, vdw_pred_vir
        th.cuda.empty_cache()

    return total_loss


def GIP_train_epoch(epoch, model, data_loader, optimizer, mdn_weight=1.0, affi_weight=1.0, aux_weight=0.001,
                    dist_threhold=None, device='cpu'):
    loss_fn = nn.MSELoss()
    model.train()
    total_loss = 0
    mdn_loss = 0
    affi_loss = 0
    atom_loss = 0
    bond_loss = 0
    probs = []
    true = []
    for batch_id, batch_data in enumerate(data_loader):
        pdbids, bgp, bgl, labels = batch_data
        bgl, bgp = bgl.to(device), bgp.to(device)

        vdw_pred, vdw_radius, pred= model(bgl, bgp, train=True)
        pred = pred.squeeze(-1)
        # mdn_pred = mdn_pred.squeeze(-1)

        labels = labels.float().to(device)
        vdw = labels * 0.8

        loss1 = abs(loss_fn(vdw_pred, vdw) - 0.5)
        loss2 = loss_fn(pred, labels)
        loss_radius = vdw_radius.pow(2).mean()
        # version3
        # =================================================================================================
        # # loss_radius = vdw_radius.pow(2).mean()
        # loss_mdn = th.corrcoef(th.stack([mdn_pred, labels]))[1, 0].item()
        #
        # print(f"loss_pred: {loss2}, mdn: {mdn}, loss_mdn: {loss_mdn}")
        # # loss = loss2 + 0.05 * loss1 + 10 * loss_radius + mdn - loss_mdn
        # # loss = loss2 + 0.1 * loss1
        # loss = loss2 + (mdn - loss_mdn) * 0.5
        # =================================================================================================

        # =================================================================================================
        # # version4
        # affi_mdn = th.corrcoef(th.stack([mdn_pred, labels]))[1, 0].item()
        # affi_score = th.corrcoef(th.stack([pred, labels]))[1, 0].item()
        # loss_mdn = loss_fn(mdn_pred, labels)
        #
        # print(f"loss_pred: {loss2}, mdn: {mdn}, loss_mdn: {loss_mdn}, affi_mdn: {affi_mdn}, affi_score: {affi_score}")
        # # loss = loss2 + 0.05 * loss1 + 10 * loss_radius + mdn - loss_mdn
        # # loss = loss2 + 0.1 * loss1
        # loss = loss2 + mdn + loss_mdn * 0.1
        # =================================================================================================

        # version 5

        # loss_mdn = th.corrcoef(th.stack([mdn_pred, labels]))[1, 0].item()
        print(f"loss1:{loss1*0.05}, loss2:{loss2}, loss_radius:{5 * loss_radius}")
        loss = loss2 + 0.02 * loss1 + 5 * loss_radius

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs.append(pred.detach())
        true.append(labels)
        # total_loss += loss.item() * batch.unique().size(0)
        total_loss += loss.detach()

        del loss, labels, pred, bgl, bgp, batch_data, pdbids, vdw_pred
        th.cuda.empty_cache()

        # for name, param in model.named_parameters():
        #     print(name, param)

    ys = th.cat(probs)
    ps = th.cat(true)
    affi_pr = th.corrcoef(th.stack([ys, ps]))[1, 0].item()

    return total_loss / len(data_loader.dataset), affi_pr


def GIP_eval_epoch(model, data_loader, pred=False, atom_contribution=False, res_contribution=False,
                   dist_threhold=None, mdn_weight=1.0, affi_weight=1.0, aux_weight=0.001, device='cpu'):
    loss_fn = nn.MSELoss()
    model.eval()
    total_loss = 0
    mdn_loss = 0
    affi_loss = 0
    atom_loss = 0
    bond_loss = 0
    probs = []
    true = []
    with th.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            pdbids, bgp, bgl, labels = batch_data
            bgl, bgp = bgl.to(device), bgp.to(device)
            # print(pdbids)
            _, _, pred, _, _, _ = model(bgl, bgp, train=False)
            pred = pred.squeeze(-1)

            labels = labels.float().to(device)

            loss = loss_fn(pred, labels)

            probs.append(pred.detach())
            true.append(labels)
            # total_loss += loss.item() * batch.unique().size(0)
            total_loss += loss.detach()

            del loss, labels, pred, bgl, bgp, batch_data, pdbids
            th.cuda.empty_cache()

    ys = th.cat(probs)
    ps = th.cat(true)
    # print(f"true is {ps}")
    # print(f"pred is {ys}")
    affi_pr = th.corrcoef(th.stack([ys, ps]))[1, 0].item()
    corr, p_value = spearmanr(ys.cpu().detach().numpy(), ps.cpu().detach().numpy())
    rmse = sqrt(mean_squared_error(ys.cpu().detach().numpy(), ps.cpu().detach().numpy()))
    print(f"corr is {corr} \t p is {p_value} \t pr is {affi_pr}")
    return corr, affi_pr, ys.cpu().numpy(), rmse

def run_a_train_epoch(epoch, model, data_loader, optimizer, mdn_weight=1.0, affi_weight=1.0, aux_weight=0.001,
                      dist_threhold=None, device='cpu'):
    loss_fn = nn.MSELoss()
    model.train()
    total_loss = 0
    mdn_loss = 0
    affi_loss = 0
    atom_loss = 0
    bond_loss = 0
    probs = []
    for batch_id, batch_data in tqdm(enumerate(data_loader)):
        pdbids, bgp, bgl, labels = batch_data
        bgl, bgp = bgl.to(device), bgp.to(device)

        atom_labels = th.argmax(bgl.x[:, :17], dim=1, keepdim=False)
        bond_labels = th.argmax(bgl.edge_attr[:, :4], dim=1, keepdim=False)

        pi, sigma, mu, dist, atom_types, bond_types, batch = model(bgl, bgp, train=True)

        mdn, prob = mdn_loss_fn(pi, sigma, mu, dist)
        mdn = mdn[th.where(dist <= model.dist_threhold)[0]]
        mdn = mdn.mean()

        batch = batch.to(device)
        if dist_threhold is not None:
            prob = prob[th.where(dist <= dist_threhold)[0]]
            y = scatter_add(prob, batch[th.where(dist <= dist_threhold)[0]], dim=0, dim_size=batch.unique().size(0)).to(
                th.float32)
        else:
            y = scatter_add(prob, batch, dim=0, dim_size=batch.unique().size(0)).to(th.float32)

        # labels = labels.float().to(device)
        # affi = loss_fn(y, labels)
        labels = labels.float().type_as(y).to(device)
        affi = th.corrcoef(th.stack([y, labels]))[1, 0]
        atom = F.cross_entropy(atom_types, atom_labels)
        bond = F.cross_entropy(bond_types, bond_labels)
        loss = (mdn * mdn_weight) + (affi * affi_weight) + (atom * aux_weight) + (bond * aux_weight)

        probs.append(y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # total_loss += loss.item() * batch.unique().size(0)
        mdn_loss += mdn.item() * batch.unique().size(0)
        # affi_loss += affi.item() * batch.unique().size(0)
        atom_loss += atom.item() * batch.unique().size(0)
        bond_loss += bond.item() * batch.unique().size(0)
        total_loss += mdn_loss + atom_loss * aux_weight + bond_loss * aux_weight + affi * affi_weight

        # print('Step, Total Loss: {:.3f}, MDN: {:.3f}'.format(total_loss, mdn_loss))
        if np.isinf(mdn_loss) or np.isnan(mdn_loss): break

        del loss, labels, y, bgl, bgp, batch_data, pdbids
        th.cuda.empty_cache()

    ys = th.cat(probs)
    affi_loss = th.corrcoef(th.stack([ys, data_loader.dataset.labels.to(device)]))[1, 0].item()

    return total_loss / len(data_loader.dataset), mdn_loss / len(
        data_loader.dataset), affi_loss, atom_loss / len(data_loader.dataset), bond_loss / len(data_loader.dataset)


def run_an_eval_epoch(model, data_loader, pred=False, atom_contribution=False, res_contribution=False,
                      dist_threhold=None, mdn_weight=1.0, affi_weight=1.0, aux_weight=0.001, device='cpu'):
    model.eval()
    total_loss = 0
    affi_loss = 0
    mdn_loss = 0
    atom_loss = 0
    bond_loss = 0
    probs = []
    at_contrs = []
    res_contrs = []
    with th.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            pdbids, bgp, bgl, labels = batch_data
            bgl, bgp = bgl.to(device), bgp.to(device)
            atom_labels = th.argmax(bgl.x[:, :17], dim=1, keepdim=False)
            bond_labels = th.argmax(bgl.edge_attr[:, :4], dim=1, keepdim=False)

            pi, sigma, mu, dist, atom_types, bond_types, batch = model(bgl, bgp, train=False)

            if pred or atom_contribution or res_contribution:
                prob = calculate_probablity(pi, sigma, mu, dist)
                if dist_threhold is not None:
                    prob[th.where(dist > dist_threhold)[0]] = 0.

                batch = batch.to(device)
                if pred:
                    probx = scatter_add(prob, batch, dim=0, dim_size=batch.unique().size(0))
                    probs.append(probx)
                if atom_contribution or res_contribution:
                    contribs = [prob[batch == i].reshape(len(bgl.x[bgl.batch == i]), len(bgp.x[bgp.batch == i])) for i
                                in th.arange(0, len(batch.unique()))]
                    if atom_contribution:
                        at_contrs.extend(
                            [contribs[i].sum(1).cpu().detach().numpy() for i in th.arange(0, len(batch.unique()))])
                    if res_contribution:
                        res_contrs.extend(
                            [contribs[i].sum(0).cpu().detach().numpy() for i in th.arange(0, len(batch.unique()))])

            else:
                mdn, prob = mdn_loss_fn(pi, sigma, mu, dist)
                mdn = mdn[th.where(dist <= model.dist_threhold)[0]]
                mdn = mdn.mean()

                batch = batch.to(device)
                if dist_threhold is not None:
                    prob = prob[th.where(dist <= dist_threhold)[0]]
                    y = scatter_add(prob, batch[th.where(dist <= dist_threhold)[0]], dim=0,
                                    dim_size=batch.unique().size(0))
                else:
                    y = scatter_add(prob, batch[th.where(dist <= dist_threhold)[0]], dim=0,
                                    dim_size=batch.unique().size(0))
                labels = labels.float().type_as(y).to(device)

                affi = th.corrcoef(th.stack([y, labels]))[1, 0]

                atom = F.cross_entropy(atom_types, atom_labels)
                bond = F.cross_entropy(bond_types, bond_labels)
                # loss = (mdn * mdn_weight) + (atom * aux_weight) + (bond * aux_weight)
                loss = mdn + affi * affi_weight + (atom * aux_weight) + (bond * aux_weight)

                probs.append(y)

                total_loss += loss.item() * batch.unique().size(0)
                mdn_loss += mdn.item() * batch.unique().size(0)
                atom_loss += atom.item() * batch.unique().size(0)
                bond_loss += bond.item() * batch.unique().size(0)

    if atom_contribution or res_contribution:
        if pred:
            preds = th.cat(probs)
            affi_loss = th.corrcoef(th.stack([ys, data_loader.dataset.labels.to(device)]))[1, 0].item()
            print(f"pr is {affi_loss}")
            return [preds.cpu().detach().numpy(), at_contrs, res_contrs]
        else:
            return [None, at_contrs, res_contrs]
    else:
        if pred:
            preds = th.cat(probs)
            # corr, p_value = spearmanr(preds.cpu().detach().numpy(), data_loader.dataset.labels.cpu().detach().numpy())
            # affi_loss = th.corrcoef(th.stack([preds, data_loader.dataset.labels.to(device)]))[1, 0].item()
            # print(f"pr is {affi_loss}")
            # print(f"spearman is {corr}")
            return preds.cpu().detach().numpy(), data_loader.dataset.labels, data_loader.dataset.pdbids
            # return preds.cpu().detach().numpy()
        else:
            preds = th.cat(probs)
            corr, p_value = spearmanr(preds.cpu().detach().numpy(), data_loader.dataset.labels.cpu().detach().numpy())
            affi_loss = th.corrcoef(th.stack([preds, data_loader.dataset.labels.to(device)]))[1, 0].item()
            # print(f"ys is {preds}")
            print(f"pr is {affi_loss}")
            print(f"spearman is {corr}")
            # total_loss += - affi_loss
            # del preds
            # return total_loss / len(data_loader.dataset) + affi_loss * affi_weight, mdn_loss / len(
            #     data_loader.dataset), affi_loss, atom_loss / len(data_loader.dataset), bond_loss / len(
            #     data_loader.dataset)
            return corr, affi_loss, preds.cpu().numpy()


def set_random_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    # th.backends.cudnn.benchmark = False
    # th.backends.cudnn.deterministic = True
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)


def calculate_probablity(pi, sigma, mu, y):
    normal = Normal(mu, sigma)
    logprob = normal.log_prob(y.expand_as(normal.loc))
    logprob += th.log(pi)
    prob = logprob.exp().sum(1)

    return prob


def mdn_loss_fn(pi, sigma, mu, y, eps1=1e-10, eps2=1e-10):
    normal = Normal(mu, sigma)
    # loss = th.exp(normal.log_prob(y.expand_as(normal.loc)))
    # loss = th.sum(loss * pi, dim=1)
    # loss = -th.log(loss)
    loglik = normal.log_prob(y.expand_as(normal.loc))
    # loss = -th.logsumexp(th.log(pi + eps) + loglik, dim=1)
    prob = (th.log(pi + eps1) + loglik).exp().sum(1)
    loss = -th.log(prob + eps2)
    return loss, prob
