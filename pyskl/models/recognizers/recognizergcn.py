import numpy as np
import torch

from ..builder import RECOGNIZERS
from .base import BaseRecognizer

from  tools.affectSimMat import *
import torch.nn.functional as F
def kl_divergence(p, q, eps=1e-10):
    """Compute KL divergence between two probability distributions."""
    p = torch.clamp(p, min=eps)  # Prevent log(0) by clamping very small values
    q = torch.clamp(q, min=eps)

    return torch.sum(p * torch.log(p / q), dim=-1)

@RECOGNIZERS.register_module()
class RecognizerGCN(BaseRecognizer):
    """GCN-based recognizer for skeleton-based action recognition. """

    def forward_train(self, keypoint, label, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        assert keypoint.shape[1] == 1
        keypoint = keypoint[:, 0]

        # Move tensors to the same device
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        keypoint = keypoint.to(device)
        label = label.to(device)
        ##################################
        losses = dict()
        x = self.extract_feat(keypoint)
        ###########################################    #MemConstGCNHead.py return to here: cls_score=self.cls_head(x)
        cls_score = self.cls_head(x)
        gt_label = label.squeeze(-1)
        loss = self.cls_head.loss(cls_score, gt_label)
        ##################################
        # 1. Select corresponding ideal probability distribution based on predicted category
        ideal_distribution_matrix = affect_sim_mat()
        cls_prob = F.softmax(cls_score, dim=-1)
        kl_loss_list = list()
        length=len(gt_label)
        for i in range(0, length):
            print("i")
            print(i)
            label_number = gt_label[i]
            q = ideal_distribution_matrix[label_number]
            # 2. Compute KL divergence between predicted probability and selected ideal probability
            p = cls_prob[i]
            if isinstance(q, np.ndarray):
                q = torch.tensor(q, dtype=p.dtype, device=p.device)  # Convert q to Tensor
            kl_loss = kl_divergence(p, q)
            kl_loss_list.append(kl_loss)
        ###################################
        average_kl_loss = torch.mean(torch.stack(kl_loss_list))
        normal_loss_value = loss['loss_cls']  # or adjust the key if different
        combined_loss_value = normal_loss_value + 0.6 * average_kl_loss
        loss['loss_cls']=combined_loss_value
        ######################################
        losses.update(loss)
        return losses

    def forward_test(self, keypoint, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        assert self.with_cls_head or self.feat_ext
        bs, nc = keypoint.shape[:2]
        keypoint = keypoint.reshape((bs * nc, ) + keypoint.shape[2:])

        x = self.extract_feat(keypoint)
        feat_ext = self.test_cfg.get('feat_ext', False)
        pool_opt = self.test_cfg.get('pool_opt', 'all')
        score_ext = self.test_cfg.get('score_ext', False)
        if feat_ext or score_ext:
            assert bs == 1
            assert isinstance(pool_opt, str)
            dim_idx = dict(n=0, m=1, t=3, v=4)

            if pool_opt == 'all':
                pool_opt == 'nmtv'
            if pool_opt != 'none':
                for digit in pool_opt:
                    assert digit in dim_idx

            if isinstance(x, tuple) or isinstance(x, list):
                x = torch.cat(x, dim=2)
            assert len(x.shape) == 5, 'The shape is N, M, C, T, V'
            if pool_opt != 'none':
                for d in pool_opt:
                    x = x.mean(dim_idx[d], keepdim=True)

            if score_ext:
                w = self.cls_head.fc_cls.weight
                b = self.cls_head.fc_cls.bias
                x = torch.einsum('nmctv,oc->nmotv', x, w)
                if b is not None:
                    x = x + b[..., None, None]
                x = x[None]
            return x.data.cpu().numpy().astype(np.float16)

        cls_score = self.cls_head(x)
        cls_score = cls_score.reshape(bs, nc, cls_score.shape[-1])
        # harmless patch
        if 'average_clips' not in self.test_cfg:
            self.test_cfg['average_clips'] = 'prob'

        cls_score = self.average_clip(cls_score)
        if isinstance(cls_score, tuple) or isinstance(cls_score, list):
            cls_score = [x.data.cpu().numpy() for x in cls_score]
            return [[x[i] for x in cls_score] for i in range(bs)]

        return cls_score.data.cpu().numpy()

    def forward(self, keypoint, label=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(keypoint, label, **kwargs)

        return self.forward_test(keypoint, **kwargs)

    def extract_feat(self, keypoint):
        """Extract features through a backbone.

        Args:
            keypoint (torch.Tensor): The input keypoints.

        Returns:
            torch.tensor: The extracted features.
        """
        return self.backbone(keypoint)
