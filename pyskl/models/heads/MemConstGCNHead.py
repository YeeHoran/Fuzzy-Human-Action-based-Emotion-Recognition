from pyskl.models import HEADS
from pyskl.models.heads import GCNHead
import torch
import torch.nn as nn
import numpy as np
from  tools.affectSimMat import *
import torch.nn.functional as F
from collections.abc import Iterable

def kl_divergence(p, q, eps=1e-10):
    """Compute KL divergence between two probability distributions."""
    p = torch.clamp(p, min=eps)  # Prevent log(0) by clamping very small values
    q = torch.clamp(q, min=eps)

    return torch.sum(p * torch.log(p / q), dim=-1)


@HEADS.register_module()
class MemConstGCNHead(GCNHead):
    def __init__(self, num_classes, in_channels, loss_cls=None, alpha=1.0, custom_weight=0.5, videos_per_gpu=16, **kwargs):
        """
        Initialize the custom head. In addition to CrossEntropyLoss,
        add a custom loss term controlled by the 'alpha' parameter.

        - alpha: Weighting factor for the constraint term
        - custom_weight: Can be used as another custom parameter if needed
        """
        # Call the parent class constructor (passing only the expected arguments)
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        # Store the alpha and custom_weight for the constraint term
        self.alpha = alpha
        self.custom_weight = custom_weight if custom_weight is not None else 1.0
        self.batchsize=videos_per_gpu


    def forward(self, x, constraint_output=None):
        """
        Forward pass that computes the final loss by combining the
        original CrossEntropyLoss and the constraint loss.

        - x: Input data
        - target: Ground truth labels
        - constraint_output: Output that will be used to calculate the constraint (can be another model's layer/output)
        """

        # 1. Get raw model output (logits)
        output = super().forward(x)
        print(output)

        '''
        # 2. Convert model output logits to probability distribution
        output_prob = F.softmax(output, dim=-1)
        print(output_prob)
        # 3. Determine predicted category (argmax selection)
        pred_category = torch.argmax(output_prob, dim=-1)
        # 4. Select corresponding ideal probability distribution based on predicted category
        ideal_distribution_matrix= affect_sim_mat()
        print(ideal_distribution_matrix)
        #ideal_list=list()
        kl_loss_list=list()
        for i in range(0,self.batchsize):

            label=pred_category[i]
            q = ideal_distribution_matrix[label]

            # 5. Compute KL divergence between predicted probability and selected ideal probability
            p=output_prob[i]
            if isinstance(q, np.ndarray):
                q = torch.tensor(q, dtype=p.dtype, device=p.device)  # Convert q to Tensor
            kl_loss = kl_divergence(p, q)
            kl_loss_list.append(kl_loss)
        # 6. Combine distributions with weighted sum
        # Ensure that ideal_list is a valid PyTorch tensor.
        # Assuming ideal_list is a list of lists and is already in float format:
        # Flatten nested lists into a single list if needed
        '''
        '''
        flattened_ideal_list = list(flatten(ideal_list))
        ideal_tensor = torch.tensor(flattened_ideal_list, dtype=torch.float32, device=output_prob.device)
        '''
        #combined_loss = self.alpha * output_prob + self.custom_weight * ideal_tensor
        return output


        '''
        # Example constraint: L2 regularization term between outputs
        constraint_loss = torch.sum((output - constraint_output) ** 2)
        return constraint_loss
        '''











