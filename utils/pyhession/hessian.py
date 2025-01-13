#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import torch
import math
from torch.autograd import Variable
import numpy as np

from utils.pyhession.hutils import \
    group_product,get_params_grad, hessian_vector_product,smooth_max,smooth_min


class Hessian():
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
    """

    def __init__(self, model,cuda=True):
        """
        model: the model that needs Hessain information
        criterion: the loss function
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """

        self.model = model.eval()  # make model is in evaluation model

        if cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # this step is used to extract the parameters from the model
        params, gradsH = get_params_grad(self.model)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation


    def trace(self, maxIter=100, tol=1e-3):
        """
        compute the trace of hessian using Hutchinson's method
        maxIter: maximum iterations used to compute trace
        tol: the relative tolerance
        """

        device = self.device
        trace_vhv = []
        trace = 0.0

        for i in range(maxIter):
            self.model.zero_grad()
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            Hv = hessian_vector_product(self.gradsH, self.params, v)

            trace_vhv.append(group_product(Hv, v).cpu().item())

            mean_trace = torch.mean(torch.tensor(trace_vhv, device=device))
            if torch.abs(mean_trace - trace) / (trace + 1e-6) < tol:
                break;
            else:
                trace = mean_trace
                # Perform smooth min-max normalization
        trace_vhv_tensor = torch.tensor(trace_vhv, device=device)
        max_trace = smooth_max(trace_vhv_tensor)
        min_trace = smooth_min(trace_vhv_tensor)
        normalized_trace_vhv = (trace_vhv_tensor - min_trace) / (max_trace - min_trace + 1e-6)

        return normalized_trace_vhv.mean().item()
