# Copyright 2019 The PytorX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn as nn


class Variation(nn.Module):

    def __init__(self, G_shape, noise_amp=0.12, random_noise=True, noise_mean=None, noise_var=None):
        super(Variation, self).__init__()
        '''
        This module performs the Var (variation) non-ideal effect injection.
            Args:
                G_shape (tensor.size): crossbar array size.
                noise_mean (float): noise mean for gaussian distribution 
                noise_var (float): noise variance for gaussian distribution
                noise_amp (float): noise amplitude for determined noise
                random_noise (float): whether use noise distribution
        '''

        # initialize a random mask
        # TODO: maybe change the variation profile to uint8 format to avoid calculating the variation defect
        # state on-the-fly, for simulation speedup. However the current setup has higher configurability
        # to simulate the real-time variation state if there is run-time change .
        self.register_buffer('p_state', torch.empty(G_shape))
        self.update_variation_profile(noise_amp, random_noise, noise_mean, noise_var)  # init the variation distribution profile

    def forward(self, input):
        '''
        The forward function alter the elements that indexed by p_state to the defected conductance,
        and mask the gradient of those defect cells owing to the auto-differentiation. 
        '''
        output = Inject_variation(input, self.p_state)
        return output

    def update_variation_profile(self, noise_amp, random_noise, noise_mean, noise_var, dist='normal'):
        if random_noise:
            if dist == 'normal':
                if noise_mean is not None and noise_var is not None:
                    self.p_state.data.normal_(noise_mean, noise_var**2)
                    # 20210313: add: clip the noise to avoid extreme condition
                    # self.p_state.clamp_(noise_mean-noise_var*3,noise_mean+noise_var*3)
                else:
                    self.p_state.data.normal_(0, abs(noise_amp / 3))
                    # 20210313: add: clip the noise to avoid extreme condition
                    # self.p_state.clamp_(-noise_amp,noise_amp)

        else:
            self.p_state.data[0].fill_(noise_amp)
            self.p_state.data[0].fill_(-noise_amp)
        return


class _variation(torch.autograd.Function):
    r'''
    This autograd function performs the gradient mask for the weight
    element with Stuck-at-Fault defects, where those weights will not
    be updated during backprop through gradient masking.

    Args:
        input (Tensor): weight tensor in FP32
        p_state (Tensor): probability tensor for indicating the variation state
    '''

    @staticmethod
    def forward(ctx, input, p_state):
        # p_state is the mask
        # ctx.save_for_backward(p_state)
        output = input.clone()
        output = output * torch.exp(p_state)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # p_state, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # mask the gradient of defect cells
        return grad_input, None


Inject_variation = _variation.apply


############################################################
# Testbenchs
############################################################

# pytest
def test_variation_update_profile():
    G_shape = torch.Size([16, 3, 3, 3])
    variation_module = Variation(G_shape)
    pre_index_SA0 = variation_module.index_SA0()
    variation_module.update_variation_profile()
    post_index_SA0 = variation_module.index_SA0()
    # # print((pre_index_SA0-post_index_SA0).sum())
    # assert (pre_index_SA0 -
    #         post_index_SA0).sum().item() != 0, 'variation profile is not updated!'
    # # print(variation_module.index_SA0())
    return


def test_SA0_SA1_overlap():
    '''
    ensure there is no variation state overlap between SA0 and SA1
    '''
    G_shape = torch.Size([3, 1, 3, 3])
    variation_module = Variation(G_shape)
    # index_SA0 = variation_module.index_SA0()
    # index_SA1 = variation_module.index_SA1()
    # assert (index_SA0 * index_SA1).sum().item() == 0, 'exist element is 1 for both SA0/1 index!'
    return


# if __name__ == '__main__':
#     test_variation_update_profile()
#     test_SA0_SA1_overlap()
