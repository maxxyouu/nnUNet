#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
import torch
from nnunet.network_architecture.custom_modules.conv_blocks import BasicResidualBlock, ResidualLayer
from nnunet.network_architecture.generic_UNet import Upsample
from nnunet.network_architecture.generic_modular_UNet import get_default_network_config
from nnunet.network_architecture.neural_network import SegmentationNetwork
from timm.models.layers import trunc_normal_tf_ as trunc_normal_
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from torch import nn
from torch.optim import SGD
from torch.backends import cudnn
import math
from torch.nn import functional as F

from nnunet.training.loss_functions.dualTasks_dice_ce_loss import MultiTask_DCCE_CE_loss



class ResidualUNetEncoder(nn.Module):
    def __init__(self, input_channels, base_num_features, num_blocks_per_stage, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, default_return_skips=True,
                 max_num_features=480, block=BasicResidualBlock, block_kwargs=None):
        """
        Following UNet building blocks can be added by utilizing the properties this class exposes

        this one includes the bottleneck layer!

        :param input_channels:
        :param base_num_features: number of features to start with
        :param num_blocks_per_stage: each stage contains multiple layers of the same set of operation
        :param feat_map_mul_on_downscale: expansion rate
        :param pool_op_kernel_sizes: eg like 3x3 kernel 
        :param conv_kernel_sizes: eg like 3x3 kernel
        :param props:
        """
        super(ResidualUNetEncoder, self).__init__()

        if block_kwargs is None:
            block_kwargs = {}

        self.default_return_skips = default_return_skips
        self.props = props

        self.stages = []
        self.stage_output_features = []
        self.stage_pool_kernel_size = []
        self.stage_conv_op_kernel_size = []

        assert len(pool_op_kernel_sizes) == len(conv_kernel_sizes)

        num_stages = len(conv_kernel_sizes)

        if not isinstance(num_blocks_per_stage, (list, tuple)):
            num_blocks_per_stage = [num_blocks_per_stage] * num_stages
        else:
            assert len(num_blocks_per_stage) == num_stages

        self.num_blocks_per_stage = num_blocks_per_stage  # decoder may need this

        # NOTE: might be the stem?
        self.initial_conv = props['conv_op'](input_channels, base_num_features, 3, padding=1, **props['conv_op_kwargs'])
        self.initial_norm = props['norm_op'](base_num_features, **props['norm_op_kwargs'])
        self.initial_nonlin = props['nonlin'](**props['nonlin_kwargs'])

        # code for repeat the conv ops in each stage
        current_input_features = base_num_features
        for stage in range(num_stages):

            # find the right number of output channels caped by the max_num_features
            current_output_features = min(base_num_features * feat_map_mul_on_downscale ** stage, max_num_features)
            
            # conv_kernel_sizes and pool_op_kernel_sizes are lists that lay out the kernel size for each stage, each stage shares the same kernel size
            current_kernel_size = conv_kernel_sizes[stage]
            current_pool_kernel_size = pool_op_kernel_sizes[stage]

            # number of convolutions is repeated inside the ResidualLayer class
            current_stage = ResidualLayer(current_input_features, current_output_features, current_kernel_size, props,
                                            self.num_blocks_per_stage[stage], current_pool_kernel_size, block,
                                            block_kwargs)

            self.stages.append(current_stage)
            self.stage_output_features.append(current_stage.output_channels)
            self.stage_conv_op_kernel_size.append(current_kernel_size)
            self.stage_pool_kernel_size.append(current_pool_kernel_size)

            # update current_input_features
            current_input_features = current_stage.output_channels

        # get the final output features of the encoder
        self.output_features = current_input_features
        self.stages = nn.ModuleList(self.stages)

    def forward(self, x, return_skips=None):
        """

        :param x:
        :param return_skips: if none then self.default_return_skips is used
        :return:
        """
        skips = []
        # stem forward pass
        x = self.initial_nonlin(self.initial_norm(self.initial_conv(x)))

        # run sequentially stage by stage and store the intermediate outputs
        for s in self.stages:
            x = s(x)
            if self.default_return_skips:
                skips.append(x)

        if return_skips is None:
            return_skips = self.default_return_skips

        # print(x.shape)

        if return_skips:
            return skips # return the output for each stage of the encoder
        else:
            return x # return the final output of the encoder only

    @staticmethod
    def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                        num_modalities, pool_op_kernel_sizes, num_conv_per_stage_encoder,
                                        feat_map_mul_on_downscale, batch_size):
        npool = len(pool_op_kernel_sizes) - 1

        current_shape = np.array(patch_size)

        tmp = (num_conv_per_stage_encoder[0] * 2 + 1) * np.prod(current_shape) * base_num_features \
              + num_modalities * np.prod(current_shape)

        num_feat = base_num_features

        for p in range(1, npool + 1):
            current_shape = current_shape / np.array(pool_op_kernel_sizes[p])
            num_feat = min(num_feat * feat_map_mul_on_downscale, max_num_features)
            num_convs = num_conv_per_stage_encoder[p] * 2 + 1  # + 1 for conv in skip in first block
            print(p, num_feat, num_convs, current_shape)
            tmp += num_convs * np.prod(current_shape) * num_feat
        return tmp * batch_size


def get_norm(name, channels):
    if name is None or name.lower() == 'none':
        return nn.Identity()

    if name.lower() == 'syncbn':
        return nn.SyncBatchNorm(channels, eps=1e-3, momentum=0.01)

def get_activation(name):
    if name is None or name.lower() == 'none':
        return nn.Identity()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'gelu':
        return nn.GELU()
    
class ConvBN_3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm=None, act=None,
                conv_type='3d', conv_init='he_normal', norm_init=1.0):
        super().__init__()
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.norm = get_norm(norm, out_channels)
        self.act = get_activation(act)

        if conv_init == 'normal':
            nn.init.normal_(self.conv.weight, std=.02)
        elif conv_init == 'trunc_normal':
            trunc_normal_(self.conv.weight, std=.02)
        elif conv_init == 'he_normal':
            # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal
            trunc_normal_(self.conv.weight, std=math.sqrt(2.0 / in_channels))
        elif conv_init == 'xavier_uniform':
            nn.init.xavier_uniform_(self.conv.weight)
        if bias:
            nn.init.zeros_(self.conv.bias)

        if norm is not None:
            nn.init.constant_(self.norm.weight, norm_init)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
class ASPP(nn.Module):
    def __init__(self, in_channels, output_channels, scale_factor, mode, atrous_rates=[6 , 12, 18]):
        super().__init__()

        self._aspp_conv0 = ConvBN_3D(in_channels, output_channels, kernel_size=1, bias=False, norm='syncbn', act='gelu')

        rate1, rate2, rate3 = atrous_rates
        self._aspp_conv1 = ConvBN_3D(in_channels, output_channels, kernel_size=3, dilation=rate1, padding=rate1, bias=False,
                                    norm='syncbn', act='gelu')

        self._aspp_conv2 = ConvBN_3D(in_channels, output_channels, kernel_size=3, dilation=rate2, padding=rate2, bias=False,
                                    norm='syncbn', act='gelu')

        self._aspp_conv3 = ConvBN_3D(in_channels, output_channels, kernel_size=3, dilation=rate3, padding=rate3, bias=False,
                                    norm='syncbn', act='gelu')

        self._avg_pool = nn.AdaptiveAvgPool3d(1) # for global average pooling in 3d
        self._aspp_pool = ConvBN_3D(in_channels, output_channels, kernel_size=1, bias=False,
                                    norm='syncbn', act='gelu')

        self._proj_conv_bn_act = ConvBN_3D(output_channels * 5, output_channels, kernel_size=1, bias=False,
                                    norm='syncbn', act='gelu')
        # https://github.com/google-research/deeplab2/blob/main/model/decoder/aspp.py#L249
        self._proj_drop = nn.Dropout(p=0.1)

        self.upsample = Upsample(scale_factor=list(scale_factor), mode=mode)

    def forward(self, x):
        #NOTE: the input and output shape should be the same based on the upsample size
        #of the global average pooling, the concatenation after, and the kernel size is 1 for the final projection.
        results = []
        results.append(self._aspp_conv0(x))
        results.append(self._aspp_conv1(x))
        results.append(self._aspp_conv2(x))
        results.append(self._aspp_conv3(x))
        align_corners = (x.shape[-1] % 2 == 1)

        # global average pooling
        global_pooled_result = self._aspp_pool(self._avg_pool(x))
        results.append(F.interpolate(global_pooled_result, size=x.shape[-3:], mode='trilinear', align_corners=align_corners))

        x = torch.cat(results, dim=1)
        #TODO: squeeze and excite it if necessary before the final convolution
        x = self._proj_conv_bn_act(x)
        # x = self._proj_drop(x)
        x = self.upsample(x) # upsample to the corresponding size NOTE: try downsample if this does not work well.
        return x

class ResidualMultiTaskUNetDecoder(nn.Module):
    """
    main class for multitask (segmentation + multiclass classification) unet
    """
    def __init__(self, previous, num_classes, num_blocks_per_stage=None, network_props=None, deep_supervision=False,
                    upscale_logits=False, block=BasicResidualBlock, block_kwargs=None):
        super(ResidualMultiTaskUNetDecoder, self).__init__()
        if block_kwargs is None:
            block_kwargs = {}
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        """
        We assume the bottleneck is part of the encoder, so we can start with upsample -> concat here
        """

        # previous is the residual unet encoder from the main class
        # obtain the configuration of the residual unet encoder
        # NOTE: the motivation behind this is that the input size of the current decoder stage need to be the same as the output size of encoder stage of the current level.
        previous_stages = previous.stages # the encoder residual layers of each stage
        previous_stage_output_features = previous.stage_output_features # the intemdiate output features for skip connections
        previous_stage_pool_kernel_size = previous.stage_pool_kernel_size # pool size configuration
        previous_stage_conv_op_kernel_size = previous.stage_conv_op_kernel_size # convolution size configuration

        if network_props is None:
            self.props = previous.props
        else:
            self.props = network_props

        if self.props['conv_op'] == nn.Conv2d:
            transpconv = nn.ConvTranspose2d
            upsample_mode = "bilinear"
        elif self.props['conv_op'] == nn.Conv3d:
            transpconv = nn.ConvTranspose3d
            upsample_mode = "trilinear"
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(self.props['conv_op']))

        if num_blocks_per_stage is None:
            num_blocks_per_stage = previous.num_blocks_per_stage[:-1][::-1]

        assert len(num_blocks_per_stage) == len(previous.num_blocks_per_stage) - 1

        # store the encoder convolution configuration to the decoder configuration
        self.stage_pool_kernel_size = previous_stage_pool_kernel_size
        self.stage_output_features = previous_stage_output_features
        self.stage_conv_op_kernel_size = previous_stage_conv_op_kernel_size

        num_stages = len(previous_stages) - 1  # we have one less as the first stage here is what comes after the， eg previous_stages = 5, num_stages = 4
        # bottleneck

        self.tus = []
        self.stages = []
        self.classification_stages = []
        self.deep_supervision_seg_outputs = []
        self.deep_supervision_class_outputs = []

        # only used for upsample_logits
        cum_upsample = np.cumprod(np.vstack(self.stage_pool_kernel_size), axis=0).astype(int)
        decoder_output_features = []
        for i, s in enumerate(np.arange(num_stages)[::-1]): # eg [3,2,1,0]
            features_below = previous_stage_output_features[s + 1] #eg: at first pick stage 4, the number of features output from the encoder
            features_skip = previous_stage_output_features[s] #eg: the first stage of decoder has input from below and the features from the second last stage of the encoder
            decoder_output_features.append(features_skip)
            # tus mainly to upsample the features, features_below inially from the output of the encoder
            self.tus.append(transpconv(features_below, features_skip, previous_stage_pool_kernel_size[s + 1],
                                        previous_stage_pool_kernel_size[s + 1], bias=False))
            
            # massage the upsampled features
            # after we tu we concat features so now we have 2xfeatures_skip
            self.stages.append(ResidualLayer(2 * features_skip, features_skip, previous_stage_conv_op_kernel_size[s],
                                            self.props, num_blocks_per_stage[i], None, block, block_kwargs))
            
            # massage the decoded features at different scale at each stage
            self.classification_stages.append(ASPP(features_skip, features_skip, scale_factor=cum_upsample[s], mode=upsample_mode))

            if deep_supervision and s != 0:
                seg_layer = self.props['conv_op'](features_skip, num_classes, 1, 1, 0, 1, 1, bias=True)
                class_layer = nn.Sequential(
                    #TODO: check if this need to massage the feature before pooling
                    self.props['conv_op'](features_skip, features_skip, 3, 1, 0, 1, 1, bias=True), # the input are the vanilla upsampled , masage it
                    nn.AdaptiveAvgPool3d(1),
                    self.props['conv_op'](features_skip, num_classes, 1, 1, 0, 1, 1, bias=True))

                self.deep_supervision_class_outputs.append(class_layer)
                if upscale_logits:
                    upsample = Upsample(scale_factor=cum_upsample[s], mode=upsample_mode)
                    self.deep_supervision_seg_outputs.append(nn.Sequential(seg_layer, upsample))
                else:
                    self.deep_supervision_seg_outputs.append(seg_layer)

        # task head
        self.segmentation_head = self.props['conv_op'](features_skip, num_classes, 1, 1, 0, 1, 1, bias=True)
        self.classification_head = nn.Sequential(
            #TODO: use np.sum(decoder_output_features) after debugging
            # since the input are the concatenated result, need to massage it a bit before classification, features_skip = 32
            self.props['conv_op'](np.sum(decoder_output_features[0]), features_skip, 3, 1, 0, 1, 1, bias=True),
            nn.AdaptiveAvgPool3d(1),
            self.props['conv_op'](features_skip, num_classes, 1, 1, 0, 1, 1, bias=True))

        # TODO: comment this after debugging
        self.nonreliable_classification_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            self.props['conv_op'](np.sum(decoder_output_features), num_classes, 1, 1, 0, 1, 1, bias=True))

        self.tus = nn.ModuleList(self.tus)
        self.stages = nn.ModuleList(self.stages)
        self.deep_supervision_seg_outputs = nn.ModuleList(self.deep_supervision_seg_outputs)


    def forward(self, skips):
        # skips come from the encoder. They are sorted so that the bottleneck is last in the list
        # what is maybe not perfect is that the TUs and stages here are sorted the other way around
        # so let's just reverse the order of skips
        skips = skips[::-1]
        seg_outputs = []
        class_outputs = [] # NOTE: assume all the outputs in here are of the same size
        class_outputs_deep_supervision = []

        x = skips[0]  # this is the bottleneck, the bridge between the encoder and decoder

        for i in range(len(self.tus)):
            x = self.tus[i](x)
            x = torch.cat((x, skips[i + 1]), dim=1) # concatenate the encoder features of the same level and the upsampled decoded features
            x = self.stages[i](x)

            xc = self.classification_stages[i](x) #aspp operations
            class_outputs.append(xc) # save the intermediate features for final classification
            
            # auxiliary predictions for deep supervisions
            if self.deep_supervision and (i != len(self.tus) - 1):
                seg_aux_output = self.deep_supervision_seg_outputs[i](x)
                seg_outputs.append(seg_aux_output)

                class_aux_output = self.deep_supervision_class_outputs[i](xc)
                class_outputs_deep_supervision.append(class_aux_output.squeeze()) # to shape [batch, classes]


        # multi-task outputs
        segmentation = self.segmentation_head(x)
        x = torch.cat(class_outputs, dim=1)

        # TODO: uncomment this after debugging
        # classification = self.classification_head(x)

        # TODO: comment this after debugging
        classification = self.classification_head(class_outputs[0])

        if self.deep_supervision:
            seg_outputs.append(segmentation)
            class_outputs_deep_supervision.append(classification.squeeze()) # to shape [batch, classes]
            return seg_outputs[::-1], class_outputs_deep_supervision[::-1]
            # seg_outputs are ordered so that the seg from the highest layer is first, the seg from
            # the bottleneck of the UNet last
        else:
            return segmentation, classification

    @staticmethod
    def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                        num_classes, pool_op_kernel_sizes, num_blocks_per_stage_decoder,
                                        feat_map_mul_on_downscale, batch_size):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :return:
        """
        npool = len(pool_op_kernel_sizes) - 1

        current_shape = np.array(patch_size)
        tmp = (num_blocks_per_stage_decoder[-1] * 2 + 1) * np.prod(
            current_shape) * base_num_features + num_classes * np.prod(current_shape)

        num_feat = base_num_features

        for p in range(1, npool):
            current_shape = current_shape / np.array(pool_op_kernel_sizes[p])
            num_feat = min(num_feat * feat_map_mul_on_downscale, max_num_features)
            num_convs = num_blocks_per_stage_decoder[-(p + 1)] * 2 + 1 + 1  # +1 for transpconv and +1 for conv in skip
            print(p, num_feat, num_convs, current_shape)
            tmp += num_convs * np.prod(current_shape) * num_feat

        return tmp * batch_size


class ResidualMultiTaskUNet(SegmentationNetwork):
    """
    main class that composite the encoder and decoder
    """
    use_this_for_batch_size_computation_2D = 858931200.0  # 1167982592.0
    use_this_for_batch_size_computation_3D = 727842816.0  # 1152286720.0
    default_base_num_features = 24
    default_conv_per_stage = (2, 2, 2, 2, 2, 2, 2, 2)
    default_blocks_per_stage_encoder = (1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4) # the extra 1 comes from the stem
    default_blocks_per_stage_decoder = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

    def __init__(self, input_channels, base_num_features, num_blocks_per_stage_encoder, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, num_classes, num_blocks_per_stage_decoder,
                 deep_supervision=False, upscale_logits=False, max_features=512, initializer=None,
                 block=BasicResidualBlock, block_kwargs=None):
        super(ResidualMultiTaskUNet, self).__init__()

        if block_kwargs is None:
            block_kwargs = {}

        self.conv_op = props['conv_op']
        self.num_classes = num_classes

        self.encoder = ResidualUNetEncoder(input_channels, base_num_features, num_blocks_per_stage_encoder,
                                            feat_map_mul_on_downscale, pool_op_kernel_sizes, conv_kernel_sizes,
                                            props, default_return_skips=True, max_num_features=max_features,
                                            block=block, block_kwargs=block_kwargs)
        self.decoder = ResidualMultiTaskUNetDecoder(self.encoder, num_classes, num_blocks_per_stage_decoder, props,
                                            deep_supervision, upscale_logits, block=block, block_kwargs=block_kwargs)
        if initializer is not None:
            self.apply(initializer)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    @staticmethod
    def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, num_conv_per_stage_encoder,
                                        num_conv_per_stage_decoder, feat_map_mul_on_downscale, batch_size):
        enc = ResidualUNetEncoder.compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                                                    num_modalities, pool_op_kernel_sizes,
                                                                    num_conv_per_stage_encoder,
                                                                    feat_map_mul_on_downscale, batch_size)
        dec = ResidualMultiTaskUNetDecoder.compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                                                    num_classes, pool_op_kernel_sizes,
                                                                    num_conv_per_stage_decoder,
                                                                    feat_map_mul_on_downscale, batch_size)

        return enc + dec


if __name__ == "__main__":
    # test the architecuture
    cudnn.deterministic = False
    cudnn.benchmark = True

    patch_size = (20, 320, 256)
    max_num_features = 320
    num_modalities = 1
    num_classes = 3 # TODO: check how this impact the different number of class for segmentation and classification
    batch_size = 2

    # torch.Size([2, 1, 20, 320, 256]), batch size, modality, channel, h, w

    # now we fiddle with the network specific hyperparameters until everything just barely fits into a titanx
    blocks_per_stage_encoder = ResidualMultiTaskUNet.default_blocks_per_stage_encoder
    blocks_per_stage_decoder = ResidualMultiTaskUNet.default_blocks_per_stage_decoder
    initial_num_features = 32

    # we neeed to add a [1, 1, 1] for the res unet because in this implementation all stages of the encoder can have a stride
    # pool_op_kernel_sizes = [[1, 1, 1],
    #                         [1, 2, 2],
    #                         [1, 2, 2],
    #                         [2, 2, 2],
    #                         [2, 2, 2],
    #                         [1, 2, 2],
    #                         [1, 2, 2]]

    # conv_op_kernel_sizes = [[1, 3, 3],
    #                         [1, 3, 3],
    #                         [3, 3, 3],
    #                         [3, 3, 3],
    #                         [3, 3, 3],
    #                         [3, 3, 3],
    #                         [3, 3, 3]]
    
    pool_op_kernel_sizes = [[1, 1, 1],
                            [1, 2, 2],
                            [2, 2, 2],
                            [1, 2, 2]]

    conv_op_kernel_sizes = [[1, 3, 3],
                            [3, 3, 3],
                            [3, 3, 3],
                            [3, 3, 3]]

    unet = ResidualMultiTaskUNet(num_modalities, initial_num_features, blocks_per_stage_encoder[:len(conv_op_kernel_sizes)], 2,
                        pool_op_kernel_sizes, conv_op_kernel_sizes,
                        get_default_network_config(3, dropout_p=None), num_classes,
                        blocks_per_stage_decoder[:len(conv_op_kernel_sizes)-1], False, False,
                        max_features=max_num_features)# .cuda()

    optimizer = SGD(unet.parameters(), lr=0.1, momentum=0.95)
    loss = MultiTask_DCCE_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
    # loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})

    dummy_input = torch.rand((batch_size, num_modalities, *patch_size))#.cuda()
    dummy_seg_gt = (torch.rand((batch_size, 1, *patch_size)) * num_classes).round().clamp_(0, 2).long()#.cuda().long()
    dummy_class_gt = torch.randn(batch_size, num_classes)
    for _ in range(20):
        optimizer.zero_grad()
        skips = unet.encoder(dummy_input)
        print([i.shape for i in skips])
        #[torch.Size([2, 32, 20, 320, 256]), torch.Size([2, 64, 20, 160, 128]), torch.Size([2, 128, 10, 80, 64]), torch.Size([2, 256, 10, 40, 32])]
        output = unet.decoder(skips)
        #TODO: check chatgpt to massage the right input.
        # input for the cross entropy loss expect a 2d tensor (batch, )
        seg_prediction, class_prediction = output
        l = loss(seg_prediction, class_prediction, dummy_seg_gt, dummy_class_gt)
        l.backward()

        optimizer.step()
        if _ == 0:
            torch.cuda.empty_cache()

        
        # example
        # # Create logits (e.g., from the output of a neural network)
        # logits = torch.randn(8, 4)  # Shape: (batch_size, num_classes)

        # # Ground truth labels (e.g., class indices)
        # targets = torch.randint(0, 4, (8,))  # Shape: (batch_size,)

        # # Instantiate the loss function
        # criterion = nn.CrossEntropyLoss()

        # # Compute the loss (no need to apply softmax to logits)
        # loss = criterion(logits, targets)

        # print(f"Cross Entropy Loss: {loss.item()}")