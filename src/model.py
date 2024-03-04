import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from STAE import STAE
import numpy as np

###############
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
import imageio
################

def build_grid(resolution):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)

    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.tensor(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)


class SoftPositionEmbed(nn.Module):
    """Adds soft positional embedding with learnable projection."""

    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.

    Args:
      hidden_size: Size of input feature dimension.
      resolution: Tuple of integers specifying width and height of grid.
    """
        super(SoftPositionEmbed, self).__init__()
        self.proj = nn.Linear(4, hidden_size)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        return inputs + self.proj(self.grid)


def spatial_broadcast(slots, resolution):
    """Broadcast slot features to a 2D grid and collapse slot dimension."""
    # `slots` has shape: [batch_size, num_slots, slot_size].
    slots = torch.reshape(slots, [-1, slots.shape[-1]])[:, None, None, :]
    grid = einops.repeat(slots, 'b_n i j d -> b_n (tilei i) (tilej j) d', tilei=resolution[0], tilej=resolution[1])
    # `grid` has shape: [batch_size*num_slots, height, width, slot_size].
    return grid


def spatial_flatten(x):
    return torch.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[-1]])


def unstack_and_split(x, batch_size, num_channels=3):
    """Unstack batch dimension and split into channels and alpha mask."""
    unstacked = einops.rearrange(x, '(b s) c h w -> b s c h w', b=batch_size)
    channels, masks = torch.split(unstacked, [num_channels, 1], dim=2)
    return channels, masks


class SlotAttention(nn.Module):
    """Slot Attention module."""

    def __init__(self, num_slots, encoder_dims, iters=3, hidden_dim=128, eps=1e-8):
        """Builds the Slot Attention module.
        Args:
            iters: Number of iterations.
            num_slots: Number of slots.
            encoder_dims: Dimensionality of slot feature vectors.
            hidden_dim: Hidden layer size of MLP.
            eps: Offset for attention coefficients before normalization.
        """
        super(SlotAttention, self).__init__()

        self.eps = eps
        self.iters = iters
        self.num_slots = num_slots
        self.scale = encoder_dims ** -0.5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.norm_input = nn.LayerNorm(encoder_dims)
        self.norm_slots = nn.LayerNorm(encoder_dims)
        self.norm_pre_ff = nn.LayerNorm(encoder_dims)

        # Parameters for Gaussian init (shared by all slots).
        # self.slots_mu = nn.Parameter(torch.randn(1, 1, encoder_dims))
        # self.slots_sigma = nn.Parameter(torch.randn(1, 1, encoder_dims))

        self.slots_embedding = nn.Embedding(num_slots, encoder_dims)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(encoder_dims, encoder_dims)
        self.project_k = nn.Linear(encoder_dims, encoder_dims)
        self.project_v = nn.Linear(encoder_dims, encoder_dims)

        # Slot update functions.
        self.gru = nn.GRUCell(encoder_dims, encoder_dims)

        hidden_dim = max(encoder_dims, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dims, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, encoder_dims)
        )

    def forward(self, inputs, num_slots=None):
        # inputs has shape [batch_size, num_inputs, inputs_size].
        inputs = self.norm_input(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        # random slots initialization,
        # mu = self.slots_mu.expand(b, n_s, -1)
        # sigma = self.slots_sigma.expand(b, n_s, -1)
        # slots = torch.normal(mu, sigma)

        # learnable slots initialization
        slots = self.slots_embedding(torch.arange(0, n_s).expand(b, n_s).to(self.device))

        # Multiple rounds of attention.
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)  # weighted mean.

            updates = torch.einsum('bjd,bij->bid', v, attn)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


class MIA(nn.Module):
    def __init__(self, input_channels, output_channels, alpha, beta):
        super(MIA, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(output_channels, output_channels, kernel_size=1, padding=0)
        self.bn = nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True)

    def forward(self, flow, res_img,frameImg_stack):
        res_img = self.bn(res_img)
        frameImg_stack = F.interpolate(frameImg_stack, size=(16, 28), mode='bicubic', align_corners=False)
        out_1 = torch.cat([flow, res_img], dim=1)
        out_1 = F.relu(self.conv1(out_1))
        out_1 = self.conv2(out_1)
        out_middle = flow*self.alpha + out_1*(1-self.alpha)
        out = torch.cat([out_middle, frameImg_stack], dim=1)
        out = F.relu(self.conv3(out))
        out = self.conv4(out)
        out=flow*self.beta+out*(1-self.beta)
        return out


################################################################################################
affine_par = True


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ASPP(nn.Module):
    def __init__(self, dilation_series, padding_series, depth, res_type=None):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        if res_type == "res":
            self.conv = nn.Conv2d(256, depth, 1, 1)
        elif res_type is None:
            self.conv = nn.Conv2d(2048, depth, 1, 1)
        self.bn_x = nn.BatchNorm2d(depth)
        if res_type == "res":
            self.conv2d_0 = nn.Conv2d(256, depth, kernel_size=1, stride=1)
        elif res_type is None:
            self.conv2d_0 = nn.Conv2d(2048, depth, kernel_size=1, stride=1)
        self.bn_0 = nn.BatchNorm2d(depth)
        if res_type == "res":
            self.conv2d_1 = nn.Conv2d(256, depth, kernel_size=3, stride=1, padding=padding_series[0],
                                      dilation=dilation_series[0])
        elif res_type is None:
            self.conv2d_1 = nn.Conv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[0],
                                      dilation=dilation_series[0])
        self.bn_1 = nn.BatchNorm2d(depth)
        if res_type == "res":
            self.conv2d_2 = nn.Conv2d(256, depth, kernel_size=3, stride=1, padding=padding_series[1],
                                      dilation=dilation_series[1])
        elif res_type is None:
            self.conv2d_2 = nn.Conv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[1],
                                      dilation=dilation_series[1])
        self.bn_2 = nn.BatchNorm2d(depth)
        if res_type == "res":
            self.conv2d_3 = nn.Conv2d(256, depth, kernel_size=3, stride=1, padding=padding_series[2],
                                      dilation=dilation_series[2])
        elif res_type is None:
            self.conv2d_3 = nn.Conv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[2],
                                      dilation=dilation_series[2])
        self.bn_3 = nn.BatchNorm2d(depth)
        self.relu = nn.ReLU(inplace=True)
        self.bottleneck = nn.Conv2d(depth * 5, 256, kernel_size=3, padding=1)  # 512 1x1Conv
        self.bn = nn.BatchNorm2d(256)
        self.prelu = nn.PReLU()
        # for m in self.conv2d_list:
        #    m.weight.data.normal_(0, 0.01)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_stage_(self, dilation1, padding1):
        Conv = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=padding1, dilation=dilation1, bias=True)  # classes
        Bn = nn.BatchNorm2d(256)
        Relu = nn.ReLU(inplace=True)
        return nn.Sequential(Conv, Bn, Relu)

    def forward(self, x):
        # out = self.conv2d_list[0](x)
        # mulBranches = [conv2d_l(x) for conv2d_l in self.conv2d_list]
        size = x.shape[2:]
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = self.bn_x(image_features)
        image_features = self.relu(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear', align_corners=True)
        out_0 = self.conv2d_0(x)
        out_0 = self.bn_0(out_0)
        out_0 = self.relu(out_0)
        out_1 = self.conv2d_1(x)
        out_1 = self.bn_1(out_1)
        out_1 = self.relu(out_1)
        out_2 = self.conv2d_2(x)
        out_2 = self.bn_2(out_2)
        out_2 = self.relu(out_2)
        out_3 = self.conv2d_3(x)
        out_3 = self.bn_3(out_3)
        out_3 = self.relu(out_3)
        out = torch.cat([image_features, out_0, out_1, out_2, out_3], 1)
        out = self.bottleneck(out)
        out = self.bn(out)
        out = self.prelu(out)
        # for i in range(len(self.conv2d_list) - 1):
        #    out += self.conv2d_list[i + 1](x)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(ASPP, [6, 12, 18], [6, 12, 18], 512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, num_classes):
        return block(dilation_series, padding_series, num_classes)

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        fea = self.layer5(x)
        return fea


class Interactive_Feature(nn.Module):
    def __init__(self, num_classes=1, all_channel=256, all_dim=60 * 60):  # 473./8=60
        super(Interactive_Feature, self).__init__()
        self.linear_e = nn.Linear(all_channel, all_channel, bias=False)
        self.channel = all_channel
        self.dim = all_dim
        self.gate = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.conv1 = nn.Conv2d(all_channel * 2, all_channel, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(all_channel * 2, all_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(all_channel)
        self.bn2 = nn.BatchNorm2d(all_channel)
        self.prelu = nn.ReLU(inplace=True)
        self.adjust_channels=nn.Conv2d(2*all_channel,out_channels=all_channel,kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # init.xavier_normal(m.weight.data)
                # m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, exemplar, query):

        fea_size = query.size()[2:]
        all_dim = fea_size[0] * fea_size[1]
        exemplar_flat = exemplar.view(-1, query.size()[1], all_dim)  # N,C,H*W
        query_flat = query.view(-1, query.size()[1], all_dim)
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batch size x dim x num
        exemplar_corr = self.linear_e(exemplar_t)  #
        A = torch.bmm(exemplar_corr, query_flat)
        #################################
        min_A=torch.min(A)
        max_A=torch.max(A)
        reversed_A=(A-min_A)/(max_A-min_A)
        reversed_A=1-reversed_A
        reversed_A=reversed_A*(max_A-min_A)+min_A
        A=reversed_A
        #################################
        A1 = F.softmax(A.clone(), dim=1)  #
        B = F.softmax(torch.transpose(A, 1, 2), dim=1)

        query_att = torch.bmm(exemplar_flat, A1).contiguous()
        exemplar_att = torch.bmm(query_flat, B).contiguous()

        input1_att = exemplar_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
        input2_att = query_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
        input1_mask = self.gate(input1_att)
        input2_mask = self.gate(input2_att)
        input1_mask = self.gate_s(input1_mask)
        input2_mask = self.gate_s(input2_mask)
        input1_att = input1_att * input1_mask
        input2_att = input2_att * input2_mask
        input1_att = torch.cat([input1_att, exemplar], 1)
        input2_att = torch.cat([input2_att, query], 1)
        input1_att = self.conv1(input1_att)
        input2_att = self.conv2(input2_att)
        input1_att = self.bn1(input1_att)
        input2_att = self.bn2(input2_att)
        input1_att = self.prelu(input1_att)
        input2_att = self.prelu(input2_att)

        return input1_att,input2_att  # shape: NxCx


################################################################################################
class SlotAttentionAutoEncoder(nn.Module):
    """Slot Attention-based auto-encoder for object discovery."""

    def __init__(self, resolution, num_slots, in_out_channels=3, iters=5):
        """Builds the Slot Attention-based Auto-encoder.

        Args:
            re solution: Tuple of integers specifying width and height of input image
            num_slots: Number of slots in Slot Attention.
            iters: Number of iterations in Slot Attention.
        """
        super(SlotAttentionAutoEncoder, self).__init__()

        self.iters = iters
        self.num_slots = num_slots
        self.resolution = resolution
        self.in_out_channels = in_out_channels
        self.encoder_arch = [64, 'MP', 128, 'MP', 256]
        self.encoder_dims = self.encoder_arch[-1]
        self.encoder_cnn, ratio = self.make_encoder(
            self.in_out_channels, self.encoder_arch)
        self.encoder_end_size = (int(resolution[0] / ratio), int(resolution[1] / ratio))
        self.encoder_pos = SoftPositionEmbed(self.encoder_dims, self.encoder_end_size)
        self.decoder_initial_size = (int(resolution[0] / 8), int(resolution[1] / 8))
        self.decoder_pos = SoftPositionEmbed(self.encoder_dims, self.decoder_initial_size)

        self.layer_norm = nn.LayerNorm(self.encoder_dims)

        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_dims, self.encoder_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_dims, self.encoder_dims)
        )

        self.slot_attention = SlotAttention(
            iters=self.iters,
            num_slots=self.num_slots,
            encoder_dims=self.encoder_dims,
            hidden_dim=self.encoder_dims)

        self.resnet = ResNet(Bottleneck, [3, 4, 23, 3], 1)
        self.stae = STAE(project_dims=256, batch_size=1)
        self.mia = MIA(input_channels=512, output_channels=256,alpha=0.75, beta=0.75)
        self.aspp_res_img = ASPP([6, 12, 18], [6, 12, 18], 1, "res")
        self.interactive_Feature = Interactive_Feature()

        self.decoder_cnn = nn.Sequential(
            # [batch_size, 64, h*2, w*2]
            nn.ConvTranspose2d(self.encoder_dims, 64, kernel_size=5, padding=2, output_padding=1, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            # [batch_size, 64, h*4, w*4]
            nn.ConvTranspose2d(64, 64, kernel_size=5, padding=2, output_padding=1, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            # [batch_size, 64, h*8, w*8]
            nn.ConvTranspose2d(64, 64, kernel_size=5, padding=2, output_padding=1, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            # [batch_size, 64, h*8, w*8]
            nn.Conv2d(64, 64, kernel_size=5, padding=2, stride=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_out_channels + 1, kernel_size=5, padding=2, stride=1)

        )

    def make_encoder(self, in_channels, encoder_arch):
        layers = []
        down_factor = 0
        for v in encoder_arch:
            if v == 'MP':
                layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]
                down_factor += 1
            else:
                conv1 = nn.Conv2d(in_channels, v, kernel_size=5, padding=2)
                conv2 = nn.Conv2d(v, v, kernel_size=5, padding=2)

                layers += [conv1, nn.InstanceNorm2d(v, affine=True), nn.ReLU(inplace=True),
                           conv2, nn.InstanceNorm2d(v, affine=True), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers), 2 ** down_factor

    def forward(self, flow, res_img, frameImg):
        frameImg_feat = self.resnet(frameImg)
        batch_size = 1  # not to set 2
        frameImg_add = []
        if frameImg.shape[0] == 4 * batch_size and flow.shape[0] == 2 * batch_size:
            for batch in range(batch_size):
                frameImg_0, frameImg_1 = self.interactive_Feature(frameImg_feat[batch * 4 + 0].unsqueeze(0),
                                                                               frameImg_feat[batch * 4 + 1].unsqueeze(
                                                                                   0))  # [2,256,17,29]
                frameImg_2, frameImg_3= self.interactive_Feature(frameImg_feat[batch * 4 + 2].unsqueeze(0),
                                                                               frameImg_feat[batch * 4 + 3].unsqueeze(
                                                                                   0))  # [2,256,17,29]
                frameImg_add.append(frameImg_0)
                frameImg_add.append(frameImg_1)
                frameImg_add.append(frameImg_2)
                frameImg_add.append(frameImg_3)
        elif frameImg.shape[0] == 8 and flow.shape[0] == 4:
            frameImg_0, frameImg_1 = self.interactive_Feature(frameImg_feat[0].unsqueeze(0),
                                                       frameImg_feat[1].unsqueeze(0))  # [2,256,17,29]
            frameImg_2, frameImg_3 = self.interactive_Feature(frameImg_feat[2].unsqueeze(0),
                                                       frameImg_feat[3].unsqueeze(0))  # [2,256,17,29]
            frameImg_4, frameImg_5 = self.interactive_Feature(frameImg_feat[4].unsqueeze(0),
                                                       frameImg_feat[5].unsqueeze(0))  # [2,256,17,29]
            frameImg_6, frameImg_7 = self.interactive_Feature(frameImg_feat[6].unsqueeze(0),
                                                       frameImg_feat[7].unsqueeze(0))  # [2,256,17,29]
            frameImg_add.append(frameImg_0)
            frameImg_add.append(frameImg_1)
            frameImg_add.append(frameImg_2)
            frameImg_add.append(frameImg_3)
            frameImg_add.append(frameImg_4)
            frameImg_add.append(frameImg_5)
            frameImg_add.append(frameImg_6)
            frameImg_add.append(frameImg_7)
        frameImg_stack = torch.stack(frameImg_add)
        frameImg_stack = einops.rearrange(frameImg_stack, 'b t c h w -> (b t) c h w')
        x = self.encoder_cnn(flow)
        res_img_expend = res_img.expand(-1, 3, -1, -1)
        res_img_expend = self.encoder_cnn(res_img_expend)
        flow_stae = self.stae(x, frameImg_feat, res_img_expend)
        x = einops.rearrange(flow_stae, 'b c h w -> b h w c')
        x = self.encoder_pos(x)  # Position embedding.
        x = spatial_flatten(x)  # Flatten spatial dimensions (treat flow as set).
        x = self.mlp(self.layer_norm(x))  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        slots = self.slot_attention(x)
        # `slots` has shape: [batch_size, num_slots, slot_size].
        # Spatial broadcast decoder.
        x = spatial_broadcast(slots, self.decoder_initial_size)

        # `x` has shape: [batch_size*num_slots, height_init, width_init, slot_size].
        x = self.decoder_pos(x)
        x = einops.rearrange(x, 'b_n h w c -> b_n c h w')

        res_img_copy = self.aspp_res_img(res_img_expend)
        # res_img_copy=torch.cat((res_img_copy,res_img_copy),dim=0)
        res_img_copy = res_img_copy.repeat_interleave(2, dim=0)
        x = self.mia(x, res_img_copy, frameImg_stack)

        x = self.decoder_cnn(x)

        # `x` has shape: [batch_size*num_slots, num_channels+1, height, width].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = unstack_and_split(x, batch_size=flow.shape[0], num_channels=self.in_out_channels)
        # `recons` has shape: [batch_size, num_slots, num_channels, height, width].
        # `masks` has shape: [batch_size, num_slots, 1, height, width].

        # Normalize alpha masks over slots.
        masks = torch.softmax(masks, axis=1)
        # x_mask_stack = einops.rearrange(x_mask_stack, '(b s) c h w -> b s c h w', b=flow.shape[0])
        # masks = masks

        recon_combined = torch.sum(recons * masks, axis=1)  # Recombine image.
        # `recon_combined` has shape: [batch_size, num_channels, height, width].
        return recon_combined, recons, masks, slots
