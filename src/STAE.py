from torch import nn
import torch
from torch.nn import functional as F
import einops
from data import readFlow
from data import readRGB


class Predictor(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Predictor, self).__init__()

        def Conv2D(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                          dilation=dilation, bias=True),
                nn.LeakyReLU(0.1)
            )

        self.conv0 = Conv2D(ch_in, 8, kernel_size=3, stride=1)
        dd = 8
        self.conv1 = Conv2D(ch_in + dd, 8, kernel_size=3, stride=1)
        dd += 8
        self.conv2 = Conv2D(ch_in + dd, 6, kernel_size=3, stride=1)
        dd += 6
        self.conv3 = Conv2D(ch_in + dd, 4, kernel_size=3, stride=1)
        dd += 4
        self.conv4 = Conv2D(ch_in + dd, 2, kernel_size=3, stride=1)
        dd += 2
        self.predict_flow = nn.Conv2d(ch_in + dd, ch_out, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = torch.cat((self.conv0(x), x), 1)
        x = torch.cat((self.conv1(x), x), 1)
        x = torch.cat((self.conv2(x), x), 1)
        x = torch.cat((self.conv3(x), x), 1)
        x = torch.cat((self.conv4(x), x), 1)
        return self.predict_flow(x)


class STAE(nn.Module):
    def __init__(self, project_dims, batch_size, in_dim=3):
        super().__init__()
        self.in_dim = in_dim
        self.batch_size = batch_size
        # i_feature_dims = [96, 192, 384, 768]
        # p_feature_dims = [64, 128, 256, 512]
        self.channel_weight_predictors = nn.Sequential(
            nn.Linear(project_dims, project_dims),
            nn.ReLU(),
            nn.Linear(project_dims, 256),
            nn.Sigmoid(),
        ).to("cuda")

        self.spatial_modules = Predictor(project_dims * 3, 1).to("cuda")
        self.channel_modules = Predictor(project_dims * 3, project_dims).to("cuda")

    def forward(self, flow, img, resimg):
        if img.shape[0] == 4 * self.batch_size and flow.shape[0] == 2 * self.batch_size:
            img_slices = [img[i:i + 4] for i in range(0, 4 * self.batch_size, 4)]
            flow_slices = [flow[i:i + 2] for i in range(0, 2 * self.batch_size, 2)]
            res_slices = [resimg[i:i + 2] for i in range(0, 2 * self.batch_size, 2)]
            flow_output = []
            for batch in range(self.batch_size):
                img_down_resized = F.interpolate(img_slices[batch], size=flow_slices[batch].shape[-2:],
                                                 mode='bilinear',
                                                 align_corners=False)
                for i in range(0, 2):
                    channel_weight = self.channel_modules(
                        torch.cat(
                            [flow_slices[batch][i].unsqueeze(0) * 1.0,
                             img_down_resized[i * 2 + 0].unsqueeze(0) * 0.5,
                             res_slices[batch][i].unsqueeze(0) * 0.5], dim=1))
                    weight = self.channel_weight_predictors(
                        F.adaptive_avg_pool2d(channel_weight, 1).flatten(1))  # (300, 256)

                    weight = weight.unsqueeze(-1).unsqueeze(-1)

                    res_channel = (res_slices[batch][i].unsqueeze(0) * weight)
                    spatial_weight = self.spatial_modules(
                        torch.cat(
                            [flow_slices[batch][i].unsqueeze(0) * 1.0,
                             img_down_resized[i * 2 + 0].unsqueeze(0) * 0.5,
                             res_slices[batch][i].unsqueeze(0) * 0.5], dim=1))
                    spatial_weight = F.softmax(spatial_weight.view(*spatial_weight.shape[:2], -1), dim=-1).view_as(
                        spatial_weight)
                    res_spatial = res_channel * spatial_weight
                    flow_fusion_00 = res_spatial + flow_slices[batch][i].unsqueeze(0)
                    channel_weight = self.channel_modules(
                        torch.cat(
                            [flow_slices[batch][i].unsqueeze(0) * 1.0,
                             img_down_resized[i * 2 + 1].unsqueeze(0) * 0.5,
                             res_slices[batch][i].unsqueeze(0) * 0.5], dim=1))
                    weight = self.channel_weight_predictors(
                        F.adaptive_avg_pool2d(channel_weight, 1).flatten(1))  # (300, 256)
                    weight = weight.unsqueeze(-1).unsqueeze(-1)
                    res_channel = (res_slices[batch][i].unsqueeze(0) * weight)
                    spatial_weight = self.spatial_modules(
                        torch.cat(
                            [flow_slices[batch][i].unsqueeze(0) * 1.0,
                             img_down_resized[i * 2 + 1].unsqueeze(0) * 0.5,
                             res_slices[batch][i].unsqueeze(0) * 0.5], dim=1))
                    spatial_weight = F.softmax(spatial_weight.view(*spatial_weight.shape[:2], -1), dim=-1).view_as(
                        spatial_weight)
                    res_spatial = res_channel * spatial_weight
                    flow_fusion_01 = res_spatial + flow_slices[batch][i].unsqueeze(0)
                    flow_output.append((flow_fusion_00 * 0.5 + flow_fusion_01 * 0.5))
            x_fusion_final = torch.stack(flow_output, dim=0)
            x_fusion_final = einops.rearrange(x_fusion_final, 'b t c h w -> (b t) c h w')



        elif img.shape[0] == 8 and flow.shape[0] == 4:
            img_slices = [img[i:i + 4] for i in range(0, 4 * 2, 4)]
            flow_slices = [flow[i:i + 2] for i in range(0, 2 * 2, 2)]
            res_slices = [resimg[i:i + 2] for i in range(0, 2 * 2, 2)]
            flow_output = []
            for batch in range(2):
                img_down_resized = F.interpolate(img_slices[batch], size=flow_slices[batch].shape[-2:],
                                                 mode='bilinear',
                                                 align_corners=False)
                for i in range(0, 2):
                    channel_weight = self.channel_modules(
                        torch.cat(
                            [flow_slices[batch][i].unsqueeze(0), img_down_resized[i * 2 + 0].unsqueeze(0),
                             res_slices[batch][i].unsqueeze(0)], dim=1))
                    weight = self.channel_weight_predictors(
                        F.adaptive_avg_pool2d(channel_weight, 1).flatten(1))  # (300, 256)
                    weight = weight.unsqueeze(-1).unsqueeze(-1)
                    res_channel = res_slices[batch][i].unsqueeze(0) * weight
                    spatial_weight = self.spatial_modules(
                        torch.cat(
                            [flow_slices[batch][i].unsqueeze(0), img_down_resized[i * 2 + 0].unsqueeze(0),
                             res_slices[batch][i].unsqueeze(0)], dim=1))
                    spatial_weight = F.softmax(spatial_weight.view(*spatial_weight.shape[:2], -1), dim=-1).view_as(
                        spatial_weight)
                    res_spatial = res_channel * spatial_weight
                    flow_fusion_00 = res_spatial + flow_slices[batch][i].unsqueeze(0)

                    channel_weight = self.channel_modules(
                        torch.cat(
                            [flow_slices[batch][i].unsqueeze(0), img_down_resized[i * 2 + 1].unsqueeze(0),
                             res_slices[batch][i].unsqueeze(0)], dim=1))
                    weight = self.channel_weight_predictors(
                        F.adaptive_avg_pool2d(channel_weight, 1).flatten(1))  # (300, 256)
                    weight = weight.unsqueeze(-1).unsqueeze(-1)
                    res_channel = res_slices[batch][i].unsqueeze(0) * weight
                    spatial_weight = self.spatial_modules(
                        torch.cat(
                            [flow_slices[batch][i].unsqueeze(0), img_down_resized[i * 2 + 1].unsqueeze(0),
                             res_slices[batch][i].unsqueeze(0)], dim=1))
                    spatial_weight = F.softmax(spatial_weight.view(*spatial_weight.shape[:2], -1), dim=-1).view_as(
                        spatial_weight)
                    res_spatial = res_channel * spatial_weight
                    flow_fusion_01 = res_spatial + flow_slices[batch][i].unsqueeze(0)
                    flow_output.append((flow_fusion_00 * 0.5 + flow_fusion_01 * 0.5))
            x_fusion_final = torch.stack(flow_output, dim=0)
            x_fusion_final = einops.rearrange(x_fusion_final, 'b t c h w -> (b t) c h w')

        return x_fusion_final


if __name__ == '__main__':
    img_forward = readRGB(sample_dir="../data/DAVIS2016/JPEGImages/480p/blackswan/00000.jpg",
                          resolution=(128, 224))
    img_back = readRGB("../data/DAVIS2016/JPEGImages/480p/blackswan/00001.jpg", resolution=(128, 224))
    flow, scale_w, scale_h = readFlow("../data/DAVIS2016/Flows_gap1/blackswan/00000.flo", resolution=(128, 224),
                                      to_rgb=True)
    resimg = readRGB("../data/DAVIS2016/residualsImages_gap1/blackswan/00000.png", resolution=(128, 224))

    img_forward = torch.from_numpy(img_forward).unsqueeze(0).to(torch.float32)
    img_back = torch.from_numpy(img_back).unsqueeze(0).to(torch.float32)
    flow = torch.from_numpy(flow).unsqueeze(0).to(torch.float32)
    resimg = torch.from_numpy(resimg).unsqueeze(0).to(torch.float32)
    stae = STAE(project_dims=128)
    flow_stae = stae(flow, img_forward, resimg)