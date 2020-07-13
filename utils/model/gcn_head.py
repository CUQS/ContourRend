import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.contour_mask_utils import *
from modeling.deeplab_resnet import *
from utils.loss import *


"""
DeepLab as Backbone and two branches
"""


class GCN(nn.Module):
    def __init__(self,
                 state_dim=256,
                 feature_dim=256):
        super(GCN, self).__init__()
        self.state_dim = state_dim

        self.gcn_0 = GraphConvolution(feature_dim, 'gcn_0', out_state_dim=self.state_dim)
        self.gcn_res_1 = GraphResConvolution(self.state_dim, 'gcn_res_1')
        self.gcn_res_2 = GraphResConvolution(self.state_dim, 'gcn_res_2')
        self.gcn_res_3 = GraphResConvolution(self.state_dim, 'gcn_res_3')
        self.gcn_res_4 = GraphResConvolution(self.state_dim, 'gcn_res_4')
        self.gcn_res_5 = GraphResConvolution(self.state_dim, 'gcn_res_5')
        self.gcn_res_6 = GraphResConvolution(self.state_dim, 'gcn_res_6')
        self.gcn_7 = GraphConvolution(self.state_dim, 'gcn_7', out_state_dim=32)  # original out_state_dim: 32

        self.fc = nn.Linear(
            in_features=32,
            out_features=2,
        )

    def forward(self, input, adj):
        input1 = self.gcn_0(input, adj)  # (2, 40, 128)
        input2 = self.gcn_res_1(input1, adj)
        input3 = self.gcn_res_2(input2, adj)
        input4 = self.gcn_res_3(input3, adj)
        input5 = self.gcn_res_4(input4, adj)
        input6 = self.gcn_res_5(input5, adj)
        input7 = self.gcn_res_6(input6, adj)  # (2, 40, 128)
        output = self.gcn_7(input7, adj)  # (2, 40, 32)

        return self.fc(output)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, state_dim, name='', out_state_dim=None):
        super(GraphConvolution, self).__init__()
        self.state_dim = state_dim

        if out_state_dim == None:
            self.out_state_dim = state_dim
        else:
            self.out_state_dim = out_state_dim
        self.fc1 = nn.Linear(
            in_features=self.state_dim,
            out_features=self.out_state_dim,
        )

        self.fc2 = nn.Linear(
            in_features=self.state_dim,
            out_features=self.out_state_dim,
        )
        self.name = name

    def forward(self, input_x, adj):
        state_in = self.fc1(input_x)

        ##torch.bmm(p1, p2)
        # p1: the first batch of matrices to be multiplied, p2 is the second
        # eg: p1:(c, w1, h1), p2:(c, w2, h2) -> bmm(p1, p2): (c, w1, h2)
        forward_input = self.fc2(torch.bmm(adj, input_x))

        return state_in + forward_input

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.name + ')'


class GraphResConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, state_dim, name=''):
        super(GraphResConvolution, self).__init__()
        self.state_dim = state_dim

        self.gcn_1 = GraphConvolution(state_dim, '%s_1' % name)
        self.gcn_2 = GraphConvolution(state_dim, '%s_2' % name)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.name = name

    def forward(self, input, adj):
        output_1 = self.gcn_1(input, adj)
        output_1_relu = self.relu1(output_1)

        output_2 = self.gcn_2(output_1_relu, adj)

        output_2_res = output_2 + input

        output = self.relu2(output_2_res)

        return output  # (2, 40, 128)


class FirstAnnotation(nn.Module):
    def __init__(self, feats_dim, feats_channels, internal):
        super(FirstAnnotation, self).__init__()
        self.grad_size = feats_dim

        self.edge_conv = nn.Conv2d(in_channels=feats_channels,
                                   out_channels=internal,
                                   kernel_size=3,
                                   padding=1)

        self.edge_fc = nn.Linear(in_features=feats_dim ** 2 * internal,
                                 out_features=feats_dim ** 2)

        self.vertex_conv = nn.Conv2d(
            in_channels=feats_channels,
            out_channels=internal,
            kernel_size=3,
            padding=1
        )

        self.vertex_fc = nn.Linear(
            in_features=feats_dim ** 2 * internal,
            out_features=feats_dim ** 2
        )

    def forward(self, feats, temperature=0.0, beam_size=1):
        """
        if temperature < 0.01, use greedy
        else, use temperature
        """
        batch_size = feats.size(0)
        conv_edge = self.edge_conv(feats)
        conv_edge = F.relu(conv_edge, inplace=True)
        edge_logits = self.edge_fc(conv_edge.view(batch_size, -1))

        conv_vertex = self.vertex_conv(feats)
        conv_vertex = F.relu(conv_vertex)
        vertex_logits = self.vertex_fc(conv_vertex.view(batch_size, -1))
        logprobs = F.log_softmax(vertex_logits, -1)

        # Sample a first vertex
        if temperature < 0.01:
            logprob, pred_first = torch.topk(logprobs, beam_size, dim=-1)
        else:
            probs = torch.exp(logprobs / temperature)
            pred_first = torch.multinomial(probs, beam_size)

            # Get loogprob of the sampled vertex
            logprob = logprobs.gather(1, pred_first)

        # Remove the last dimension if it is 1
        pred_first = torch.squeeze(pred_first, dim=-1)
        logprob = torch.squeeze(logprob, dim=-1)

        return edge_logits, vertex_logits, logprob, pred_first


class DeepLabGCNModel(nn.Module):
    def __init__(self, cfg):
        super(DeepLabGCNModel, self).__init__()

        self.node_num = cfg.cp_num
        self.cnn_feature_grids = [112, 56, 28, 28]

        self.encoder = DeepLabResnet(gcn_dim=cfg.gcn_dim)
        self.state_dim = self.encoder.gcn_dim + 2

        self.loss_flag = True if cfg.model_type == "train" else False

        self.grid_size = self.encoder.feat_size  # 28*28

        self.psp_feature = [self.cnn_feature_grids[-1]]

        self.gcn_module1 = GCN(self.state_dim, self.state_dim)
        self.gcn_module2 = GCN(self.state_dim, self.state_dim)
        self.gcn_module3 = GCN(self.state_dim, self.state_dim)

        self.first_annotation = FirstAnnotation(28, 512, 16)

        self.mlp = nn.Conv1d(514, 2, 1)

        self.fp_weight = cfg.fp_weight

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                nn.init.constant_(m.bias, 0)

    def point_sample(self, points_pos, feature, **kwargs):
        pos = torch.clamp(points_pos, 0, 1)
        pos_sample = points_pos.unsqueeze(2)
        output = F.grid_sample(feature, 2.0 * pos_sample - 1.0, **kwargs)
        output = output.squeeze(3).permute(0, 2, 1)
        output = torch.cat([output, pos], dim=2)
        return output

    @torch.no_grad()
    def sample_points(self, contours):
        B = contours.size(0)
        if self.loss_flag:
            fit_generation = torch.rand(B, self.node_num*3, 2).cuda(contours.get_device()) * 2.0 - 1.0
            polys_expand = contours.repeat(1, 1, 3).view(B, -1, 2)
            output = polys_expand + fit_generation * 0.09
        else:
            N = 15
            sample_size = 0.09
            ge_base = np.linspace(-sample_size/2.0, sample_size/2.0, N)
            ge_y = ge_base.repeat(N).reshape(N, N)
            ge_x = ge_y.T
            ge_pos = np.stack([ge_x, ge_y], axis=2).reshape((1, N*N, 2))
            fit_generation = torch.from_numpy(ge_pos).cuda(contours.get_device()).float()
            fit_generation = fit_generation.expand(B, -1, 2)
            fit_generation = fit_generation.repeat(1, self.node_num, 1)

            polys_expand = contours.repeat(1, 1, N*N).view(B, -1, 2)
            output = polys_expand + fit_generation
        return output.clamp(0, 1)

    def forward(self, input_x, *kwargs):

        out_dict = {}

        # x_prob: [bs, 2, 224, 224]
        # conv_layers: [bs, 512, 28, 28]
        x_prob, conv_layers_backbone = self.encoder.forward(input_x)

        edge_logits, vertex_logits, logprob, _ = self.first_annotation.forward(conv_layers_backbone)

        edge_logits_f = edge_logits.view(
            (-1, self.first_annotation.grad_size, self.first_annotation.grad_size)).unsqueeze(1)
        vertex_logits_f = vertex_logits.view(
            (-1, self.first_annotation.grad_size, self.first_annotation.grad_size)).unsqueeze(1)
        # [bs, 514, 28, 28]
        feature_with_edges = torch.cat([conv_layers_backbone, edge_logits_f, vertex_logits_f], 1)
        # [bs, state_dim, 28, 28]
        conv_layers = self.encoder.edge_annotation_cnn(feature_with_edges)

        out_dict["pred_polys"] = []

        init_polys = kwargs[1]
        edge_index = kwargs[0]

        for i in range(3):
            # input_feature = self.feature2points(init_polys, conv_layers)
            input_feature = self.point_sample(init_polys, conv_layers, align_corners=False)

            if i == 0:
                diff_pos = self.gcn_module1(input_feature, edge_index)
            elif i == 1:
                diff_pos = self.gcn_module2(input_feature, edge_index)
            else:
                diff_pos = self.gcn_module3(input_feature, edge_index)
            init_polys = init_polys + diff_pos

        out_dict["pred_polys"].append(init_polys)

        mlp_points = self.sample_points(init_polys)
        mlp_feature = self.point_sample(mlp_points, conv_layers_backbone, align_corners=False)
        point_rend = self.mlp(mlp_feature.permute(0, 2, 1))

        out_dict["mlp_points"] = mlp_points
        out_dict["rend"] = point_rend

        if self.loss_flag:
            loss_sum = 0
            loss_sum += match_loss(init_polys, kwargs[2],
                                   kwargs[6], kwargs[7], kwargs[8])

            curr_fp_edge_loss = self.fp_weight * fp_edge_loss(kwargs[3], edge_logits)
            loss_sum += curr_fp_edge_loss

            curr_fp_vertex_loss = self.fp_weight * fp_vertex_loss(kwargs[4], vertex_logits)
            loss_sum += curr_fp_vertex_loss

            output_prob = F.softmax(x_prob, dim=1)
            boundary_loss = SurfaceLoss(output_prob, kwargs[5])
            loss_sum += boundary_loss

            gt_rend = self.point_sample(mlp_points,
                                        kwargs[9],
                                        align_corners=True).long()
            rend_loss = F.cross_entropy(point_rend, gt_rend[:, :, 1]) * 200.0
            loss_sum += rend_loss

            out_dict["loss_sum"] = loss_sum
            out_dict["loss_rend"] = rend_loss

        return out_dict

    def rend_single(self, contours, mlp_point_cpu, mlp_rend_cpu):
        curr_pred_poly = poly01_to_poly0g(contours, 224)
        masks_temp = np.zeros((224, 224), dtype=np.uint8)
        masks_temp = draw_poly(masks_temp, curr_pred_poly)
        mask_rend = torch.from_numpy(masks_temp.astype(np.float32))
        mlp_idx = (mlp_point_cpu[[0], :, 1] * 223).long() * 224 + (mlp_point_cpu[[0], :, 0] * 223).long()
        mask_rend = mask_rend.view(1, -1).scatter_(1, mlp_idx, mlp_rend_cpu[[0], 1, :]).view(224, 224)
        mask_rend = mask_rend.detach().numpy()
        mask_rend[mask_rend > 0.3] = 1.0
        mask_rend[mask_rend <= 0.3] = 0
        mask_rend = erode_dilate(mask_rend)
        mask_rend_org = torch.zeros((224, 224)).view(1, -1).scatter_(1, mlp_idx, mlp_rend_cpu[[0], 1, :]).view(224, 224)
        return mask_rend, mask_rend_org

    def load_weight(self, weight_file):
        self.load_state_dict(torch.load(weight_file))

    def change_mode(self, mode):
        if mode == "train":
            self.loss_flag = True
        else:
            self.loss_flag = False
