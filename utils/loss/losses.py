import torch
import torch.nn.functional as F
import dataset.Utils.utils as utils


def fp_edge_loss(gt_edges, edge_logits):
    edges_shape = gt_edges.size()
    gt_edges = gt_edges.view(edges_shape[0], -1)

    loss = F.binary_cross_entropy_with_logits(edge_logits, gt_edges)

    return torch.mean(loss)


def fp_vertex_loss(gt_verts, vertex_logits):
    verts_shape = gt_verts.size()
    gt_verts = gt_verts.view(verts_shape[0], -1)

    loss = F.binary_cross_entropy_with_logits(vertex_logits, gt_verts)

    return torch.mean(loss)


def SurfaceLoss(probs, dis_map):
    assert not utils.one_hot(dis_map)
    pc = probs[:, 1, :].type(torch.float32)
    dc = dis_map[:, 1, ...]

    multipled = torch.mul(pc, dc)
    loss = torch.mean(multipled)
    return loss
