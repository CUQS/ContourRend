import torch


def match_loss(pred, gt, pid, pdfv, pcid, loss_type="L2"):
    pnum = gt.size(1)
    pnum_c = pred.size(1)
    fill_gap = pnum // pnum_c
    bs = pred.size(0)
    p_id = pid.expand(bs, pid.size(1), 2)
    pdf_v = pdfv.expand(bs, pnum, 2)
    pc_id = pcid.expand(bs, pnum_c, 2)

    gt_expand = gt.expand(bs, pnum, 2)
    gt_expand = torch.gather(gt_expand, 1, p_id).view(bs, pnum, pnum, 2).detach()

    # sample pred
    pred_c = pred

    pred_shift = pred_c.gather(1, pc_id)
    pred_diff = (pred_shift - pred_c) / fill_gap
    pred_diff = pred_diff.repeat(1, 1, fill_gap).view(bs, -1, 2)

    pred_c = pred_c.repeat(1, 1, fill_gap).view(bs, -1, 2)
    pdf_c = pdf_v * pred_diff
    pred_c = pred_c + pdf_c
    pred_expand = pred_c.repeat(1, pnum, 1).view(bs, pnum, pnum, 2)

    dis = pred_expand - gt_expand

    if loss_type == "L2":
        dis = (dis ** 2).sum(3).sqrt().sum(2)
    elif loss_type == "L1":
        dis = torch.abs(dis).sum(3).sum(2)

    min_dis, min_id = torch.min(dis, dim=1, keepdim=True)
    # print(min_id)

    # min_id = torch.from_numpy(min_id.data.cpu().numpy()).to(device)
    # min_gt_id_to_gather = min_id.unsqueeze_(2).unsqueeze_(3).long().\
    #                         expand(min_id.size(0), min_id.size(1), gt_expand.size(2), gt_expand.size(3))
    # gt_right_order = torch.gather(gt_expand, 1, min_gt_id_to_gather).view(batch_size, pnum, 2)

    return torch.mean(min_dis)
