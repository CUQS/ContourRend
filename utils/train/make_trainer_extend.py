import json
import time
from tqdm import tqdm
from .data_parallel import UserScatteredDataParallel
import torch
import torch.nn as nn
from utils.contour_mask_utils import *

import matplotlib.pyplot as plt
import numpy as np


class TrainerExtend(object):
    def __init__(self, model, cfg):
        gpus = cfg.gpus.split(",")
        gpus = [int(x) for x in gpus]
        model = UserScatteredDataParallel(model, device_ids=gpus)
        model = model.cuda()
        self.model = model
        self.gpus = gpus

        no_wd = []
        wd = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "bn" in name or "bias" in name:
                no_wd.append(p)
            else:
                wd.append(p)
        self.optimizer = torch.optim.Adam([
            {"params": no_wd, "weight_decay": 0.0},
            {"params": wd}
        ], lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=False)
        self.lr_decay = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg.lr_decay,
                                                        gamma=0.1)

        self.grad_clip = cfg.grad_clip

        self.epoch = cfg.epoch
        self.val_iter = cfg.val_iter
        self.save_iter = cfg.save_iter
        self.data_name = cfg.data_name
        self.save_demo = cfg.test_save_demo
        self.cp_num = cfg.cp_num
        self.pnum = cfg.p_num
        self.fill_gap = self.pnum // self.cp_num

    def train(self, dataset, epoch_i):
        batch_all = len(dataset)
        batch = 0
        self.model.train()
        self.model.module.change_mode("train")
        running_loss = 0.0
        running_loss_rend = 0.0
        running_iou = 0.0
        print("lr: ", self.optimizer.param_groups[0]["lr"])

        pidxall = np.zeros(shape=(1, self.pnum, self.pnum), dtype=np.int32)
        for i in range(self.pnum):
            pidx = (np.arange(self.pnum) + i) % self.pnum
            pidxall[0, i] = pidx
        pidxall = torch.from_numpy(np.reshape(pidxall, newshape=(1, -1)))

        p_id = pidxall.unsqueeze_(2).long().expand(pidxall.size(0), pidxall.size(1), 2)
        pdf_v = torch.arange(self.fill_gap).repeat(2 * self.cp_num).view(2, -1).t().view(1, -1, 2).float()

        pc_id = torch.arange(self.cp_num)
        pc_id = torch.cat([pc_id[1:], pc_id[0].unsqueeze(0)]).repeat(2).view(2, -1).t().view(1, -1, 2)

        for data in tqdm(dataset):

            if len(data["img"]) == 1:
                continue

            img_x = data["img"]
            init_polys = data["fwd_poly"]
            edge_index_tensor = create_adjacency_matrix_cat(img_x.size(0), 4, self.cp_num)
            edge_index_tensor = torch.from_numpy(edge_index_tensor)

            # sub-batch
            per_gpu = img_x.size(0) // len(self.gpus)
            input_x = []

            for gpu_i in self.gpus:
                mask_bsi = data["gt_mask"][gpu_i * per_gpu:(gpu_i + 1) * per_gpu]
                mask_bsi = mask_bsi.unsqueeze(1).expand(mask_bsi.size(0), 2, 224, 224)
                input_x.append([img_x[gpu_i * per_gpu:(gpu_i + 1) * per_gpu, ...],
                                edge_index_tensor[gpu_i * per_gpu:(gpu_i + 1) * per_gpu, ...],
                                init_polys[gpu_i * per_gpu:(gpu_i + 1) * per_gpu, ...],
                                data["gt_poly"][gpu_i * per_gpu:(gpu_i + 1) * per_gpu, ...],
                                data["edge_mask"][gpu_i * per_gpu:(gpu_i + 1) * per_gpu, ...],
                                data["vertex_mask"][gpu_i * per_gpu:(gpu_i + 1) * per_gpu, ...],
                                data["mask_distmap"][gpu_i * per_gpu:(gpu_i + 1) * per_gpu, ...],
                                p_id, pdf_v, pc_id, mask_bsi])

            self.optimizer.zero_grad()

            output = self.model(input_x)

            loss_sum = output["loss_sum"]
            loss_sum = torch.mean(loss_sum)
            loss_rend = output["loss_rend"]
            loss_rend = torch.mean(loss_rend)

            loss_sum.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            pred_polys = output["pred_polys"][-1]
            preds = pred_polys.detach().cpu().numpy()
            iou = 0
            orig_poly = data["orig_poly"]

            for i in range(preds.shape[0]):
                # gcn iou
                curr_pred_poly = poly01_to_poly0g(preds[i], 224)
                curr_gt_poly = poly01_to_poly0g(orig_poly[i], 224)
                masks_temp = np.zeros((2, 224, 224), dtype=np.uint8)
                masks_temp[0] = draw_poly(masks_temp[0], curr_pred_poly)
                masks_temp[1] = draw_poly(masks_temp[1], curr_gt_poly)
                cur_iou = iou_from_mask(masks_temp[0], masks_temp[1])
                iou += cur_iou

            iou = iou / preds.shape[0]

            running_iou += iou
            running_loss += loss_sum.item()
            running_loss_rend += loss_rend.item()

            if batch % self.val_iter == (self.val_iter - 1):
                bs = img_x.shape[0]
                running_loss = running_loss / self.val_iter
                running_iou = running_iou / self.val_iter
                running_loss_rend = running_loss_rend / self.val_iter
                data_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                json_data = {"time": data_time,
                             "epoch": epoch_i,
                             "batch": batch * bs,
                             "loss": running_loss,
                             "loss_rend": running_loss_rend,
                             "iou": running_iou}
                self.dump_data(json_data, data_time)
                running_loss = 0.0
                running_iou = 0.0
                running_loss_rend = 0.0

            if ((batch % (batch_all // 3)) == batch_all // 3 - 1) or batch == 0:
                save_path = self.data_name + "/epoch{}_{}.pth".format(epoch_i, batch * preds.shape[0])
                torch.save(self.model.module.state_dict(), save_path)
                print("\nsaved: ", save_path)
            batch += 1

        self.lr_decay.step()

    def val(self, dataset):
        self.model.eval()
        self.model.module.change_mode("test")
        torch.cuda.empty_cache()

        pidxall = np.zeros(shape=(1, self.pnum, self.pnum), dtype=np.int32)
        for i in range(self.pnum):
            pidx = (np.arange(self.pnum) + i) % self.pnum
            pidxall[0, i] = pidx
        pidxall = torch.from_numpy(np.reshape(pidxall, newshape=(1, -1)))

        p_id = pidxall.unsqueeze_(2).long().expand(pidxall.size(0), pidxall.size(1), 2)
        pdf_v = torch.arange(self.fill_gap).repeat(2 * self.cp_num).view(2, -1).t().view(1, -1, 2).float()

        pc_id = torch.arange(self.cp_num)
        pc_id = torch.cat([pc_id[1:], pc_id[0].unsqueeze(0)]).repeat(2).view(2, -1).t().view(1, -1, 2)

        evaluate_df = {"cat": [], "iou": [], "iou_rend": []}
        val_result_draw = []
        val_cat = {"car": [0, 5], "person": [0, 4], "bicycle": [0, 4],
                   "bus": [0, 2], "train": [0, 1], "truck": [0, 2],
                   "rider": [0, 2], "motorcycle": [0, 1]}
        val_count = 0
        for data in tqdm(dataset):
            with torch.no_grad():

                if len(data["img"]) == 1:
                    continue

                img_x = data["img"]
                init_polys = data["fwd_poly"]

                edge_index_tensor = create_adjacency_matrix_cat(img_x.size(0), 4, self.cp_num)
                edge_index_tensor = torch.from_numpy(edge_index_tensor)

                # sub-batch
                per_gpu = img_x.size(0) // len(self.gpus)
                input_x = []
                for gpu_i in self.gpus:
                    mask_bsi = data["gt_mask"][gpu_i * per_gpu:(gpu_i + 1) * per_gpu]
                    mask_bsi = mask_bsi.unsqueeze(1).expand(mask_bsi.size(0), 2, 224, 224)
                    input_x.append([img_x[gpu_i * per_gpu:(gpu_i + 1) * per_gpu, ...],
                                    edge_index_tensor[gpu_i * per_gpu:(gpu_i + 1) * per_gpu, ...],
                                    init_polys[gpu_i * per_gpu:(gpu_i + 1) * per_gpu, ...],
                                    data["gt_poly"][gpu_i * per_gpu:(gpu_i + 1) * per_gpu, ...],
                                    data["edge_mask"][gpu_i * per_gpu:(gpu_i + 1) * per_gpu, ...],
                                    data["vertex_mask"][gpu_i * per_gpu:(gpu_i + 1) * per_gpu, ...],
                                    data["mask_distmap"][gpu_i * per_gpu:(gpu_i + 1) * per_gpu, ...],
                                    p_id, pdf_v, pc_id, mask_bsi])

                output = self.model(input_x)

                mlp_points = output["mlp_points"].detach().cpu()  # (bs, 60*9*9, 2)
                mlp_rend = output["rend"].detach().cpu().softmax(1).clamp(0, 1)  # (bs, 2, 60*9*9)
                pred_polys = output["pred_polys"][-1]
                pred_polys = pred_polys.detach().cpu().numpy()

                orig_poly = data["orig_poly"]
                gt_mask = data["gt_mask"].detach().numpy()

                for i in range(pred_polys.shape[0]):
                    curr_pred_poly = poly01_to_poly0g(pred_polys[i], 224)
                    curr_gt_poly = poly01_to_poly0g(orig_poly[i], 224)
                    masks_temp = np.zeros((224, 224), dtype=np.uint8)
                    masks_temp = draw_poly(masks_temp, curr_pred_poly)
                    iou_i = iou_from_mask(masks_temp, gt_mask[i].astype(np.uint8))
                    mask_rend, mask_rend_org = self.model.module.rend_single(pred_polys[i], mlp_points[[i]], mlp_rend[[i]])
                    iou_rend = iou_from_mask(mask_rend.astype(np.uint8), gt_mask[i].astype(np.uint8))
                    evaluate_df["iou"].append(iou_i)
                    evaluate_df["iou_rend"].append(iou_rend)
                    evaluate_df["cat"].append(data["label"][i])

                    label = data["label"][i]

                    if label == "bicycle":
                        iou_thresh = 0.61
                    elif label in ["bus", "person", "truck", "car", "rider"]:
                        iou_thresh = 0.68
                    elif label == "train":
                        iou_thresh = 0.5
                    else:
                        iou_thresh = 0.58
                    if val_count < 21 and iou_i > iou_thresh and data["patch_w"][i] > 150 and iou_rend > iou_i + 0.015:
                        img_t = img_x[i].permute(1, 2, 0).detach().numpy()

                        for cat in val_cat.keys():
                            if data["label"][i] == cat and val_cat[cat][0] < val_cat[cat][1]:
                                val_cat[cat][0] += 1
                                if val_count < 21:
                                    val_count += 1
                                    val_result_draw.append(
                                        [img_t, curr_pred_poly, gt_mask[i].astype(np.uint8), mask_rend,
                                         iou_i, iou_rend, mask_rend_org.detach().numpy()])
                                    val_print_count = [[cat, value[0]] for cat, value in
                                                       zip(val_cat.keys(), val_cat.values())]
                                    print("\n", val_print_count)

                    if val_count == 21:
                        self.draw_val_img(val_result_draw, "val21.png")
                        val_count += 1
                        print("saved!!")

                    if self.save_demo and data["patch_w"][i] > 200:
                        if iou_i > 0.7 and iou_rend > iou_i + 0.06:
                            demo_name = "demo_results/img_" + str(len(evaluate_df["cat"]))
                            img_demo = img_x[i].permute(1, 2, 0).detach().numpy()
                            plt.imshow(img_demo)
                            plt.axis('off')
                            plt.savefig(demo_name+"_0.png", format="png", dpi=175, bbox_inches='tight')
                            plt.close()
                            plt.plot(curr_pred_poly[:, 0], curr_pred_poly[:, 1], marker=".", c="b")
                            plt.imshow(img_demo)
                            plt.axis('off')
                            plt.savefig(demo_name+"_1.png", format="png", dpi=175, bbox_inches='tight')
                            plt.close()
                            plt.plot(curr_gt_poly[:, 0], curr_gt_poly[:, 1], marker=".", c="r")
                            plt.imshow(img_demo)
                            plt.axis('off')
                            plt.savefig(demo_name + "_2.png", format="png", dpi=175, bbox_inches='tight')
                            plt.close()
                            img_pred = apply_mask(img_demo.copy(), mask_rend,
                                                  color=(0.0, 0.0, 1.0), alpha=0.2)
                            plt.imshow(img_pred)
                            plt.axis('off')
                            plt.savefig(demo_name+"_3.png", format="png", dpi=175, bbox_inches='tight')
                            plt.close()
                            img_gt = apply_mask(img_demo.copy(), gt_mask[i].astype(np.uint8),
                                                color=(1.0, 0.0, 0.0), alpha=0.2)
                            plt.imshow(img_gt)
                            plt.axis('off')
                            plt.savefig(demo_name + "_4.png", format="png", dpi=175, bbox_inches='tight')
                            plt.close()
                            plt.imshow(mask_rend_org.detach().numpy())
                            plt.axis('off')
                            plt.savefig(demo_name+"_5.png", format="png", dpi=175, bbox_inches='tight')
                            plt.close()
                            print("\n|-------------------------|")
                            print("---------saved!!---------")
                            print("|-------------------------|")

        if val_count != 22:
            self.draw_val_img(val_result_draw, "val21_e.png")
            print("saved but incomplete!!")

        return evaluate_df

    def draw_val_img(self, val_result_draw, img_name):
        fig = plt.figure(figsize=(25, 22))
        for i in range(len(val_result_draw)):
            ax2 = fig.add_subplot(6, 7, i // 7 * 14 + i % 7 + 1)
            title2 = "iou: %.2f, rend: %.2f" % (100 * val_result_draw[i][4], 100 * val_result_draw[i][5])
            ax2.set_title(title2, fontsize=15)
            plt.axis('off')
            plt.plot(val_result_draw[i][1][:, 0], val_result_draw[i][1][:, 1], marker=".", c="b")
            img = apply_mask(val_result_draw[i][0], val_result_draw[i][3], color=(0.0, 0.0, 1.0), alpha=0.2)
            img = apply_mask(img, val_result_draw[i][2], color=(1.0, 0.0, 0.0), alpha=0.2)
            plt.imshow(img)

            ax = fig.add_subplot(6, 7, i // 7 * 14 + i % 7 + 8)
            title = "rend raw"
            ax.set_title(title, fontsize=15)
            plt.axis('off')
            plt.imshow(val_result_draw[i][6])

        plt.subplots_adjust(hspace=0.093)
        plt.savefig(img_name, format="png", dpi=175, bbox_inches='tight')
        plt.close()

    def dump_data(self, json_data, data_time):
        with open(self.data_name + "_process.json", "a+") as f:
            json.dump(json_data, f)
            f.write("\n")
        info = "\ntime: %s, epoch: %d, batch: %07d, loss: %.5f, L_rend: %.2f, iou: %.2f" % (
            data_time, json_data["epoch"], json_data["batch"], json_data["loss"], json_data["loss_rend"], json_data["iou"])
        print(info)


def make_trainer_extend(model, cfg):
    return TrainerExtend(model, cfg)
