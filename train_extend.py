from utils.config import make_config
from utils.train import make_trainer_extend
from utils.model import DeepLabGCNModel
from dataset import get_data_loaders_extend
import json
import pandas as pd


if __name__ == "__main__":
    cfg = make_config("configs/config.json")
    train_loader, val_loader = get_data_loaders_extend(cfg.data_root, cfg.p_num, cfg.cp_num)
    print("build model...")
    model = DeepLabGCNModel(cfg)
    if cfg.resume:
        model.load_weight(cfg.ckpt)
    print("build success!!")
    trainer = make_trainer_extend(model, cfg)

    for epoch_i in range(cfg.epoch):
        print("train...")
        trainer.train(train_loader, epoch_i)
        print("val...")
        evaluate_df = trainer.val(val_loader)
        length = len(evaluate_df["iou"])
        val_data = {"val_iou": [], "epoch": [], "iou_rend": []}

        df = pd.DataFrame(evaluate_df)
        class_iou = []
        class_rend = []
        for u_i in df["cat"].unique():
            class_iou.append(df[df["cat"] == u_i]["iou"].mean())
            class_rend.append(df[df["cat"] == u_i]["iou_rend"].mean())

        val_data["val_iou"].append(sum(class_iou)/len(class_iou))
        val_data["iou_rend"].append(sum(class_rend) / len(class_rend))
        val_data["epoch"].append(epoch_i)
        with open(cfg.data_name + "_train_val.json", "a+") as f:
            json.dump(val_data, f)
            f.write("\n")
