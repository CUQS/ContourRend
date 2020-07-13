from utils.config import make_config
from utils.train import make_trainer_extend
from utils.model import DeepLabGCNModel
from dataset import get_val_loaders_extend
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, default="0,1,2,3",
                    help='string(default: "0,1,2,3"): device')
parser.add_argument('--ckpt', type=str, default="weight/latest.pth",
                    help='string(default: "weight/latest.pth"): weight file')
opt = parser.parse_args()

if __name__ == "__main__":
    cfg = make_config("configs/config.json")
    cfg.model_type = "test"
    cfg.gpus = opt.gpus
    cfg.ckpt = opt.ckpt
    val_loader = get_val_loaders_extend(cfg.data_root)
    print("build model...")
    model = DeepLabGCNModel(cfg)
    model.load_weight(cfg.ckpt)
    print("build success!!")
    trainer = make_trainer_extend(model, cfg)
    print("test...")

    evaluate_df = trainer.val(val_loader)

    df = pd.DataFrame(evaluate_df)
    df.to_csv("evaluate_result.csv")

    print("-------------------------")
    print("mean")
    print("-------------------------")
    class_iou = []
    class_rend = []
    for u_i in df["cat"].unique():
        class_iou.append(df[df["cat"] == u_i]["iou"].mean())
        class_rend.append(df[df["cat"] == u_i]["iou_rend"].mean())
    print("contour mean iou: ", 100 * sum(class_iou) / len(class_iou))
    print("rend mean iou: ", 100 * sum(class_rend) / len(class_rend))

    for u_i in df["cat"].unique():
        print("-------------------------")
        print(u_i)
        print("-------------------------")
        df_t = df[df["cat"] == u_i]
        print(df_t.describe())