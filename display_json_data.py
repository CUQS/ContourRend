from torch.utils.tensorboard import SummaryWriter
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--process', type=str, default="ckpt_process.json",
                    help='string(default: "ckpt_process.json"): training process file')
opt = parser.parse_args()

if __name__ == "__main__":
    f = open(opt.process, 'r')

    summary = []
    for l_i in f.readlines():
        summary.append(json.loads(l_i))
    f.close()

    writer = SummaryWriter("runs/val")
    for idx, s_i in enumerate(summary):
        loss = s_i['train_loss']
        iou = s_i['train_iou']
        epoch = s_i['epoch']
        batch = s_i['batch']
        if epoch > 0:
            break
        writer.add_scalar('loss', loss, batch)
        writer.flush()
        writer.add_scalar('iou', iou, batch)
        writer.flush()

    print("finish!!\ncheck with tensorboard!!")
    print("tensorboard --logdir=runs")
