from dataset import get_data_loaders_extend
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.config import make_config
from utils.contour_mask_utils import apply_mask


if __name__ == "__main__":
    cfg = make_config("configs/config.json")
    cfg.p_num = 60
    train, val = get_data_loaders_extend(cfg.data_root, cfg.p_num)
    count = 0
    for data in tqdm(val):
        img_x = data["img"]
        mask = data["gt_mask"]
        poly = data["gt_poly"]
        poly_raw = data["orig_poly"]
        bs = img_x.shape[0]
        for bsi in range(bs):
            img_show = img_x[bsi].permute(1, 2, 0).detach().numpy()
            mask_bsi = mask[bsi][:, :].detach().numpy()
            apply_mask(img_show, mask_bsi, (1, 0, 0), 0.3)
            plt.imshow(img_show)
            pi_show = poly[bsi].detach().numpy() * 223
            plt.plot(pi_show[:, 0], pi_show[:, 1], marker=".", c="b")
            plt.axis('off')
            plt.show()

    print(1)

