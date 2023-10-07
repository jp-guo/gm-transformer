import torch
import numpy as np
from pathlib import Path
import random

from data.data_loader import GMDataset, get_dataloader
from src.utils.model_sl import load_model
from parallel import DataParallel
from utils.hungarian import hungarian
import matplotlib
try:
    import _tkinter
except ImportError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"

from utils.config import cfg


def visualize_model(models, dataloader, device, num_images=6, set='train', cls=None, save_img=False):
    print('Visualizing model...')
    assert set in ('train', 'test')

    for model in models:
        was_training = model.training
        model.eval()
    images_so_far = 0

    names = ['source', 'GMN', 'PCA-GM', 'IPCA-GM']
    num_cols = num_images + 1

    old_cls = dataloader[set].dataset.cls
    if cls is not None:
        dataloader[set].dataset.cls = cls

    lap_solver = hungarian

    visualize_path = Path(cfg.OUTPUT_PATH) / 'visual'
    if save_img:
        if not visualize_path.exists():
            visualize_path.mkdir(parents=True)
    with torch.no_grad():
        fig = plt.figure(figsize=(50, 20), dpi=120)
        for idx in range(len(names)):
            ax = plt.subplot(len(names), num_cols, idx * num_cols + 1)
            ax.axis('off')
            plt.text(0.5, 0.5, names[idx], fontsize=40, horizontalalignment='center')

        for i, inputs in enumerate(dataloader[set]):
            if 'images' in inputs:
                data1, data2 = [_.cuda() for _ in inputs['images']]
                inp_type = 'img'
            elif 'features' in inputs:
                data1, data2 = [_.cuda() for _ in inputs['features']]
                inp_type = 'feat'
            else:
                raise ValueError('no valid data key (\'images\' or \'features\') found from dataloader!')
            P1_gt, P2_gt = [_.cuda() for _ in inputs['Ps']]
            n1_gt, n2_gt = [_.cuda() for _ in inputs['ns']]
            e1_gt, e2_gt = [_.cuda() for _ in inputs['es']]
            G1_gt, G2_gt = [_.cuda() for _ in inputs['Gs']]
            H1_gt, H2_gt = [_.cuda() for _ in inputs['Hs']]
            KG, KH = [_.cuda() for _ in inputs['Ks']]
            perm_mat = inputs['gt_perm_mat'].cuda()

            s_pred_perms = []
            for model in models:
                s_pred, _ = model(data1, data2, P1_gt, P2_gt, G1_gt, G2_gt, H1_gt, H2_gt, n1_gt, n2_gt, KG, KH, inp_type)
                if type(s_pred) is list:
                    s_pred = s_pred[-1]
                s_pred_perm = lap_solver(s_pred, n1_gt, n2_gt)
                s_pred_perms.append(s_pred_perm)

            for j in range(s_pred.size()[0]):
                if n1_gt[j] <= 5:
                    continue

                matched = []
                for idx, s_pred_perm in enumerate(s_pred_perms):
                    matched_num = torch.sum(s_pred_perm[j, :n1_gt[j], :n2_gt[j]] * perm_mat[j, :n1_gt[j], :n2_gt[j]])
                    matched.append(matched_num)

                if random.choice([0, 1]) == 1:
                    if not ((matched[0] < matched[1] or matched[0] < matched[2]) and (matched[0] <= matched[1] and matched[0] <= matched[2])):
                        continue
                else:
                    if not (matched[0] <= matched[1] and matched[0] <= matched[2]):
                        continue

                images_so_far += 1
                print(chr(13) + 'Visualizing {:4}/{:4}'.format(images_so_far, num_images))  # chr(13)=CR

                colorset = np.random.rand(n1_gt[j], 3)
                ax = plt.subplot(1 + len(s_pred_perms), num_cols, images_so_far + 1)
                ax.axis('off')
                #plt.title('source')
                plot_helper(data1[j], P1_gt[j], n1_gt[j], ax, colorset)

                for idx, s_pred_perm in enumerate(s_pred_perms):
                    ax = plt.subplot(1 + len(s_pred_perms), num_cols, (idx + 1) * num_cols + images_so_far + 1)
                    ax.axis('off')
                    #plt.title('predict')
                    plot_helper(data2[j], P2_gt[j], n1_gt[j], ax, colorset, 'tgt', s_pred_perm[j], perm_mat[j])

                #ax = plt.subplot(2 + len(s_pred_perms), num_images + 1, (len(s_pred_perms) + 1) * num_images + images_so_far)
                #ax.axis('off')
                #plt.title('groundtruth')
                #plot_helper(data2[j], P2_gt[j], n1_gt[j], ax, colorset, 'tgt', perm_mat[j])

                if not save_img:
                    plt.show()
                    print("Press Enter to continue...", end='', flush=True)  # prevent new line
                    input()

                if images_so_far == num_images:
                    fig.savefig(str(visualize_path / '{:0>4}.pdf'.format(images_so_far)), bbox_inches='tight')

                    model.train(mode=was_training)
                    dataloader[set].dataset.cls = old_cls
                    return

    dataloader[set].dataset.cls = old_cls


def plot_helper(img, P, n, ax, colorset, mode='src', pmat=None, gt_pmat=None):
    assert mode in ('src', 'tgt')
    if mode == 'tgt':
        assert pmat is not None
    img = tensor2np(img.cpu())
    plt.imshow(img)

    P = P.cpu().numpy()
    if mode == 'src':
        for i in range(n):
            mark = plt.Circle(P[i], 7, edgecolor='w', facecolor=colorset[i])
            ax.add_artist(mark)
    else:
        pmat = pmat.cpu().numpy()
        gt_pmat = gt_pmat.cpu().numpy()
        idx = np.argmax(pmat, axis=-1)
        idx_gt = np.argmax(gt_pmat, axis=-1)
        matched = 0
        for i in range(n):
            mark = plt.Circle(P[idx[i]], 7, edgecolor='w' if idx[i] == idx_gt[i] else 'r', facecolor=colorset[i])
            ax.add_artist(mark)
            if idx[i] == idx_gt[i]:
                matched += 1
        plt.title('{:d}/{:d}'.format(matched, n), y=-0.2, fontsize=25)


def tensor2np(inp):
    """Tensor to numpy array for plotting"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(cfg.NORM_MEANS)
    std = np.array(cfg.NORM_STD)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


if __name__ == '__main__':
    from utils.parse_args import parse_args
    args = parse_args('Deep learning of graph matching visualization code.')

    import importlib
    from utils.config import cfg_from_file

    dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, 'test': cfg.EVAL.SAMPLES}
    image_dataset = {
        x: GMDataset('WillowObject',
                     sets=x,
                     length=dataset_len[x],
                     pad=cfg.PAIR.PADDING,
                     obj_resize=cfg.PAIR.RESCALE)
        for x in ('train', 'test')}
    dataloader = {x: get_dataloader(image_dataset[x], fix_seed=(x == 'test'))
        for x in ('train', 'test')}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_paths = ['/home/wangrunzhong/dl-of-gm/output/vgg16_sm_ol_voc/params/params_0009.pt.6044',
                   '/home/wangrunzhong/dl-of-gm/output/vgg16_pcatest_voc/params/params_0005.pt.6456',
                   '/home/wangrunzhong/dl-of-gm/output/vgg16_pca_iter_voc/params/params_0010.pt.6619']
    models = []

    for i, model_path in enumerate(model_paths):
        if i == 0:
            cfg_from_file('experiments/vgg16_sm_voc.yaml')
        elif i == 1:
            cfg_from_file('experiments/vgg16_pca_voc.yaml')
        else:
            cfg_from_file('experiments/vgg16_pca_iter_voc.yaml')

        mod = importlib.import_module(cfg.MODULE)
        Net = mod.Net

        model = Net()
        model = model.to(device)
        model = DataParallel(model, device_ids=cfg.GPUS)

        #model_path = None
        #if cfg.VISUAL.EPOCH != 0:
        #    model_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'params_{:04}.pt'.format(cfg.VISUAL.EPOCH))
        #elif len(cfg.VISUAL.WEIGHT_PATH) != 0:
        #    model_path = cfg.VISUAL.WEIGHT_PATH
        #if model_path is not None:
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)
        models.append(model)

    visualize_model(models, dataloader, device,
                    num_images=cfg.VISUAL.NUM_IMGS,
                    cls=cfg.VISUAL.CLASS if cfg.VISUAL.CLASS != 'none' else None,
                    save_img=cfg.VISUAL.SAVE)
