# 将 val数据集换为test，取消val(如ChangeFormer)

import argparse
import os
from collections import OrderedDict
from PIL import Image
from utils.camutils_CD import cam_to_label, multi_scale_cam
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
from datasets import weaklyCD
from utils import evaluate_CD
from models.model_transwcd_fastvit_metric import TransWCD_single, TransWCD_dual, reparameterize_model
from torchmetrics import Accuracy, Precision, Recall, F1Score, StatScores
import pandas as pd
import time
from pytorch_metric_learning import distances

parser = argparse.ArgumentParser()
# LEVIR/DSIFN/WHU.yaml
parser.add_argument("--config",default='configs/bottle_dual_mitb1_ce.yaml',type=str,
                    help="config")
parser.add_argument("--save_dir", default="./results/bottle_dual_mitb1_ce", type=str, help="save_dir")
parser.add_argument("--eval_set", default="val", type=str, help="eval_set")
parser.add_argument("--model_path", default="/home/jovyan/VTE_CD/TransWCD/transwcd/work_dir_bottle_dual_mitb1_ce/checkpoints/2025-01-14-08-15/transwcd_f1_iter_500.pth", type=str, help="model_path")

parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--bkg_score", default=0.45, type=float, help="bkg_score")
parser.add_argument("--resize_long", default=256, type=int, help="resize the long side (256 or 512)")

parser.add_argument(
    "--use_inference_mode",
    dest="use_inference_mode",
    action="store_true",
    default=False,
    help="use inference mode version of model definition.",
)

def test(model, dataset, accuracy_metric, precision_metric, recall_metric, f1_metric, stats_metric, test_scales=1.0, distance=None):
    gts, cams = [], []

    img_gt_pred = []

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.test.batch_size, shuffle=False, num_workers=2,
                                              pin_memory=False)

    with torch.no_grad(), torch.cuda.device(0):
        model.cuda(0)
        # Measure latency for all inputs
        latencies = []
        for idx, data in tqdm(enumerate(data_loader)):
            ### 注意此处cls_label ###
            name, inputs_A, inputs_B, labels, cls_label = data

            inputs_A = inputs_A.cuda()
            inputs_B = inputs_B.cuda()

            b, c, h, w = inputs_A.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda()


            _, _, h, w = inputs_A.shape
            ratio = args.resize_long / max(h, w)
            _h, _w = int(h * ratio), int(w * ratio)
            inputs_A = F.interpolate(inputs_A, size=(_h, _w), mode='bilinear', align_corners=False)

            _, _, h, w = inputs_B.shape
            ratio = args.resize_long / max(h, w)
            _h, _w = int(h * ratio), int(w * ratio)
            inputs_B = F.interpolate(inputs_B, size=(_h, _w), mode='bilinear', align_corners=False)
            
            if cfg.loss == "cross_entropy":
                tic = time.process_time()

                cls = model(inputs_A, inputs_B)

                latencies.append(time.process_time() - tic)
                # Convert logits to predictions
                probs = torch.sigmoid(cls)
                preds = (probs >= 0.5).float()
            elif cfg.loss == "contrastive" or cfg.loss == "triplet":
                tic = time.process_time()

                cls_1, cls_2 = model(inputs_A, inputs_B)

                latencies.append(time.process_time() - tic)

                pairwise_distances  = distance(query_emb=cls_1, ref_emb=cls_2)  # Shape: [B,B]

                # Extract diagonal values
                diagonal_distances = torch.diag(pairwise_distances).unsqueeze(1) # Shape: [B, 1]

                # Apply threshold
                threshold = 1.0
                preds = (diagonal_distances > threshold).float()  # Shape: [B, 1]

            # Update each metric with the current batch predictions and labels
            accuracy_metric.update(preds, cls_label)
            precision_metric.update(preds, cls_label)
            recall_metric.update(preds, cls_label)
            f1_metric.update(preds, cls_label)
            stats_metric.update(preds, cls_label)  # Update stats metric

            for img_name, gt_label, pred in zip(name, cls_label, preds):
                img_gt_pred.append({"image_name": img_name, "cls_label": int(gt_label.item()), "pred_label": int(pred.item())})

            _cams = multi_scale_cam(model, inputs_A, inputs_B, cfg.cam.scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_to_label(resized_cam, cls_label, cfg=cfg)

            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

            # 以png格式保存cam结果
            cam_path = args.save_dir + '/prediction/' + name[0]
            cam_img = Image.fromarray((cam_label.squeeze().cpu().numpy() * 255).astype(np.uint8))
            cam_img.save(cam_path)

            ### FN and FP color ###
            cam = cam_label.squeeze().cpu().numpy()
            labels = labels.squeeze().cpu().numpy()

            # Create RGB image from labels
            label_rgb = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
            label_rgb[labels == 0] = [0, 0, 0]  # Background (black)
            label_rgb[labels == 1] = [255, 255, 255]  # Foreground (white)

            # Mark FN pixels as blue
            fn_pixels = np.logical_and(cam == 0, labels == 1)  # False Negatives
            label_rgb[fn_pixels] = [0, 0, 255]  # Blue

            # Mark FP pixels as red
            fp_pixels = np.logical_and(cam == 1, labels == 0)  # False Positives
            label_rgb[fp_pixels] = [255, 0, 0]  # Red

            # Save the labeled image
            label_with_fn_fp_path = args.save_dir + '/prediction_color/' + name[0]
            label_with_fn_fp_img = Image.fromarray(label_rgb)
            label_with_fn_fp_img.save(label_with_fn_fp_path)

        print(f"Average GPU Inference Latency per Input: {np.mean(latencies):.3f} s")
        print(sum(latencies))
        return inputs_A, inputs_B, gts, cams, img_gt_pred


def main(cfg):
    test_dataset = weaklyCD.CDDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=args.eval_set,
        stage='test',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    if cfg.scheme == "transwcd_dual":
        transwcd = TransWCD_dual(backbone=cfg.backbone.config,
                                 stride=cfg.backbone.stride,
                                 num_classes=cfg.dataset.num_classes,
                                 embedding_dim=cfg.backbone.embedding_dim,
                                 pretrained=True,
                                 pooling=args.pooling, 
                                 backbone_arg=cfg.backbone,
                                 loss=cfg.loss,
                                 )
    elif cfg.scheme == "transwcd_single":
        transwcd = TransWCD_single(backbone=cfg.backbone.config,
                                   stride=cfg.backbone.stride,
                                   num_classes=cfg.dataset.num_classes,
                                   embedding_dim=cfg.backbone.embedding_dim,
                                   pretrained=True,
                                   pooling=args.pooling, 
                                   backbone_arg=cfg.backbone,
                                   loss=cfg.loss,
                                   )
    else:
        print('Please fill in cfg.scheme!')

    trained_state_dict = torch.load(args.model_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        if 'diff.0.bias' in k:
            k = k.replace('diff.0.bias', 'diff.bias')
        if 'diff.0.weight' in k:
            k = k.replace('diff.0.weight', 'diff.weight')
        new_state_dict[k] = v

    transwcd.load_state_dict(state_dict=new_state_dict, strict=True)  # True
    transwcd.eval()

    if "fastvit" in cfg.backbone.config:
        if not args.use_inference_mode:
            print("Reparameterizing Model %s" % (cfg.backbone.config))
            transwcd = reparameterize_model(transwcd)
            transwcd.eval()
            torch.save(transwcd.state_dict(), args.model_path.split(".")[0] + "_reparam.pth")

    # Initialize metric objects
    accuracy_metric = Accuracy(task="binary").to('cuda')
    precision_metric = Precision(task="binary", zero_division=1).to('cuda')
    recall_metric = Recall(task="binary", zero_division=1).to('cuda')
    f1_metric = F1Score(task="binary", zero_division=1).to('cuda')
    stats_metric = StatScores(task="binary").to('cuda')  # To get TP, TN, FP, FN

    if cfg.loss == "contrastive" or cfg.loss == "triplet":
        lp_distance = distances.LpDistance(p=2)

    ###   test 输出 ###
    if cfg.loss == "cross_entropy":
        inputs_A, inputs_B, gts, cams, img_gt_pred = test(
            model=transwcd, 
            dataset=test_dataset, 
            accuracy_metric=accuracy_metric, 
            precision_metric=precision_metric, 
            recall_metric=recall_metric, 
            f1_metric=f1_metric, 
            stats_metric=stats_metric, 
            test_scales=[1.0])
    
    elif cfg.loss == "contrastive" or cfg.loss == "triplet":
        inputs_A, inputs_B, gts, cams, img_gt_pred = test(
            model=transwcd, 
            dataset=test_dataset, 
            accuracy_metric=accuracy_metric, 
            precision_metric=precision_metric, 
            recall_metric=recall_metric, 
            f1_metric=f1_metric, 
            stats_metric=stats_metric, 
            test_scales=[1.0],
            distance=lp_distance)

    # Compute final metrics after the loop
    accuracy = accuracy_metric.compute()
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    f1 = f1_metric.compute()
    # Get TP, TN, FP, FN from the stats metric
    tp, fp, tn, fn, _ = stats_metric.compute()

    torch.cuda.empty_cache()

    # Convert to DataFrame
    df = pd.DataFrame(img_gt_pred)

    # Save to CSV
    df.to_csv(args.save_dir + "/output.csv", index=False)

    cams_score = evaluate_CD.scores(gts, cams)
    cls_score = {"accuracy": float(accuracy.item()), "precision": float(precision.item()), "recall": float(recall.item()), "f1": float(f1.item()), "tp": int(tp.item()), "fp": int(fp.item()), "tn": int(tn.item()), "fn": int(fn.item())}

    # Print or log metrics
    print("cls_score:")
    print(cls_score)

    print("cams score:")
    print(cams_score)

    return True


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    cfg.cam.bkg_score = args.bkg_score
    print(cfg)
    print(args)

    args.save_dir = os.path.join(args.save_dir, args.eval_set)

    os.makedirs(args.save_dir + "/prediction", exist_ok=True)
    os.makedirs(args.save_dir + "/prediction_color", exist_ok=True)

    main(cfg=cfg)




