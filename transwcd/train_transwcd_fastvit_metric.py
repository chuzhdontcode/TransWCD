import argparse
import datetime
import logging
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import weaklyCD
from utils import evaluate_CD, imutils
from utils.AverageMeter import AverageMeter
from utils.camutils_CD import cam_to_label,multi_scale_cam
from utils.optimizer import PolyWarmupAdamW
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from models.model_transwcd_fastvit_metric import TransWCD_dual, TransWCD_single

from pathlib import Path
from torchmetrics import Accuracy, Precision, Recall, F1Score, StatScores
from pytorch_metric_learning import losses, distances
import wandb

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
# LEVIR/DSIFN/WHU.yaml
parser.add_argument("--project", default="transwcd_fastvit_metric", type=str, help="wandb project name")
parser.add_argument("--config",
                    default='configs/bottle_dual_mitb1_ce.yaml',
                    type=str,
                    help="config")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--crop_size", default=256, type=int, help="crop_size")
parser.add_argument("--scheme", default='transwcd_dual', type=str, help="transwcd_dual or transwcd_single")
parser.add_argument('--pretrained', default= True, type=bool, help="pretrained")
parser.add_argument('--checkpoint_path', default= False, type=str, help="checkpoint_path" )
parser.add_argument(
    "--wandb",
    dest="wandb",
    action="store_true",
    default=False,
    help="log in wandb",
)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(filename='test.log'):
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)

def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)

    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

def get_down_size(ori_shape=(512, 512), stride=16):
    h, w = ori_shape
    _h = h // stride + 1 - ((h % stride) == 0)
    _w = w // stride + 1 - ((w % stride) == 0)
    return _h, _w

def validate(accuracy_metric, precision_metric, recall_metric, f1_metric, stats_metric, model=None, data_loader=None, cfg=None, distance=None):
    preds, gts, cams = [], [], []
    model.eval()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs_A, inputs_B, labels, cls_label = data

            inputs_A = inputs_A.cuda()
            inputs_B = inputs_B.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            if cfg.loss == "cross_entropy":
                cls = model(inputs_A, inputs_B)
                # Convert logits to predictions
                probs = torch.sigmoid(cls)
                preds = (probs >= 0.5).float()
            elif cfg.loss == "contrastive" or cfg.loss == "triplet":
                cls_1, cls_2 = model(inputs_A, inputs_B)

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

            _cams = multi_scale_cam(model, inputs_A, inputs_B, cfg.cam.scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_to_label(resized_cam, cls_label, cfg=cfg)

            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

    # Compute final metrics after the loop
    accuracy = accuracy_metric.compute()
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    f1 = f1_metric.compute()
    # Get TP, TN, FP, FN from the stats metric
    tp, fp, tn, fn, _ = stats_metric.compute()
    cam_score = evaluate_CD.scores(gts, cams)

    # Reset metrics after each validation phase
    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    stats_metric.reset()

    model.train()
    return cam_score, labels, accuracy, precision, recall, f1, tp, fp, tn, fn

def train(cfg):
    num_workers = 10

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset = weaklyCD.ClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        num_classes=cfg.dataset.num_classes,
    )

    val_dataset = weaklyCD.CDDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='val',
        aug=False,
        num_classes=cfg.dataset.num_classes,
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.batch_size,
                              # shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True,
                              prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.val.batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False)
    device = torch.device('cuda')

    if cfg.scheme == "transwcd_dual":
        transwcd = TransWCD_dual(backbone=cfg.backbone.config,
                                 stride=cfg.backbone.stride,
                                 num_classes=cfg.dataset.num_classes,
                                 embedding_dim=cfg.backbone.embedding_dim,
                                 pretrained=args.pretrained,
                                 pooling=args.pooling, 
                                 backbone_arg=cfg.backbone,
                                 loss=cfg.loss,
                                 )
    elif cfg.scheme == "transwcd_single":
        transwcd = TransWCD_single(backbone=cfg.backbone.config,
                                 stride=cfg.backbone.stride,
                                 num_classes=cfg.dataset.num_classes,
                                 embedding_dim=cfg.backbone.embedding_dim,
                                 pretrained=args.pretrained,
                                 pooling=args.pooling, 
                                 backbone_arg=cfg.backbone,
                                 loss=cfg.loss,
                                 )
    else:
        print("Please choose a baseline structure in /configs/...yaml")

    #logging.info('\nNetwork config: \n%s' % (transwcd))
    if cfg.hyperparam == "transwcd":
        param_groups = transwcd.get_param_groups()
    transwcd.to(device)

    writer = SummaryWriter(cfg.work_dir.logger_dir)
    print('writer:',writer)
    iter_per_epoch = int(np.ceil(len(train_dataset)/cfg.train.batch_size))
    if cfg.hyperparam == "transwcd":
        optimizer = PolyWarmupAdamW(
            params=[
                {
                    "params": param_groups[0],
                    "lr": cfg.optimizer.learning_rate,
                    "weight_decay": cfg.optimizer.weight_decay,
                },
                {
                    "params": param_groups[1],
                    "lr": 0.0,  ## freeze norm layers
                    "weight_decay": 0.0,
                },
                {
                    "params": param_groups[2],
                    "lr": cfg.optimizer.learning_rate * 10,
                    "weight_decay": cfg.optimizer.weight_decay,
                },
            ],
            lr=cfg.optimizer.learning_rate,
            weight_decay=cfg.optimizer.weight_decay,
            betas=cfg.optimizer.betas,
            warmup_iter=cfg.scheduler.warmup_iter,
            max_iter=cfg.train.max_iters,
            warmup_ratio=cfg.scheduler.warmup_ratio,
            power=cfg.scheduler.power
        )
    elif cfg.hyperparam == "fastvit":
        assert int(cfg.train.max_iters/iter_per_epoch) == cfg.epochs

        optimizer = create_optimizer_v2(transwcd, **optimizer_kwargs(cfg=cfg))
        lr_scheduler, num_epochs = create_scheduler(cfg, optimizer)

        logging.info("Scheduled epochs: {}".format(num_epochs))

        cfg.train.max_iters = num_epochs * iter_per_epoch

        logging.info("Scheduled max iters: {}".format(cfg.train.max_iters))

    train_loader_iter = iter(train_loader)

    if cfg.loss == "contrastive":
        # Initialize ContrastiveLoss with LpDistance
        lp_distance = distances.LpDistance(p=2)
        loss_fn = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    elif cfg.loss == "triplet":
        # Initialize TripletLoss with LpDistance
        lp_distance = distances.LpDistance(p=2)
        loss_fn = losses.TripletMarginLoss()

    avg_meter = AverageMeter()

    # Initialize metric objects
    accuracy_metric = Accuracy(task="binary").to('cuda')
    precision_metric = Precision(task="binary", zero_division=1).to('cuda')
    recall_metric = Recall(task="binary", zero_division=1).to('cuda')
    f1_metric = F1Score(task="binary", zero_division=1).to('cuda')
    stats_metric = StatScores(task="binary").to(device)  # To get TP, TN, FP, FN

    bkg_cls = torch.ones(size=(cfg.train.batch_size, 1))

    best_F1 = 0.0
    best_iou = 0.0
    best_F1_iter = 0
    best_iou_iter = 0
    epoch = 0
    batch_idx = 0
    for n_iter in range(cfg.train.max_iters):

        try:
            img_name, inputs_A, inputs_B, cls_labels, img_box = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            img_name, inputs_A, inputs_A, cls_labels, img_box = next(train_loader_iter)

        inputs_A = inputs_A.to(device, non_blocking=True)
        inputs_B = inputs_B.to(device, non_blocking=True)

        cls_labels = cls_labels.to(device, non_blocking=True)

        if cfg.loss == "cross_entropy":
            cls = transwcd(inputs_A, inputs_B)
        elif cfg.loss == "contrastive" or cfg.loss == "triplet":
            cls_1, cls_2 = transwcd(inputs_A, inputs_B) # Shape: [8, 512]

        cams = multi_scale_cam(transwcd, inputs_A=inputs_A, inputs_B=inputs_B, scales=cfg.cam.scales)

        valid_cam, pred_cam = cam_to_label(cams.detach(), cls_label=cls_labels, img_box=img_box, ignore_mid=True,
                                           cfg=cfg)

        bkg_cls = bkg_cls.to(cams.device)
        _cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

        # change classification loss
        if cfg.loss == "cross_entropy":
            cc_loss = F.binary_cross_entropy_with_logits(cls, cls_labels)

        elif cfg.loss == "contrastive" or cfg.loss == "triplet":
            cls_labels_loss = cls_labels.squeeze(1)  # Adjust label shape [8, 1] -> [8]
            cls_labels_ref = torch.zeros_like(cls_labels_loss) # create cls labels 0 for ref images (good)

            # Compute the loss
            cc_loss = loss_fn(embeddings=cls_1, labels=cls_labels_ref, ref_emb=cls_2, ref_labels=cls_labels_loss)

        if n_iter <= cfg.train.cam_iters:
            loss = 1.0 * cc_loss
        else:
            loss = 1.0 * cc_loss

        avg_meter.add({'cc_loss': cc_loss.item()})

        optimizer.zero_grad()
        loss.backward()
        
        # Logging Gradient Norms
        total_norm = 0
        for name, param in transwcd.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.norm(2)  # L2 norm
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        batch_idx = batch_idx + 1
        if batch_idx == iter_per_epoch:
            batch_idx = 0
            epoch = epoch + 1

        optimizer.step()

        if cfg.hyperparam == "fastvit":
            if lr_scheduler is not None:
                lr_scheduler.step_update(num_updates=n_iter+1, metric=avg_meter.get('cc_loss'))

            if (n_iter + 1) % iter_per_epoch == 0:
                if lr_scheduler is not None:
                    # step LR for next epoch
                    lr_scheduler.step(epoch, "top1")

        if (n_iter + 1) % cfg.train.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            pred_cam = pred_cam.cpu().numpy().astype(np.int16)

            train_cc_loss = avg_meter.pop('cc_loss')

            logging.info(
                "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cc_loss: %.4f" % (
                    n_iter + 1, delta, eta, cur_lr, train_cc_loss,))

            grid_imgs_A, grid_cam_A = imutils.tensorboard_image(imgs=inputs_A.clone(), cam=valid_cam)
            grid_imgs_B, grid_cam_B = imutils.tensorboard_image(imgs=inputs_B.clone(), cam=valid_cam)

            grid_pred_cam = imutils.tensorboard_label(labels=pred_cam)

            writer.add_image("train/images_A"+str(img_name), grid_imgs_A, global_step=n_iter)
            writer.add_image("train/images_B"+str(img_name), grid_imgs_B, global_step=n_iter)
            writer.add_image("cam/valid_cams_A", grid_cam_A, global_step=n_iter)
            writer.add_image("cam/valid_cams_B", grid_cam_B, global_step=n_iter)
            writer.add_image("train/preds_cam", grid_pred_cam, global_step=n_iter)

            writer.add_scalars('train/loss', {"cc_loss": cc_loss.item()},
                               global_step=n_iter)

        if (n_iter + 1) % cfg.train.eval_iters == 0:
            logging.info('CD Validating...')
            if cfg.loss == "cross_entropy":
                cam_score, labels, accuracy, precision, recall, f1, tp, fp, tn, fn = validate(
                    accuracy_metric, 
                    precision_metric, 
                    recall_metric, 
                    f1_metric, 
                    stats_metric, 
                    model=transwcd, 
                    data_loader=val_loader, 
                    cfg=cfg
                )  # _ 为 labels
            elif cfg.loss == "contrastive" or cfg.loss == "triplet":
                cam_score, labels, accuracy, precision, recall, f1, tp, fp, tn, fn = validate(
                    accuracy_metric, 
                    precision_metric, 
                    recall_metric, 
                    f1_metric, 
                    stats_metric, 
                    model=transwcd, 
                    data_loader=val_loader, 
                    cfg=cfg, 
                    distance=lp_distance,
                )
            cls_score = {"accuracy": float(accuracy.item()), "precision": float(precision.item()), "recall": float(recall.item()), "f1": float(f1.item()), "tp": int(tp.item()), "fp": int(fp.item()), "tn": int(tn.item()), "fn": int(fn.item())}
            
            if cls_score['f1'] > best_F1:
                best_F1 = cls_score['f1']
                best_F1_iter = n_iter + 1
                
                pth_files = Path(cfg.work_dir.ckpt_dir).glob('*.pth')
                for pth_file in pth_files:
                    if "transwcd_f1_iter_" in str(pth_file):
                        pth_file.unlink()
                
                ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "transwcd_f1_iter_%d.pth" % (n_iter + 1))
                torch.save(transwcd.state_dict(), ckpt_name)

            if cam_score['iou'][1] > best_iou:
                best_iou = cam_score['iou'][1]
                best_iou_iter = n_iter + 1
                
                pth_files = Path(cfg.work_dir.ckpt_dir).glob('*.pth')
                for pth_file in pth_files:
                    if "transwcd_iou_iter_" in str(pth_file):
                        pth_file.unlink()
                
                ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "transwcd_iou_iter_%d.pth" % (n_iter + 1))
                torch.save(transwcd.state_dict(), ckpt_name)

            logging.info("val cams score: %s, \n[best_iou_iter]: %s", cam_score, best_iou_iter)
            logging.info("val cls_score: %s, \n[best_F1_iter]: %s", cls_score, best_F1_iter)
            
            if args.wandb:
                wandb.log({
                    "val/accuracy": float(accuracy.item()), 
                    "val/f1": float(f1.item()),
                    "val/fg_iou": cam_score['iou'][1],
                    "train/loss": train_cc_loss,
                    "lr": cur_lr,
                    "total_norm": total_norm,
                    "epoch": epoch
                })
              
    return True

def flatten_omegaconf(conf, parent_key='', sep='.'):
    """
    Flattens an OmegaConf object to a dictionary with dot notation keys.

    Parameters:
    - conf: The OmegaConf object to flatten.
    - parent_key (str): The base key for recursion (used internally).
    - sep (str): The separator for keys (default is a dot).

    Returns:
    - dict: A flattened dictionary with dot notation keys.
    """
    items = {}
    for k, v in conf.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if OmegaConf.is_dict(v):
            items.update(flatten_omegaconf(v, new_key, sep))
        else:
            items[new_key] = v
    return items

if __name__ == "__main__":
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    if args.wandb:
        # Convert to OmegaConf object
        conf = OmegaConf.create(cfg)
        flattened_dict = flatten_omegaconf(conf)

        wandb.login()

        run = wandb.init(
            # Set the project where this run will be logged
            project=args.project,
            name=args.config.split("/")[-1].split(".")[0],
            # Track hyperparameters and run metadata
            config=flattened_dict,
        )

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.logger_dir, timestamp)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.logger_dir, exist_ok=True)

    setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp + '.log'))
    logging.info('\nargs: %s' % args)
    logging.info('\nconfigs: %s' % cfg)

    setup_seed(1)
    train(cfg=cfg)

