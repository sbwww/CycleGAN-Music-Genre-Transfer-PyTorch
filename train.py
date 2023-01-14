import argparse
import logging
import os
import sys
import time

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataloader import MusicDataset
from model import CycleGAN

log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt="%m/%d %I:%M:%S %p")
fh = logging.FileHandler("info.log")
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()


def make_parses():
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument(
        '--epoch',
        default=30,
        type=int
    )
    parser.add_argument(
        '--batch_size',
        default=2,
        type=int
    )
    parser.add_argument(
        '--model_name',
        default='CP',
        type=str,
        help='Optional: CP, JC, JP'
    )
    parser.add_argument(
        '--resume',
        default='',
        type=str,
        help="Checkpoint directory. Only set this if you want to resume training from a checkpoint."
    )
    parser.add_argument(
        '--gamma',
        default=1,
        type=float,
        help="Weight of extra discriminator loss."
    )
    parser.add_argument(
        '--sigma',
        default=0.01,
        type=float,
        help="Variance of Gaussian noise. \sigma^2 in actual."
    )
    parser.add_argument(
        '--lamb',
        default=10,
        type=float,
        help="Weight of cycle consistency loss."
    )
    parser.add_argument(
        '--sample_size',
        default=50,
        type=int,
        help="Max size of Sampler."
    )
    parser.add_argument(
        '--save_frq',
        default=1000,
        type=int,
        help="Save model checkpoint every X steps."
    )
    parser.add_argument(
        '--log_frq',
        default=1000,
        type=int,
        help="Log loss every X steps."
    )
    parser.add_argument(
        '--lr',
        default=2e-3,
        type=float,
        help="Learning rate."
    )
    parser.add_argument(
        '--wd',
        default=1e-2,
        type=float,
        help="Weight decay"
    )
    parser.add_argument(
        '--data_mode',
        default='full',
        type=str,
        help='Optional: full, partial'
    )
    return parser.parse_args()


def train():
    # ------- set the directory of training dataset --------
    args = make_parses()
    model_name = args.model_name  # JC CP JP.
    writer = SummaryWriter(comment=model_name+str(time.time()))

    data_dir = os.path.join(os.getcwd(), 'data' + os.sep)

    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_num = args.epoch
    batch_size_train = args.batch_size

    save_frq = args.save_frq
    log_frq = args.log_frq

    music_dataset = MusicDataset(data_dir, train_mode=model_name, data_mode=args.data_mode, is_train='train')
    train_num = len(music_dataset)
    logger.info("train data contains 2*{} items".format(train_num))
    music_dataloader = DataLoader(
        music_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0, drop_last=True)

    logger.info("load model with mode {}".format(model_name))

    # ------- define model --------
    model = CycleGAN(sigma=args.sigma, sample_size=args.sample_size, lamb=args.lamb,
                     mode='train', lr=args.lr, wd=args.wd, gamma=args.gamma, device=device)
    if args.resume:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        if 'model_name' in checkpoint.keys():
            assert model_name == checkpoint['model_name']
        model.load_state_dict(checkpoint['state_dict'])
    else:
        start_epoch = 0

    if torch.cuda.is_available():
        model.cuda()

    # ------- training process --------
    logger.info("---start training---")
    ite = 0
    g_running_loss = 0.0
    d_running_loss = 0.0
    ite_num4val = 0

    start = time.time()
    for epoch in trange(start_epoch, epoch_num):
        model.train()
        for i, data in enumerate(tqdm(music_dataloader)):
            if data['baridx'].size()[0] < batch_size_train:
                continue

            ite = ite + 1
            ite_num4val = ite_num4val + 1
            real_a, real_b, real_mixed = data['bar_a'], data['bar_b'], data['bar_mixed']
            real_a = torch.FloatTensor(real_a)
            real_b = torch.FloatTensor(real_b)
            real_mixed = torch.FloatTensor(real_mixed)

            model.set_input(real_a, real_b, real_mixed)
            model.forward_and_optimize()

            g_running_loss += model.G_loss.data.item()
            d_running_loss += model.D_loss.data.item()

            writer.add_scalar('cycle_loss', model.c_loss, global_step=ite)
            writer.add_scalar('G_A2B_loss', model.G_A2B_loss, global_step=ite)
            writer.add_scalar('G_B2A_loss', model.G_B2A_loss, global_step=ite)
            writer.add_scalar('G_loss', model.G_loss, global_step=ite)
            writer.add_scalar('D_A_loss', model.D_A_loss, global_step=ite)
            writer.add_scalar('D_B_loss', model.D_B_loss, global_step=ite)
            writer.add_scalar('D_A_all_loss', model.D_A_all_loss, global_step=ite)
            writer.add_scalar('D_B_all_loss', model.D_B_all_loss, global_step=ite)
            writer.add_scalar('D_loss', model.D_loss, global_step=ite)

            if ite % log_frq == 0:
                end = time.time()
                logger.info("[epoch: %3d/%3d, "
                            "batch: %5d/%5d, "
                            "ite: %d, "
                            "time: %3f] "
                            "g_loss : %3f, "
                            "d_loss : %3f " % (
                                epoch + 1, epoch_num,
                                (i) * batch_size_train, train_num,
                                ite,
                                end - start,
                                g_running_loss / ite_num4val,
                                d_running_loss / ite_num4val))
                start = end

            if save_frq > 0 and ite % save_frq == 0:
                saved_model_name = model_dir + model_name + "_itr_%d_G_%3f_D_%3f.pth" % (
                    ite, g_running_loss / ite_num4val, d_running_loss / ite_num4val)
                torch.save({
                    'epoch': epoch,
                    'model_name': model_name,
                    'state_dict': model.state_dict()},
                    saved_model_name)
                logger.info("saved model {}".format(saved_model_name))
                model.train()
                g_running_loss = 0.0
                d_running_loss = 0.0
                ite_num4val = 0

    saved_model_name = model_dir + "final.pth"
    torch.save({
        'epoch': args.epoch,
        'model_name': model_name,
        'state_dict': model.state_dict()},
        saved_model_name)
    logger.info("saved model {}".format(saved_model_name))


if __name__ == "__main__":
    train()
