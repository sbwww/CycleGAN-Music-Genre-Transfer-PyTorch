import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataloader import CustomDataset, MusicDataset
from model import CycleGAN
from utils import save_midis, to_binary


def make_parses():
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument(
        '--data_dir',
        default=None,
        type=str,
        help="Custom data directory. Only set this if you use custom dataset."
    )
    parser.add_argument(
        '--model_dir',
        default='',
        type=str,
        help="saved_model directory."
    )
    parser.add_argument(
        '--batch_size',
        default=1,
        type=int
    )
    parser.add_argument(
        '--model_name',
        default='CP',
        type=str,
        help='Optional: CP, JC, JP'
    )
    parser.add_argument(
        '--test_mode',
        default='A2B',
        type=str,
        help='Optional: A2B, B2A'
    )
    return parser.parse_args()


def test():
    # JC CP JP.
    args = make_parses()
    model_name = args.model_name
    mode = args.test_mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = args.data_dir if args.data_dir else os.path.join(os.getcwd(), 'data' + os.sep)

    save_dir = os.path.join(os.getcwd(), 'test' + os.sep)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.data_dir is None:
        music_dataset = MusicDataset(data_dir, train_mode='CP', is_train='test')
    else:
        music_dataset = CustomDataset(data_dir)

    music_dataloader = DataLoader(
        music_dataset, batch_size=1, shuffle=False, num_workers=0)
    print("test dataset contains {} items".format(len(music_dataset)))

    # ------- 3. define model --------
    model = CycleGAN(mode=mode, device=device)
    checkpoint = torch.load(args.model_dir)
    if 'model_name' in checkpoint.keys():
        assert checkpoint['model_name'] == model_name
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # ------- 5. training process --------
    print("---start testing...")

    for i, data in enumerate(tqdm(music_dataloader)):
        real_a, real_b, real_mixed = data['bar_a'], data['bar_b'], data['bar_mixed']
        real_a = torch.FloatTensor(real_a)
        real_b = torch.FloatTensor(real_b)
        real_mixed = torch.FloatTensor(real_mixed)

        model.set_input(real_a, real_b, real_mixed)
        transfered, cycle = model()
        transfered = transfered.permute(0, 2, 3, 1)
        cycle = cycle.permute(0, 2, 3, 1)

        trans_np = to_binary(transfered.detach().cpu().numpy())
        cycle_np = to_binary(cycle.detach().cpu().numpy())

        name = music_dataset._get_name(data['baridx'])
        # print('save to ' + save_dir + name + '_transfered.mid')
        save_midis(trans_np, save_dir + name + '_transfered.mid')
        save_midis(real_a.permute(0, 2, 3, 1).detach().cpu().numpy(),
                   save_dir + name + '_origin.mid')
        save_midis(cycle_np, save_dir + name + '_cycle.mid')


if __name__ == '__main__':
    test()
