# Modeling Irregular Time Series with Continuous Recurrent Units (CRUs)
# Copyright (c) 2022 Robert Bosch GmbH
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
import os
from lib.pendulum_generation import generate_pendulums


# new code component 
def load_data(args):
    file_path = f'{args.dir_name}/{args.dataset}/'
    
    # Pendulum 
    if args.dataset == 'pendulum':
        
        if args.task == 'interpolation':
            raise NotImplementedError()
        
        elif args.task == 'regression':
            if not os.path.exists(os.path.join(file_path, 'pend_regression.npz')):
                print(f'Generating pendulum trajectories and saving to {file_path} ...')
                generate_pendulums(file_path, task=args.task)

            train = Pendulum_regression(file_path=file_path, name='pend_regression.npz',
                               mode='train', sample_rate=args.sample_rate, random_state=args.data_random_seed)
            test = Pendulum_regression(file_path=file_path, name='pend_regression.npz',
                                       mode='test', sample_rate=args.sample_rate, random_state=args.data_random_seed)
            valid = Pendulum_regression(file_path=file_path, name='pend_regression.npz',
                               mode='valid', sample_rate=args.sample_rate, random_state=args.data_random_seed)
        else:
            raise Exception('Task not available for Pendulum data')
        collate_fn = None
        
    # USHCN
    elif args.dataset == 'ushcn':
        raise NotImplementedError()
    
    # Physionet
    elif args.dataset == 'physionet':
        raise NotImplementedError()
    
    train_dl = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_dl = DataLoader(test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=args.pin_memory)
    valid_dl = DataLoader(valid, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=args.pin_memory)

    print('\nSanity Checksums:')
    print('Trn obs: ', train_dl.dataset.obs.sum())
    print('Tst obs: ', test_dl.dataset.obs.sum())
    print('Val obs: ', valid_dl.dataset.obs.sum(), '\n')

    return train_dl, test_dl, valid_dl


# new code component 
class Pendulum_regression(Dataset):
    def __init__(self, file_path, name, mode, sample_rate=0.5, random_state=0):

        data = dict(np.load(os.path.join(file_path, name)))
        train_obs, train_targets, valid_obs, valid_targets, test_obs, test_targets, \
        train_time_points, valid_time_points, test_time_points = subsample(
            data, sample_rate=sample_rate, random_state=random_state)

        if mode == 'train':
            self.obs = train_obs
            self.targets = train_targets
            self.time_points = train_time_points
        elif mode == 'valid':
            self.obs = valid_obs
            self.targets = valid_targets
            self.time_points = valid_time_points
        elif mode == 'test':
            self.obs = test_obs
            self.targets = test_targets
            self.time_points = test_time_points
        else:
            raise RuntimeError(f"Mode {mode} not recognised.")

        self.obs = np.ascontiguousarray(
            np.transpose(self.obs, [0, 1, 4, 2, 3]))/255.0

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.obs[idx, ...].astype(np.float64))
        targets = torch.from_numpy(self.targets[idx, ...].astype(np.float64))
        time_points = torch.from_numpy(self.time_points[idx, ...])
        obs_valid = torch.ones_like(time_points, dtype=torch.bool)
        return obs, targets, time_points, obs_valid


# new code component 
def subsample(data, sample_rate, imagepred=False, random_state=0):
    train_obs, train_targets = data["train_obs"], data["train_targets"]
    valid_obs, valid_targets = data["valid_obs"], data["valid_targets"]
    test_obs, test_targets = data["test_obs"], data["test_targets"]

    seq_length = train_obs.shape[1]
    train_time_points = []
    valid_time_points = []
    test_time_points = []
    n = int(sample_rate*seq_length)

    if imagepred:
        train_obs_valid = data["train_obs_valid"]
        data_components = train_obs, train_targets, train_obs_valid
        train_obs_sub, train_targets_sub, train_obs_valid_sub = [np.zeros_like(x[:, :n, ...]) for x in data_components]

        valid_obs_valid = data["valid_obs_valid"]
        data_components = valid_obs, valid_targets, valid_obs_valid
        valid_obs_sub, valid_targets_sub, valid_obs_valid_sub = [np.zeros_like(x[:, :n, ...]) for x in data_components]

        test_obs_valid = data["test_obs_valid"]
        data_components = test_obs, test_targets, test_obs_valid
        test_obs_sub, test_targets_sub, test_obs_valid_sub = [np.zeros_like(x[:, :n, ...]) for x in data_components]

    else:
        data_components = train_obs, train_targets, valid_obs, valid_targets, test_obs, test_targets
        train_obs_sub, train_targets_sub, valid_obs_sub, valid_targets_sub, test_obs_sub, test_targets_sub = [
            np.zeros_like(x[:, :n, ...]) for x in data_components]

    for i in range(train_obs.shape[0]):
        rng_train = np.random.default_rng(random_state+i+train_obs.shape[0])
        choice = np.sort(rng_train.choice(seq_length, n, replace=False))
        train_time_points.append(choice)
        train_obs_sub[i, ...], train_targets_sub[i, ...] = [
            x[i, choice, ...] for x in [train_obs, train_targets]]
        if imagepred:
            train_obs_valid_sub[i, ...] = train_obs_valid[i, choice, ...]

    for i in range(valid_obs.shape[0]):
        rng_valid = np.random.default_rng(random_state+i+valid_obs.shape[0])
        choice = np.sort(rng_valid.choice(seq_length, n, replace=False))
        valid_time_points.append(choice)
        valid_obs_sub[i, ...], valid_targets_sub[i, ...] = [
            x[i, choice, ...] for x in [valid_obs, valid_targets]]
        if imagepred:
            valid_obs_valid_sub[i, ...] = valid_obs_valid[i, choice, ...]

    for i in range(test_obs.shape[0]):
        rng_test = np.random.default_rng(random_state+i)
        choice = np.sort(rng_test.choice(seq_length, n, replace=False))
        test_time_points.append(choice)
        test_obs_sub[i, ...], test_targets_sub[i, ...] = [
            x[i, choice, ...] for x in [test_obs, test_targets]]
        if imagepred:
            test_obs_valid_sub[i, ...] = test_obs_valid[i, choice, ...]

    train_time_points, valid_time_points, test_time_points = np.stack(
        train_time_points, 0), np.stack(valid_time_points, 0), np.stack(test_time_points, 0)

    if imagepred:
        return train_obs_sub, train_targets_sub, train_time_points, train_obs_valid_sub, \
               valid_obs_sub, valid_targets_sub, valid_time_points, valid_obs_valid_sub, \
               test_obs_sub, test_targets_sub, test_time_points, test_obs_valid_sub
    else:
        return train_obs_sub, train_targets_sub, valid_obs_sub, valid_targets_sub, test_obs_sub, test_targets_sub, \
               train_time_points, valid_time_points, test_time_points

