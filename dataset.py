from torch.utils.data import Dataset
import numpy as np
from bisect import bisect_right
import random

class MahjongGBDataset(Dataset):
    
    def __init__(self, begin = 0, end = 1, augment = False):
        import json
        with open('data/count.json') as f:
            self.match_samples = json.load(f)
        self.total_matches = len(self.match_samples)
        self.total_samples = sum(self.match_samples)
        self.begin = int(begin * self.total_matches)
        self.end = int(end * self.total_matches)
        self.match_samples = self.match_samples[self.begin : self.end]
        self.matches = len(self.match_samples)
        self.samples = sum(self.match_samples)
        self.augment = augment
        t = 0
        for i in range(self.matches):
            a = self.match_samples[i]
            self.match_samples[i] = t
            t += a
        self.cache = {'obs': [], 'mask': [], 'act': []}
        for i in range(self.matches):
            if i % 128 == 0: print('loading', i)
            d = np.load('data/%d.npz' % (i + self.begin))
            for k in d:
                self.cache[k].append(d[k])
    
    def __len__(self):
        return self.samples
    
    def __getitem__(self, index):
        match_id = bisect_right(self.match_samples, index, 0, self.matches) - 1
        sample_id = index - self.match_samples[match_id]
        obs = self.cache['obs'][match_id][sample_id]
        mask = self.cache['mask'][match_id][sample_id]
        act = self.cache['act'][match_id][sample_id]

        if self.augment:
            obs = self._augment_data(obs)

        return obs, mask, act

    def _augment_data(self, obs):
        # 1. 花色互换
        if random.random() < 0.5:
            obs = self._swap_suits(obs)
        
        # 2. 数字镜像
        if random.random() < 0.5:
            obs = self._mirror_numbers(obs)
        
        # 3. 打乱手牌顺序
        # obs = self._shuffle_hand(obs)
        
        return obs

    def _swap_suits(self, obs):
        # 假设万筒条分别在0-8, 9-17, 18-26列
        suits = [obs[:, :9], obs[:, 9:18], obs[:, 18:27]]
        random.shuffle(suits)
        return np.concatenate([*suits, obs[:, 27:]], axis=1)

    def _mirror_numbers(self, obs):
        # 对数字牌进行镜像
        for i in range(3):  # 万筒条
            start = i * 9
            end = start + 9
            obs[:, start:end] = obs[:, start:end][:, ::-1]
        return obs

    def _shuffle_hand(self, obs):
        hand = obs[2:6]
        # 打乱非零元素的顺序
        non_zero = np.nonzero(hand)
        values = hand[non_zero]
        np.random.shuffle(values)
        hand[non_zero] = values
        obs[2:6] = hand
        return obs

# from torch.utils.data import Dataset
# import numpy as np
# from bisect import bisect_right
# import json
# from collections import OrderedDict

# class MahjongGBDataset(Dataset):
#     def __init__(self, begin=0, end=1, augment=False, cache_size=10):
#         with open('data/count.json') as f:
#             self.match_samples = json.load(f)
#         self.total_matches = len(self.match_samples)
#         self.total_samples = sum(self.match_samples)
#         self.begin = int(begin * self.total_matches)
#         self.end = int(end * self.total_matches)
#         self.match_samples = self.match_samples[self.begin : self.end]
#         self.matches = len(self.match_samples)
#         self.samples = sum(self.match_samples)
#         self.augment = augment

#         # Calculate cumulative samples
#         t = 0
#         for i in range(self.matches):
#             a = self.match_samples[i]
#             self.match_samples[i] = t
#             t += a

#         # Initialize cache
#         self.cache_size = cache_size
#         self.cache = OrderedDict()

#     def __len__(self):
#         return self.samples

#     def __getitem__(self, index):
#         match_id = bisect_right(self.match_samples, index, 0, self.matches) - 1
#         sample_id = index - self.match_samples[match_id]

#         # Check if data is in cache
#         if match_id not in self.cache:
#             # If cache is full, remove the least recently used item
#             if len(self.cache) >= self.cache_size:
#                 self.cache.popitem(last=False)

#             # Load data
#             data = np.load(f'data/{match_id + self.begin}.npz')
#             self.cache[match_id] = {
#                 'obs': data['obs'],
#                 'mask': data['mask'],
#                 'act': data['act']
#             }
#         else:
#             # Move this item to the end to mark it as recently used
#             self.cache.move_to_end(match_id)

#         return (
#             self.cache[match_id]['obs'][sample_id],
#             self.cache[match_id]['mask'][sample_id],
#             self.cache[match_id]['act'][sample_id]
#         )