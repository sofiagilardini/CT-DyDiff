import numpy as np 
import torch 
import matplotlib.pyplot as plt
import os
import random


class Triplet(torch.utils.data.Dataset):
    
    def __init__(self, triplets, D=1000):
        self.triplets = triplets
        self.D = D

        self._noise_schedule()

    def __len__(self):
        # how many samples? 
        return len(self.triplets)
    
    def __getitem__(self, index):

        og_triplet = self.triplets[index]
        
        # TODO: make this dict.
        clean_curr = og_triplet[0]
        prev_frame = og_triplet[1]
        delta_t = og_triplet[2]
        
        d_level = np.random.randint(0, self.D)
        noise = np.random.normal(size=clean_curr.shape)

        noisy_curr = self._add_noise(clean_curr=clean_curr, 
                                     noise=noise, 
                                     d_level=d_level)
        
        return {
            'noisy_curr' : torch.tensor(noisy_curr, dtype=torch.float32), 
            'd_level' : torch.tensor(d_level, dtype=torch.float32), 
            'prev_frame' : torch.tensor(prev_frame, dtype=torch.float32), 
            'delta_t' : torch.tensor(delta_t, dtype=torch.float32), 
            'noise' : torch.tensor(noise, dtype=torch.float32)
        }

    
    def _noise_schedule(self):


        betas = np.linspace(0.0001, 0.02, self.D)
        alphas_raw = [(1-beta) for beta in betas]

        self.alphas_d = np.cumprod(alphas_raw)

    
    
    def _add_noise(self, clean_curr, noise, d_level) -> tuple:

        alpha_d = self.alphas_d[d_level]

        noisy_curr = np.sqrt(alpha_d) * clean_curr + np.sqrt(1-alpha_d) * noise

        return noisy_curr








    

