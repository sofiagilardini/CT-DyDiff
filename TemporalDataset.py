import numpy as np 
import torch 
import matplotlib.pyplot as plt
import os
import random




class TemporalDataset:
    """handle dataset"""

    def __init__(self, dst_name:str):
        self.dst_name = 'synthetic'

    def load_dataset(self):
        
        if self.dst_name == 'synthetic':
            data_path_head = './synth_temporal_data'
            image_data = os.path.join(data_path_head, "images.npy")
            # data.shape (500, 20, 32, 32, 3), (Patients, timesteps, pixel, pixel, RGB)
            
        self.data = np.load(image_data)
        

        return self.data

    def temporal_cut(self, masking_dec:float):
        """
        masking_dec = 0.15 : decimal, how much we want to mask. 
        """

        timesteps = np.arange(self.data.shape[1])
        keep_count = int(len(timesteps) * (1-masking_dec))

        self.num_patients = self.data.shape[0]

        self.real_timesteps_kept = np.array([np.sort(np.random.choice(timesteps, keep_count, replace=False)) for _ in range(self.num_patients)])

        sample_indx = np.arange(self.num_patients)[:, None]
        self.filtered_data = self.data[sample_indx, self.real_timesteps_kept, :, :, :]

        return self.filtered_data, self.real_timesteps_kept
    

    def trainval(self, train_split=0.8):
        """
        If train=True, training split, else: val
        """

        # Split into train/val by patients
        num_train = int(self.num_patients * train_split)

        self.train_data = self.data[:num_train]
        self.real_timesteps_kept_train = self.real_timesteps_kept[:num_train]

        self.val_data = self.data[num_train:]
        self.real_timesteps_kept_val = self.real_timesteps_kept[num_train:]

        # Create triplets (current_frame, prev_frame, delta_t)
        # delta_t = t(current-prev)


    def create_triplets(self, train:bool, max_gap=None, gap_weights = 'exponential'):
        """
        Sample pairs with variable gaps. 
        Smaller gaps more common (exponential) (based on consecutive)
        """

        triplets = []

        if train: 
            data = self.train_data
            timesteps = self.real_timesteps_kept_train

        else: 
            data = self.val_data
            timesteps = self.real_timesteps_kept_val


        for patient_idx in range(data.shape[0]):
            n_times = len(timesteps)

            # for each valid current frame, sample a previous frame

            for curr_idx in range(1, n_times):
                # all valid previous indices 
                valid_prev = list(range(curr_idx))

                if gap_weights == 'exponential':
                    # higher weight towards recent frames

                    weights = np.array([np.exp(-0.5 * (curr_idx - prev_idx)) for prev_idx in valid_prev])
                    weights /= weights.sum()

                    # sample one previous frame
                    prev_idx = np.random.choice(valid_prev, p=weights)

                else: 
                    prev_idx = np.random.choice(valid_prev)
            
            curr_frame = self.train_data[patient_idx, curr_idx]
            prev_frame = self.train_data[patient_idx, prev_idx]

            # delta_t difference in frames
            delta_t = timesteps[curr_idx] - timesteps[prev_idx]

            triplets.append((curr_frame, prev_frame, delta_t))

        return triplets




        











        

        





