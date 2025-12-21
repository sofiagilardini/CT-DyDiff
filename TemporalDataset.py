import numpy as np 
import torch 
import matplotlib.pyplot as plt
import os
import random




class TemporalDataset:
    """handle dataset"""

    def __init__(self, dst_name:str):
        """
        dst_name: ['synth0', 'synth1']
        """
        self.dst_name = dst_name

    def _load_dataset(self):

        ## TODO: handle greyscale & RGB
        
        if self.dst_name == 'synth0':
            data_path_head = './toy_problem_data0'
            image_data = os.path.join(data_path_head, "images.npy")
            # data.shape (500, 20, 32, 32, 3), (Patients, timesteps, pixel, pixel, RGB)

        if self.dst_name == 'synth1':
            data_path_head = './toy_problem_data1'
            image_data = os.path.join(data_path_head, "images.npy")
            
        self.data = np.load(image_data)
        

        return self.data

    def _temporal_cut(self, masking_dec:float):
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
    

    def _trainvalsplit(self, train_split=0.8):
        """
        If train=True, training split, else: val
        """

        # Split into train/val by patients
        num_train = int(self.num_patients * train_split)

        self.train_data = self.data[:num_train]
        self.real_timesteps_kept_train = self.real_timesteps_kept[:num_train]

        self.val_data = self.data[num_train:]
        self.real_timesteps_kept_val = self.real_timesteps_kept[num_train:]

    def prepare_dataset(self,masking_dec:float, train_split=0.8):
        self._load_dataset()
        self._temporal_cut(masking_dec)
        self._trainvalsplit(train_split=train_split)



    def create_triplets(self, train:bool, max_gap=None, gap_weights = 'exponential'):
        """
        Sample pairs with variable gaps. 
        Smaller gaps more common (exponential) (based on consecutive)
        """

        triplets = []


        data = self.train_data if train else self.val_data


        for patient_idx in range(data.shape[0]):

            if train: 
                timesteps = self.real_timesteps_kept_train[patient_idx] # self.real_timesteps_kept_train shape: (patients, length of sequence)

            else: 
                timesteps = self.real_timesteps_kept_val[patient_idx]
            
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
            
            curr_frame = data[patient_idx, curr_idx]
            prev_frame = data[patient_idx, prev_idx]

            # delta_t difference in frames
            delta_t = timesteps[curr_idx] - timesteps[prev_idx]


            triplets.append((curr_frame, prev_frame, delta_t))

        return triplets
    
    def visualise_triplet(self, triplets, num_samples=5, figsize=(15, 3)):
        """
        Visualise a few triplets showing previous and current frames with delta_t.
        Handles both greyscale and RGB datasets automatically.

        Args:
            triplets: List of (current_frame, prev_frame, delta_t) tuples
            num_samples: Number of triplets to visualize
            figsize: Figure size for each row of visualization
        """
        num_samples = min(num_samples, len(triplets))

        for i in range(num_samples):
            curr_frame, prev_frame, delta_t = triplets[i]

            fig, axes = plt.subplots(1, 2, figsize=figsize)

            # Normalize images to [0, 1] range for display
            def normalize_image(img):
                img = img.astype(np.float32)
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min)
                return np.clip(img, 0, 1)

            # Detect if greyscale or RGB
            # Greyscale: shape (H, W) or (H, W, 1)
            # RGB: shape (H, W, 3)
            is_greyscale = prev_frame.ndim == 2 or (prev_frame.ndim == 3 and prev_frame.shape[2] == 1)

            # Prepare images for display
            prev_normalized = normalize_image(prev_frame)
            curr_normalized = normalize_image(curr_frame)

            # Squeeze channel dimension if greyscale with shape (H, W, 1)
            if prev_frame.ndim == 3 and prev_frame.shape[2] == 1:
                prev_normalized = prev_normalized.squeeze(-1)
                curr_normalized = curr_normalized.squeeze(-1)

            # Display previous frame
            if is_greyscale:
                axes[0].imshow(prev_normalized, cmap='gray')
            else:
                axes[0].imshow(prev_normalized)
            axes[0].set_title(f'Previous Frame', fontsize=12, fontweight='bold')
            axes[0].axis('off')

            # Display current frame
            if is_greyscale:
                axes[1].imshow(curr_normalized, cmap='gray')
            else:
                axes[1].imshow(curr_normalized)
            axes[1].set_title(f'Current Frame', fontsize=12, fontweight='bold')
            axes[1].axis('off')

            # Add delta_t as figure title
            fig.suptitle(f'Triplet {i+1} | Δt = {delta_t} timesteps',
                        fontsize=14, fontweight='bold', y=1.02)

            plt.tight_layout()
            plt.show()

    



        











        

        





