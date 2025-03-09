import torch
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from denseav.shared import load_trained_model
from denseav.plotting import _prep_sims_for_plotting
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torchvision
import torchaudio
import json
from pycocotools import mask as maskUtils
from spatioalignment_utils import get_alignment_score_object, \
    get_glancing_score_object, get_alignment_score_word, \
        get_glancing_score_word
from collections import defaultdict
import wandb

def decode_mask(rle):
    if isinstance(rle["count"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return maskUtils.decode(rle)


def time_to_frame(t, sampling_rate=16000):
    return round(t * sampling_rate)

FPS = 8

class AlignmentDataset(Dataset):
    def __init__(self, dataset_csv, img_dir, audio_dir, annotations_dir):
        self.img_dir = img_dir
        self.audio_dir = audio_dir
        self.annotations_dir = annotations_dir
        self.triplets = pd.read_csv(dataset_csv)
        self.img_files = self.triplets["img"]
        self.wav_files = self.triplets["wav"]
        self.annotations = self.triplets["annotations"]

    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = torchvision.io.read_image(img_path)
        
        audio_path = os.path.join(self.audio_dir, self.wav_files[idx])
        wav, audio_sample_rate = torchaudio.load(audio_path)
        duration = wav.shape[1] / audio_sample_rate
        # Resample to 16000Hz if not already
        if audio_sample_rate != 16000:
            wav = torchaudio.transforms.Resample(orig_freq=audio_sample_rate,
                                                 new_freq=16000)(wav)

        gt = self.annotations[idx]

        n_frames = int(duration * FPS)
        img.unsqueeze(0).repeat(n_frames, 1, 1, 1)

        return img.to(torch.float32), wav.to(torch.float32), gt


def eval_model(model, dataloader, device="cpu"):

    output_list = []

    with torch.no_grad():
        for idx, sample in enumerate(dataloader):
            img, audio, gt = sample
            audio = audio.squeeze(1)
            
            audio_feats = model.forward_audio({"audio": audio.to(device)})
            img_feats = model.forward_image({"frames": img.to(device)})

            sims = model.sim_agg.get_pairwise_sims(
                {**img_feats, **audio_feats},
                 raw=False,
                 agg_sim=False,
                 agg_heads=True
            ).mean(dim=-2)

            sims = _prep_sims_for_plotting(sims, frames=img.shape[0])
            
            for label in gt:
                obj = gt[label]
                mask = obj["mask"]
                start, end = time_to_frame(obj["start"], sampling_rate=FPS), time_to_frame(obj["end"], sampling_rate=FPS)
                mask = decode_mask(mask)

                AS_obj = get_alignment_score_object(sims, start, end, mask)
                GS_obj = get_glancing_score_object(sims, start, end, mask)
                AS_word = get_alignment_score_word(sims, start, end, mask)
                GS_word = get_glancing_score_word(sims, start, end, mask)

                score_dict = {
                    "label": label,
                    "image_id": idx, 
                    "AS_obj": AS_obj, "GS_obj": GS_obj,
                    "AS_word": AS_word, "GS_word": GS_word
                }

                output_list.append(score_dict)
    
    score_df = pd.DataFrame.from_dict(output_list)
    
    return score_df


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "language"
    model = torch.hub.load('mhamilton723/DenseAV', model_name)
    model.to(device)

    wandb.init(project="eval_denseav", settings=wandb.Settings(start_method="fork"))

    alignment_data = AlignmentDataset("/home/brdiep/scratch/flickr_eval_triplets/flickr_eval_triplets.csv",
                                  img_dir="/home/brdiep/scratch/flickr_eval_triplets/imgs",
                                  audio_dir="/home/brdiep/scratch/flickr_eval_triplets/wavs",
                                  annotations_dir="~/scratch/anns")
    eval_dataloader  = DataLoader(alignment_data, batch_size=1)
    df = eval_model(model, eval_dataloader, device)

    score_table = wandb.Table(dataframe=df)
    wandb.log({"Output": score_table})
