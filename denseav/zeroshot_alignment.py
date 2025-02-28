import torch
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from denseav.data.EvaluationDatasets import FlickrAudio
from denseav.shared import load_trained_model
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

def decode_mask(rle):
    if isinstance(rle["count"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return maskUtils.decode(rle)


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

        # Resample to 16000Hz if not already
        if audio_sample_rate != 16000:
            wav = torchaudio.transforms.Resample(orig_freq=audio_sample_rate,
                                                 new_freq=16000)(wav)

        annotation_path = os.path.join(self.annotations, self.annotations[idx])
        with open(annotation_path, "r") as annotations:
            gt = json.load(annotations)

        return img, wav, gt


def eval_model(model, dataloader, device="cpu"):

    output_list = []

    with torch.no_grad():
        for idx, sample in enumerate(dataloader):
            audio, img, gt = sample
            audio_feats = model.forward_audio({"audio": audio})
            img_feats = model.forward_image({"image": img})

            sims = model.sim_agg.get_pairwise_sims(
                {**img_feats, **audio_feats},
                 raw=False,
                 agg_sim=False,
                 agg_heads=True
            ).mean(dim=-2)
            
            for obj in gt:
                label, mask, start, end = obj["class"], obj["mask"], obj["start"], obj["end"]
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

    alignment_data = AlignmentDataset("~/scratch/flickr8k.csv",
                                  img_dir="~/scratch/img",
                                  audio_dir="~/scratch/wav",
                                  annotations_dir="~/scratch/anns")
    eval_dataloader  = DataLoader(alignment_data, batch_size=1)
    df = eval_model(model, eval_dataloader, device)
