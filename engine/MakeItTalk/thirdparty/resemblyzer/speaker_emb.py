from .hparams import *
from .audio import normalize_volume
from .voice_encoder import VoiceEncoder
import numpy as np
import torch
import librosa

def pp_wav(fpath_or_wav):
    # Load the wav from disk
    wav, source_sr = librosa.load(fpath_or_wav, sr=None)
    
    # Resample the wav
    if source_sr is not None:
        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=sampling_rate)
        
    # Apply the preprocessing: normalize volume and shorten long silences 
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    #wav = trim_long_silences(wav)
    
    return wav

def get_spk_emb(audio_file_dir, segment_len=960000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resemblyzer_encoder = VoiceEncoder(device=device)

    wav = pp_wav(audio_file_dir)
    l = len(wav) // segment_len # segment_len = 16000 * 60
    l = np.max([1, l])
    all_embeds = []
    for i in range(l):
        mean_embeds, cont_embeds, wav_splits = resemblyzer_encoder.embed_utterance(
            wav[segment_len * i:segment_len* (i + 1)], return_partials=True, rate=2)
        all_embeds.append(mean_embeds)
    all_embeds = np.array(all_embeds)
    mean_embed = np.mean(all_embeds, axis=0)

    return mean_embed, all_embeds
