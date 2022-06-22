import torch
import numpy as np

import soundfile as sf

from model_encoder import Encoder, Encoder_lf0
from model_decoder import Decoder_ac
from model_encoder import SpeakerEncoder as Encoder_spk
import os

import subprocess
from spectrogram import logmelspectrogram
import kaldiio

import resampy
import pyworld as pw

import argparse


class VQMIVC(torch.nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.encoder = Encoder(in_channels=80, channels=512, n_embeddings=512, z_dim=64, c_dim=256)
        self.encoder_lf0 = Encoder_lf0()
        self.encoder_spk = Encoder_spk()
        self.decoder = Decoder_ac(dim_neck=64)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.encoder_spk.load_state_dict(checkpoint["encoder_spk"])
            self.decoder.load_state_dict(checkpoint["decoder"])

    def forward(self, src_mel, src_lf0, ref_mel):
        z, _, _, _ = self.encoder.encode(src_mel)
        lf0_embs = self.encoder_lf0(src_lf0)
        spk_emb = self.encoder_spk(ref_mel)
        converted_mel = self.decoder(z, lf0_embs, spk_emb)
        return converted_mel


def extract_logmel(wav_path, mean, std, sr=16000):
    # wav, fs = librosa.load(wav_path, sr=sr)
    wav, fs = sf.read(wav_path)
    if fs != sr:
        wav = resampy.resample(wav, fs, sr, axis=0)
        fs = sr
    # wav, _ = librosa.effects.trim(wav, top_db=15)
    # duration = len(wav)/fs
    assert fs == 16000
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak
    mel = logmelspectrogram(
        x=wav,
        fs=fs,
        n_mels=80,
        n_fft=400,
        n_shift=160,
        win_length=400,
        window='hann',
        fmin=80,
        fmax=7600,
    )

    mel = (mel - mean) / (std + 1e-8)
    tlen = mel.shape[0]
    frame_period = 160 / fs * 1000
    f0, timeaxis = pw.dio(wav.astype('float64'), fs, frame_period=frame_period)
    f0 = pw.stonemask(wav.astype('float64'), f0, timeaxis, fs)
    f0 = f0[:tlen].reshape(-1).astype('float32')
    nonzeros_indices = np.nonzero(f0)
    lf0 = f0.copy()
    lf0[nonzeros_indices] = np.log(f0[nonzeros_indices])  # for f0(Hz), lf0 > 0 when f0 != 0
    mean, std = np.mean(lf0[nonzeros_indices]), np.std(lf0[nonzeros_indices])
    lf0[nonzeros_indices] = (lf0[nonzeros_indices] - mean) / (std + 1e-8)
    return mel, lf0


def export_onnx(params):
    from torch.onnx import export as ex_to_onnx
    onnx_path = 'VQMIVC.onnx'
    src_wav_path = params.source_wav
    ref_wav_path = params.reference_wav

    out_dir = params.converted_wav_path
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vqmivc_model = VQMIVC(params.model_path)
    vqmivc_model.eval()

    mel_stats = np.load('./mel_stats/stats.npy')
    mean = mel_stats[0]
    std = mel_stats[1]
    feat_writer = kaldiio.WriteHelper("ark,scp:{o}.ark,{o}.scp".format(o=str(out_dir) + '/feats.1'))
    src_mel, src_lf0 = extract_logmel(src_wav_path, mean, std)
    ref_mel, _ = extract_logmel(ref_wav_path, mean, std)
    src_mel = torch.FloatTensor(src_mel.T).unsqueeze(0).to(device)
    src_lf0 = torch.FloatTensor(src_lf0).unsqueeze(0).to(device)
    ref_mel = torch.FloatTensor(ref_mel.T).unsqueeze(0).to(device)
    out_filename = os.path.basename(src_wav_path).split('.')[0]
    with torch.no_grad():
        converted_mel = vqmivc_model(src_mel, src_lf0, ref_mel)
        ex_to_onnx(vqmivc_model,
                   (src_mel, src_lf0, ref_mel),
                   onnx_path,
                   export_params=True,
                   verbose=True,
                   opset_version=14,
                   do_constant_folding=True,
                   input_names=['src_mel_spectrogram',
                                'normed_fundamental_freq',
                                'ref_mel_spectrogram'],
                   output_names=['converted_mel_spectrogram'],
                   dynamic_axes={'src_mel_spectrogram': {0: 'batch_size'},
                                 'normed_fundamental_freq': {0: 'batch_size'},
                                 'ref_mel_spectrogram': {0: 'batch_size'},
                                 'converted_mel_spectrogram': {0: 'batch_size'}})

        feat_writer[out_filename + '_converted'] = converted_mel.squeeze(0).cpu().numpy()
        feat_writer[out_filename + '_source'] = src_mel.squeeze(0).cpu().numpy().T
        feat_writer[out_filename + '_reference'] = ref_mel.squeeze(0).cpu().numpy().T


def convert(args):
    src_wav_path = args.source_wav
    ref_wav_path = args.reference_wav

    out_dir = args.converted_wav_path
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(in_channels=80, channels=512, n_embeddings=512, z_dim=64, c_dim=256)
    encoder_lf0 = Encoder_lf0()
    encoder_spk = Encoder_spk()
    decoder = Decoder_ac(dim_neck=64)
    encoder.to(device)
    encoder_lf0.to(device)
    encoder_spk.to(device)
    decoder.to(device)

    checkpoint_path = args.model_path
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder_spk.load_state_dict(checkpoint["encoder_spk"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    encoder_spk.eval()
    decoder.eval()

    mel_stats = np.load('./mel_stats/stats.npy')
    mean = mel_stats[0]
    std = mel_stats[1]
    feat_writer = kaldiio.WriteHelper("ark,scp:{o}.ark,{o}.scp".format(o=str(out_dir) + '/feats.1'))
    src_mel, src_lf0 = extract_logmel(src_wav_path, mean, std)
    ref_mel, _ = extract_logmel(ref_wav_path, mean, std)
    src_mel = torch.FloatTensor(src_mel.T).unsqueeze(0).to(device)
    src_lf0 = torch.FloatTensor(src_lf0).unsqueeze(0).to(device)
    ref_mel = torch.FloatTensor(ref_mel.T).unsqueeze(0).to(device)
    out_filename = os.path.basename(src_wav_path).split('.')[0]
    with torch.no_grad():
        z, _, _, _ = encoder.encode(src_mel)
        lf0_embs = encoder_lf0(src_lf0)
        spk_emb = encoder_spk(ref_mel)
        output = decoder(z, lf0_embs, spk_emb)

        feat_writer[out_filename + '_converted'] = output.squeeze(0).cpu().numpy()
        feat_writer[out_filename + '_source'] = src_mel.squeeze(0).cpu().numpy().T
        feat_writer[out_filename + '_reference'] = ref_mel.squeeze(0).cpu().numpy().T

    feat_writer.close()
    print('synthesize waveform...')
    cmd = ['parallel-wavegan-decode', '--checkpoint',
           './vocoder/checkpoint-3000000steps.pkl',
           '--feats-scp', f'{str(out_dir)}/feats.1.scp', '--outdir', str(out_dir)]
    subprocess.call(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_wav', '-s', type=str, required=True)
    parser.add_argument('--reference_wav', '-r', type=str, required=True)
    parser.add_argument('--converted_wav_path', '-c', type=str, default='converted')
    parser.add_argument('--model_path', '-m', type=str, required=True)
    args = parser.parse_args()
    convert(args)
