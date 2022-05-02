# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR  INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import argparse
from email.policy import default
import sys
import time
from pathlib import Path

import dllogger as DLLogger
import numpy as np
import torch
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity
from fastpitch.pitch_transform import pitch_transform_custom
from scipy.stats import norm
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import models
from common.tb_dllogger import (init_inference_metadata, stdout_metric_format,
                                unique_log_fpath)
from common.text import cmudict
from common.text.text_processing import TextProcessing


class MelGenerator:
    def __init__(self):
        self.device = None
        self.generator = None 
        self.p_arpabet = 1.0
        self.cmudict_path = 'cmudict/cmudict-0.7b'
        self.fastpitch = 'FastPitch_checkpoint_1000.pt' # checkpoint path
        self.cudnn_benchmark = True
        self.output = './gen_mels'
        self.amp = False
        self.ema = False
        self.pace = 1.0
        self.repeats = 1 
        self.cuda = True # Run inference on a GPU using CUDA
        self.speaker = 0 # Speaker ID for a multi-speaker model
        self.batch_size = 32
        self.symbol_set = 'english_basic'
        self.text_cleaners = ['english_cleaners_v2']
        self.warmup_steps = 0
        self.sampling_rate = 22050
        self.stft_hop_length = 256
        self.log_file='nvlog_infer.json'
        self.pitch_transform_amplify = 1.0
        self.pitch_transform_custom = False
        self.pitch_transform_flatten = False
        self.pitch_transform_invert = False
        self.pitch_transform_shift = 0.0
        #sys.argv = ['inference.py', '-i', '', '-o', './gen_mels',
        #            '--log-file', 'nvlog_infer.json', '--save-mels', '--fastpitch'
        #    , 'FastPitch_checkpoint_1000.pt', '--batch-size',
        #            '32', '--repeats', '1', '--warmup-steps', '0', '--speaker', '0', '--n-speakers', '1', '--cuda',
        #            '--cudnn-benchmark', '--p-arpabet', '1.0', '--energy-conditioning']
        parser = argparse.ArgumentParser(description='PyTorch FastPitch Inference',
                                         allow_abbrev=False)
        parser = self.parse_args(parser)
        #self.args, unk_args = parser.parse_known_args()
        # args.amp = False
        # args.batch_size = 32
        # args.cmudict_path = 'cmudict/cmudict-0.7b'
        # args.cuda = True
        # args.cudnn_benchmark = True
        # args.dataset_path = None
        # args.denoising_strength = 0.01
        # args.ema = False
        # args.fade_out = 10
        # args.fastpitch = 'FastPitch_checkpoint_1000.pt'
        # args.heteronyms_path = 'cmudict/heteronyms'
        # args.input=''
        # args.log_file='nvlog_infer.json'
        # args.n_speakers = 1
        # args.output = 'gen_mels'
        # args.p_arpabet = 1.0
        # args.pace = 1.0
        # args.pitch_transform_amplify = 1.0
        # args.pitch_transform_custom = False
        # args.pitch_transform_flatten = False
        # args.pitch_transform_invert = False
        # args.pitch_transform_shift = 0.0
        # args.repeats = 1
        # args.sampling_rate = 22050
        # args.save_mels = True
        # args.sigma_infer = 0.9
        # args.speaker = 0
        # args.stft_hop_length = 256
        # args.symbol_set = 'english_basic'
        # args.text_cleaners = ['english_cleaners_v2']
        # args.torchscript = False
        # args.warmup_steps = 0
        if self.p_arpabet > 0.0:
            cmudict.initialize(self.cmudict_path, keep_ambiguous=True)

        torch.backends.cudnn.benchmark = self.cudnn_benchmark

        if self.output is not None:
            Path(self.output).mkdir(parents=False, exist_ok=True)

        log_fpath = self.log_file or str(Path(self.output, 'nvlog_infer.json'))
        log_fpath = unique_log_fpath(log_fpath)
        try:
            DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT, log_fpath),
                                    StdOutBackend(Verbosity.VERBOSE,
                                                  metric_format=stdout_metric_format)])

            init_inference_metadata()
            #[DLLogger.log("PARAMETER", {k: v}) for k, v in vars(self.args).items()]
        except Exception as e:
            print("Logger error", e)
        self.device = torch.device('cuda' if self.cuda else 'cpu')

        self.generator = MelGenerator.load_and_setup_model(parser, self.fastpitch, self.amp, self.device,
                                                               #unk_args=unk_args,
                                                               unk_args=[],
                                                                forward_is_infer=True,
                                                               ema=self.ema)
        
        #if len(unk_args) > 0:
        #    raise ValueError(f'Invalid options {unk_args}')
        if self.generator is None:
            raise Exception("Can't load generator")

    
    def parse_args(self,parser):
        """
        Parse commandline arguments.
        """
        parser.add_argument('-i', '--input', type=str,  # required=True,
                            help='Full path to the input text (phareses separated by newlines)')
        parser.add_argument('-o', '--output', default=self.output,
                            help='Output folder to save audio (file per phrase)')
        parser.add_argument('--log-file', type=str, default=self.log_file,
                            help='Path to a DLLogger log file')
        parser.add_argument('--save-mels', action='store_true', help='')
        parser.add_argument('--cuda', action='store_true',
                            help='Run inference on a GPU using CUDA')
        parser.add_argument('--cudnn-benchmark', action='store_true',
                            help='Enable cudnn benchmark mode')
        parser.add_argument('--fastpitch', type=str, default = self.fastpitch,
                            help='Full path to the generator checkpoint file (skip to use ground truth mels)')
        parser.add_argument('-sr', '--sampling-rate', default=self.sampling_rate, type=int,
                            help='Sampling rate')
        parser.add_argument('--stft-hop-length', type=int, default=self.stft_hop_length,
                            help='STFT hop length for estimating audio length from mel size')
        parser.add_argument('--amp', action='store_true',default = self.amp,
                            help='Inference with AMP')
        parser.add_argument('-bs', '--batch-size', type=int, default=self.batch_size)
        parser.add_argument('--warmup-steps', type=int, default=0,
                            help='Warmup iterations before measuring performance')
        parser.add_argument('--repeats', type=int, default=self.repeats,
                            help='Repeat inference for benchmarking')
        parser.add_argument('--ema', action='store_true',default = self.ema,
                            help='Use EMA averaged model (if saved in checkpoints)')
        parser.add_argument('--dataset-path', type=str,
                            help='Path to dataset (for loading extra data fields)')
        parser.add_argument('--speaker', type=int, default=self.speaker,
                            help='Speaker ID for a multi-speaker model')
        parser.add_argument('--p-arpabet', type=float, default=self.p_arpabet, help='')
        parser.add_argument('--heteronyms-path', type=str, default='cmudict/heteronyms',
                            help='')
        parser.add_argument('--cmudict-path', type=str, default=self.cmudict_path,
                            help='')
        transform = parser.add_argument_group('transform')
        transform.add_argument('--fade-out', type=int, default=10,
                               help='Number of fadeout frames at the end')
        transform.add_argument('--pace', type=float, default=self.pace,
                               help='Adjust the pace of speech')
        transform.add_argument('--pitch-transform-flatten', action='store_true',default = self.pitch_transform_flatten,
                               help='Flatten the pitch')
        transform.add_argument('--pitch-transform-invert', action='store_true',default = self.pitch_transform_invert,
                               help='Invert the pitch wrt mean value')
        transform.add_argument('--pitch-transform-amplify', type=float, default=self.pitch_transform_amplify,
                               help='Amplify pitch variability, typical values are in the range (1.0, 3.0).')
        transform.add_argument('--pitch-transform-shift', type=float, default=self.pitch_transform_shift,
                               help='Raise/lower the pitch by <hz>')
        transform.add_argument('--pitch-transform-custom', action='store_true',default = self.pitch_transform_custom,
                               help='Apply the transform from pitch_transform.py')

        text_processing = parser.add_argument_group('Text processing parameters')
        text_processing.add_argument('--text-cleaners', nargs='*',
                                     default=['english_cleaners_v2'], type=str,
                                     help='Type of text cleaners for input text')
        text_processing.add_argument('--symbol-set', type=str, default='english_basic',
                                     help='Define symbol set for input text')

        cond = parser.add_argument_group('conditioning on additional attributes')
        cond.add_argument('--n-speakers', type=int, default=1,
                          help='Number of speakers in the model.')

        return parser

    @staticmethod
    def load_model_from_ckpt(checkpoint_path, ema, model):
        checkpoint_data = torch.load(checkpoint_path)
        status = ''

        if 'state_dict' in checkpoint_data:
            sd = checkpoint_data['state_dict']
            if ema and 'ema_state_dict' in checkpoint_data:
                sd = checkpoint_data['ema_state_dict']
                status += ' (EMA)'
            elif ema and not 'ema_state_dict' in checkpoint_data:
                print(f'WARNING: EMA weights missing for {checkpoint_data}')

            if any(key.startswith('module.') for key in sd):
                sd = {k.replace('module.', ''): v for k, v in sd.items()}
            status += ' ' + str(model.load_state_dict(sd, strict=False))
        else:
            model = checkpoint_data['model']
        print(f'Loaded {checkpoint_path}{status}')

        return model

    @staticmethod
    def load_and_setup_model(parser, checkpoint, amp, device,
                             unk_args=[], forward_is_infer=False, ema=True):
        model_parser = models.parse_model_args(parser, add_help=False)
        model_args, model_unk_args = model_parser.parse_known_args()
        unk_args[:] = list(set(unk_args) & set(model_unk_args))

        model_config = models.get_model_config(model_args)

        model = models.get_model(model_config, device,
                                 forward_is_infer=forward_is_infer)
        if checkpoint is not None:
            model = MelGenerator.load_model_from_ckpt(checkpoint, ema, model)

        if amp:
            model.half()
        model.eval()
        return model.to(device)

    @staticmethod
    def prepare_input_sequence(fields, device, symbol_set, text_cleaners,
                               batch_size=128, p_arpabet=0.0):
        tp = TextProcessing(symbol_set, text_cleaners, p_arpabet=p_arpabet)
        fields['text'] = [torch.LongTensor(tp.encode_text(text))
                          for text in fields['text']]
        order = np.argsort([-t.size(0) for t in fields['text']])

        fields['text'] = [fields['text'][i] for i in order]
        fields['text_lens'] = torch.LongTensor([t.size(0) for t in fields['text']])

        for t in fields['text']:
            print(tp.sequence_to_text(t.numpy()))
        if 'output' in fields:
            fields['output'] = [fields['output'][i] for i in order]

        # cut into batches & pad
        batches = []
        for b in range(0, len(order), batch_size):
            batch = {f: values[b:b + batch_size] for f, values in fields.items()}
            for f in batch:
                if f == 'text':
                    batch[f] = pad_sequence(batch[f], batch_first=True)
                if type(batch[f]) is torch.Tensor:
                    batch[f] = batch[f].to(device)
            batches.append(batch)

        return batches

    def build_pitch_transformation(self):
        if self.pitch_transform_custom:
            def custom_(pitch, pitch_lens, mean, std):
                return (pitch_transform_custom(pitch * std + mean, pitch_lens)
                        - mean) / std

            return custom_

        fun = 'pitch'
        if self.pitch_transform_flatten:
            fun = f'({fun}) * 0.0'
        if self.pitch_transform_invert:
            fun = f'({fun}) * -1.0'
        if self.pitch_transform_amplify:
            ampl = self.pitch_transform_amplify
            fun = f'({fun}) * {ampl}'
        if self.pitch_transform_shift != 0.0:
            hz = self.pitch_transform_shift
            fun = f'({fun}) + {hz} / std'
        return eval(f'lambda pitch, pitch_lens, mean, std: {fun}')

    class MeasureTime(list):
        def __init__(self, *args, cuda=True, **kwargs):
            super(MelGenerator.MeasureTime, self).__init__(*args, **kwargs)
            self.cuda = cuda

        def __enter__(self):
            if self.cuda:
                torch.cuda.synchronize()
            self.t0 = time.perf_counter()

        def __exit__(self, exc_type, exc_value, exc_traceback):
            if self.cuda:
                torch.cuda.synchronize()
            self.append(time.perf_counter() - self.t0)

        def __add__(self, other):
            assert len(self) == len(other)
            return MelGenerator.MeasureTime((sum(ab) for ab in zip(self, other)), cuda=cuda)

    def get_mel(self, text: str):
        """
        Launches text to speech (inference).
        Inference is executed on a single GPU.
        """
        lines = (text,)
        columns = ('text',)
        fields = (lines,)
        fields = {c: f for c, f in zip(columns, fields)}  # load_fields()
        print(fields)
        batches = MelGenerator.prepare_input_sequence(
            fields, self.device, self.symbol_set, self.text_cleaners, self.batch_size,
            p_arpabet=self.p_arpabet)

        # Use real data rather than synthetic - FastPitch predicts len
        for _ in tqdm(range(self.warmup_steps), 'Warmup'):
            with torch.no_grad():
                if self.generator is not None:
                    b = batches[0]
                    mel, *_ = self.generator(b['text'])

        gen_measures = MelGenerator.MeasureTime(cuda=self.cuda)

        gen_kw = {'pace': self.pace,
                  'speaker': self.speaker,
                  'pitch_tgt': None,
                  'pitch_transform': self.build_pitch_transformation()}

        all_utterances = 0
        all_samples = 0
        all_letters = 0
        all_frames = 0

        reps = self.repeats
        log_enabled = reps == 1
        log = lambda s, d: DLLogger.log(step=s, data=d) if log_enabled else None

        for rep in (tqdm(range(reps), 'Inference') if reps > 1 else range(reps)):
            for b in batches:
                if self.generator is None:
                    log(rep, {'Synthesizing from ground truth mels'})
                    mel, mel_lens = b['mel'], b['mel_lens']
                else:
                    with torch.no_grad(), gen_measures:
                        mel, mel_lens, *_ = self.generator(b['text'], **gen_kw)
                    gen_infer_perf = mel.size(0) * mel.size(2) / gen_measures[-1]
                    all_letters += b['text_lens'].sum().item()
                    all_frames += mel.size(0) * mel.size(2)
                    log(rep, {"fastpitch_frames/s": gen_infer_perf})
                    log(rep, {"fastpitch_latency": gen_measures[-1]})

                    for i, mel_ in enumerate(mel):
                        m = mel_[:, :mel_lens[i].item()].permute(1, 0)
                        # fname = b['output'][i] if 'output' in b else f'mel_{i}.npy'
                        # mel_path = Path(args.output, Path(fname).stem + '.npy')
                        result = m.cpu().numpy()
                        # np.save(mel_path, m.cpu().numpy())
        log_enabled = True
        if self.generator is not None:
            gm = np.sort(np.asarray(gen_measures))
            rtf = all_samples / (all_utterances * gm.mean() * self.sampling_rate)
            log((), {"avg_fastpitch_letters/s": all_letters / gm.sum()})
            log((), {"avg_fastpitch_frames/s": all_frames / gm.sum()})
            log((), {"avg_fastpitch_latency": gm.mean()})
            log((), {"avg_fastpitch_RTF": rtf})
            log((), {"90%_fastpitch_latency": gm.mean() + norm.ppf((1.0 + 0.90) / 2) * gm.std()})
            log((), {"95%_fastpitch_latency": gm.mean() + norm.ppf((1.0 + 0.95) / 2) * gm.std()})
            log((), {"99%_fastpitch_latency": gm.mean() + norm.ppf((1.0 + 0.99) / 2) * gm.std()})
        DLLogger.flush()
        return result
