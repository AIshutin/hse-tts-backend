import argparse
import wave
import contextlib
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('wav', type=Path, help='wav file')
args = parser.parse_args()
fname = str(args.wav)
with contextlib.closing(wave.open(fname,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)
    print(duration)