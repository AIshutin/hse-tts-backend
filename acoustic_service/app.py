import sys

from flask import Flask
from flask import request
from flask import jsonify

import inference

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello_world():  # put application's code here
    text = request.args.get('text')
    print(text)
    sys.argv = ['inference.py', '-i', '', '-o', './gen_mels',
                '--log-file', 'nvlog_infer.json', '--save-mels', '--fastpitch'
        , 'FastPitch_checkpoint_1000.pt', '--batch-size',
                '32', '--repeats', '1', '--warmup-steps', '0', '--speaker', '0', '--n-speakers', '1', '--cuda',
                '--cudnn-benchmark', '--p-arpabet', '1.0', '--energy-conditioning']
    ans = inference.get_mel(text)
    print(ans)
    ans = jsonify(ans.tolist())


    return ans


if __name__ == '__main__':
    app.run()
