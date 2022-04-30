from flask import Flask
from flask import jsonify
from flask import request

from inference import MelGenerator

app = Flask(__name__)
# True if need to load model from checkpoint
mel_generator = MelGenerator()


@app.route('/', methods=['GET', 'POST'])
def get_mel():  # put application's code here
    text = request.args.get('text')
    print(text)
    ans = mel_generator.get_mel(text)
    print(ans)
    ans = jsonify(ans.tolist())

    return ans


if __name__ == '__main__':
    app.run()
