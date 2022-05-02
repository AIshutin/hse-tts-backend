# Проверка работы сервиса
import json

import requests
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
#ans = requests.get('http://127.0.0.1:5000/?text=Hello+I+am+text+And+it+works')
text = "I live in a house near the mountains. I have two brothers and one sister, and I was born last. My father teaches mathematics, and my mother is a nurse at a big hospital. My brothers are very smart and work hard in school. My sister is a nervous girl, but she is very kind. My grandmother also lives with us. She came from Italy when I was two years old. She has grown old, but she is still very strong. She cooks the best food!"
print("text len", len(text))
ans = requests.get('http://0.0.0.0:5000/', params={'text': text})
print("request to acoustic status:", ans.status_code)
#fig, ax = plt.subplots()
#mel = np.array(ans.json()).T
#img = librosa.display.specshow(mel)
#fig.colorbar(img, ax=ax, format='%+2.0f dB')
#ax.set(title='Mel-frequency spectrogram')
#plt.show()
#np.savetxt('/home/aishutin/hse-tts-backend/acoustic_service/test1.txt',np.array(ans.json()).T , fmt='%d')
print("Try connect to vocoder")
try:
    host = "http://127.0.0.1:8000"
    resp = requests.get(host)
    #print(np.array(ans.json()).T.shape)
    x = torch.from_numpy(np.array(ans.json()).T).unsqueeze(0)
    print(x.shape)
    obj = x.tolist()
    resp = requests.post(host, json=x.tolist())
    print("request to vocoder status:", resp.status_code)
    open('out.wav', 'wb').write(resp.content)
    print("audio saved to out.wav")
except:
    print("Can't connect to vocoder")