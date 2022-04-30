# import asyncio
import aiogram
from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
#from aiogram.methods.send_audio import SendAudio
import config
import states
import text
import requests
import numpy as np
import torch
import base64
from io import BytesIO

bot = Bot(config.Token)
dp = Dispatcher(bot, storage=MemoryStorage())

MAX_TEXT_LENGTH = 800

# Начало работы, предложение ввести текст
@dp.message_handler(commands=['start'], state='*')
async def start_message(message: types.Message, state: FSMContext):
    await bot.send_message(message.from_user.id, text.hello_message,
                           parse_mode="Markdown")
    await states.User.Entering_text.set()

def parse_analytics(data, prefix=""):
    IMG_THR = 100
    imgs = []
    text = []
    for el in data:
        if isinstance(data[el], str):
            if len(data[el]) > IMG_THR:
                imgs.append(BytesIO(base64.b64decode(data[el])))
                continue
        if isinstance(data[el], dict):
            text.append(prefix + el + ":\n")
            sub_imgs, sub_text = parse_analytics(data[el], prefix=prefix + "\t")
            imgs.extend(sub_imgs)
            text.extend(sub_text)
        else:
            text.append(prefix + el + ":\t" + str(data[el]) + "\n")
    return imgs, ''.join(text)

@dp.message_handler(commands=['analytics'], state='*')
async def start_message(message: types.Message, state: FSMContext):
    if message.from_user.id not in config.ADMIN_USER_IDS:
        print(message.from_user.id, ' analytics access attempt', flush=True)
        bot.send_message(message.from_user.id, f"You are not one of the admins. Please add your user id {message.from_user.id} to access this feature.")
        await states.User.Entering_text.set()
        return
    resp = requests.get('http://dispatcher_service:7000/analytics')
    if resp.status_code != 200:
        await bot.send_message(message.from_user.id, "Something was wrong... Try again")
    else:
        my_imgs, my_text = parse_analytics(resp.json())
        await bot.send_message(message.from_user.id, my_text)
        for el in my_imgs:
            await bot.send_photo(message.from_user.id, el)
    await states.User.Entering_text.set()

# Перевод текста в аудио
@dp.message_handler(state=states.User.Entering_text)
async def enter_text_message(message: types.Message, state: FSMContext):
    await bot.send_message(message.from_user.id, "processing..",
                            parse_mode="Markdown")
    if len(message.text) > MAX_TEXT_LENGTH:
        await bot.send_message(message.from_user.id, f"The text is too long, maximum supported length is {MAX_TEXT_LENGTH}")
        await states.User.Entering_text.set()
        return
    try:
        #resp = requests.get('http://0.0.0.0:7000/', params={'text': message.text})
        resp = requests.get('http://dispatcher_service:7000/', params={'text': message.text})
        if resp.status_code == 200:
            await bot.send_audio(message.from_user.id, resp.content)  # open("LJ001-0001.wav",'rb'))
        else:
            print(resp)
            await bot.send_message(message.from_user.id, "Something was wrong... Try again",
                                   parse_mode="Markdown")
    except Exception as e:
        print(e)
        await bot.send_message(message.from_user.id, "Something was wrong... Try again",
                               parse_mode="Markdown")

    await states.User.Entering_text.set()


# Команда не существует
@dp.message_handler(state="*")
async def wrong_command(message: types.Message, state: FSMContext):
    await bot.send_message(message.from_user.id, text.wrong_command_text,
                           parse_mode="Markdown")
    await states.User.Entering_text.set()


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)