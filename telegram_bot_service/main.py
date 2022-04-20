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
bot = Bot(config.Token)
dp = Dispatcher(bot, storage=MemoryStorage())


# Начало работы, предложение ввести текст
@dp.message_handler(commands=['start'], state='*')
async def start_message(message: types.Message, state: FSMContext):
    await bot.send_message(message.from_user.id, text.hello_message,
                           parse_mode="Markdown")
    await states.User.Entering_text.set()


# Перевод текста в аудио
@dp.message_handler(state=states.User.Entering_text)
async def enter_text_message(message: types.Message, state: FSMContext):
    await bot.send_message(message.from_user.id, "processing..",
                            parse_mode="Markdown")
    resp = requests.get('http://127.0.0.1:7000/', params={'text': message.text})
    if resp.status_code == 200:
        await bot.send_audio(message.from_user.id,resp.content) #open("LJ001-0001.wav",'rb'))
    else:
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