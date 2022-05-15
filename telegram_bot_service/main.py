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
import admin_analytics
import numpy as np
#import torch
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd

bot = Bot(config.Token)
dp = Dispatcher(bot, storage=MemoryStorage())

MAX_TEXT_LENGTH = 250 # Maximun lenth of message for audio generation


@dp.message_handler(commands=['start'], state='*')
async def start_message(message: types.Message, state: FSMContext):
    """
    Start working with /start command
    Send first message.
    """
    await bot.send_message(message.from_user.id, text.hello_message,
                           parse_mode="Markdown")
    await states.User.Entering_text.set()



@dp.message_handler(commands=['analytics'], state='*')
async def get_analytics(message: types.Message, state: FSMContext):
    """
    Send analitics with command /analitics. (only for admins).
    """
    if message.from_user.id not in config.ADMIN_USER_IDS:
        print(message.from_user.id, ' analytics access attempt', flush=True)
        bot.send_message(message.from_user.id, f"You are not one of the admins. Please add your user id {message.from_user.id} to access this feature.")
        await states.User.Entering_text.set()
        return
    resp = requests.get('http://dispatcher_service:7000/analytics')
    if resp.status_code != 200:
        await bot.send_message(message.from_user.id, "Something was wrong... Try again")
    else:
        my_imgs, my_text = admin_analytics.parse_analytics(resp.json())
        await bot.send_message(message.from_user.id, my_text)
        for el in my_imgs:
            await bot.send_photo(message.from_user.id, el)
    await states.User.Entering_text.set()


@dp.message_handler(commands=['latency_test'], state='*')
async def latency_test(message: types.Message, state: FSMContext):
    """
    Send results of latency test. (only for admins).
    """
    if message.from_user.id not in config.ADMIN_USER_IDS:
        print(message.from_user.id, ' latency_test access attempt', flush=True)
        bot.send_message(message.from_user.id, f"You are not one of the admins. Please add your user id {message.from_user.id} to access this feature.")
        await states.User.Entering_text.set()
        return
    try:
        resp = requests.get('http://dispatcher_service:7000/', params={'text': "text text"})
        #await bot.send_message(message.from_user.id,message.text)
        if resp.status_code != 200:
            await bot.send_message(message.from_user.id, "Something was wrong... Try again")
        else:
            mes = message.text.split()
            if(len(mes)==3 and mes[1].isdecimal() and mes[2].isdecimal()):
                users_count = abs(int(mes[1]))
                time_of_test = abs(int(mes[2]))
                await bot.send_message(message.from_user.id, text.latency_test_start+"users count = "+mes[1]+" time of test = "+mes[2]+"s")
                try:
                    ans,img1 = admin_analytics.latency_test(users_count,time_of_test)
                except:
                    await bot.send_message(message.from_user.id, text.fail_test)
                    await states.User.Entering_text.set()
                    #raise
                    return
            else:
                await bot.send_message(message.from_user.id, text.latency_test_start+" defaul parametrs")
                try:
                    ans,img1 = admin_analytics.latency_test()
                except:
                    await bot.send_message(message.from_user.id, text.fail_test)
                    await states.User.Entering_text.set()
                    #raise
                    return
            await bot.send_message(message.from_user.id,ans)
            await bot.send_photo(message.from_user.id, img1)
    except:
        await bot.send_message(message.from_user.id, "Something was wrong... Try again")
        #raise
    await states.User.Entering_text.set()

@dp.message_handler(state=states.User.Entering_text)
async def enter_text_message(message: types.Message, state: FSMContext):
    """
        Generate pitch from text.
    """
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


@dp.message_handler(state="*")
async def wrong_command(message: types.Message, state: FSMContext):
    """
    Commands doesn't exist.
    """
    await bot.send_message(message.from_user.id, text.wrong_command_text,
                           parse_mode="Markdown")
    await states.User.Entering_text.set()


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)