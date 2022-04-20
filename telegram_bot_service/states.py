from aiogram.dispatcher.filters.state import State, StatesGroup


class User(StatesGroup):
    Started_chat = State()
    Entering_text = State()