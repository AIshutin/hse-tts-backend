from aiogram.dispatcher.filters.state import State, StatesGroup


class User(StatesGroup):
    Entering_text = State()