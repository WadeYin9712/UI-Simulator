"""Temporary action classes"""
from typing import Literal


class BaseAction:
    def __init__(self):
        pass
    def __str__(self) -> str:
        raise NotImplementedError
    def execute(self):
        raise NotImplementedError
    
# status, answer, click, 
    
class Click(BaseAction):
    INTRO = "click [id]: click on element with index id on current page."
    def __init__(self, id):
        self.id = id
    def __str__(self) -> str:
        return "click [{}]".format(self.id)
    
class OpenApp(BaseAction):
    INTRO = "open_app [app_name]: open app with name app_name."
    def __init__(self, app):
        self.app = app
    def __str__(self) -> str:
        return "open_app [{}]".format(self.app)
    
    
class Type(BaseAction):
    INTRO = "input_text [id] [content]: use keyboard to write \"content\" into a text box with index id."
    def __init__(self, id, text):
        self.id = id
        self.text = text
    def __str__(self) -> str:
        return "input_text [{}] [{}]".format(self.id, self.text)
    
class KeyboradEnter(BaseAction):
    INTRO = "keyborad_enter: press enter key"
    def __str__(self) -> str:
        return "keyborad_enter"

class Scroll(BaseAction):
    INTRO = "scroll[direction]: move current UI in a specific direction. You can choose from scroll('up'), scroll('down'), scroll('left'), scroll('right')."
    def __init__(self, direction: Literal['up', 'down', 'left', 'right']):
        self.direction = direction
    def __str__(self) -> str:
        return "scroll [{}]".format(self.direction)
    
class Navigate_Back(BaseAction):
    INTRO = "navigate_back: go back to last UI page"
    def __str__(self) -> str:
        return "navigate_back"
    
class Navigate_Home(BaseAction):
    INTRO = "navigate_home: go to the root UI page"
    def __str__(self) -> str:
        return "navigate_home"
    
    
class Wait(BaseAction):
    INTRO = "wait: do nothing during a specific time length"
    def __str__(self) -> str:
        return "wait"
