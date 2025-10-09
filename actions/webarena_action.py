"""Temporary action classes"""
from typing import Literal
from actions.action import BaseAction
    
    
class WA_Click(BaseAction):
    INTRO = "click [id]: use mouse to click on element with id on current Web."
    def __init__(self, id):
        self.id = id
    def __str__(self) -> str:
        return "click [{}]".format(self.id)
    
class WA_Type(BaseAction):
    INTRO = "type [id] [content] [0|1]: use keyboard to write \"content\" into a text box with id, and press enter if the third argument is 1"
    def __init__(self, id, text, press_enter_after=1):
        self.id = id
        self.text = text
        self.press_enter_after = press_enter_after
    def __str__(self) -> str:
        return "type [{}] [{}] [{}]".format(self.id, self.text, self.press_enter_after)

class WA_Hover(BaseAction):
    INTRO = "hover [id]: use mouse to hover on element with id on current Web. Usually used to trigger a popup/dropdown menu."
    def __init__(self, id):
        self.id = id
    def __str__(self) -> str:
        return "hover [{}]".format(self.id)
    
class WA_Press(BaseAction):
    INTRO = "press [key_comb]: press a specific keyboard combination, for example, press [ctrl+c]."
    def __init__(self, key):
        self.key = key
    def __str__(self) -> str:
        return "press [{}]".format(self.key)
    
class WA_Scroll(BaseAction):
    INTRO = "scroll [direction=down|up|left|right]: move current Web in a specific direction. You can choose from scroll [up], scroll [down], scroll [left], scroll [right]."
    def __init__(self, direction: Literal['up', 'down', 'left', 'right']):
        self.direction = direction
    def __str__(self) -> str:
        return "scroll [{}]".format(self.direction)

class WA_Navigate_Forward(BaseAction):
    INTRO = "go_forward: go forward to the next Web page, if a previous `go_back` action was executed."
    def __str__(self) -> str:
        return "go_forward"
    
class WA_Navigate_Back(BaseAction):
    INTRO = "go_back: go back to last Web page"
    def __str__(self) -> str:
        return "go_back"
    

class WA_New_Tab(BaseAction):
    INTRO = "new_tab: open a new empty browser tab"
    def __str__(self) -> str:
        return "new_tab"
    
    
class WA_Switch_Tab(BaseAction):
    INTRO = "tab_focus [index]: Switch the browser's focus to a specific tab using its index."
    def __init__(self, index):
        self.index = index
    def __str__(self) -> str:
        return "tab_focus [{}]".format(self.index)
    
class WA_Close_Tab(BaseAction):
    INTRO = "close_tab: close the current tab"
    def __str__(self) -> str:
        return "close_tab"
    
class WA_Go_To(BaseAction):
    INTRO = "goto [url]: go to a specific URL"
    def __init__(self, url):
        self.url = url
    def __str__(self) -> str:
        return "goto [{}]".format(self.url)
