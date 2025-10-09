from actions.action import *
from actions.webarena_action import *

AVAILABLE_ACTIONS = {
    'click': Click.INTRO,
    'input_text': Type.INTRO,
    'open_app': OpenApp.INTRO,
    'keyborad_enter': KeyboradEnter.INTRO,
    'scroll': Scroll.INTRO,
    'navigate_back': Navigate_Back.INTRO,
    'navigate_home': Navigate_Home.INTRO,
    'wait': Wait.INTRO,
}

AVAILABLE_WEBARENA_ACTIONS = {
    'click': WA_Click.INTRO,
    'type': WA_Type.INTRO,
    'hover': WA_Hover.INTRO,
    # 'press': WA_Press.INTRO,
    # 'scroll': WA_Scroll.INTRO,
    'go_forward': WA_Navigate_Forward.INTRO,
    'go_back': WA_Navigate_Back.INTRO,
    'new_tab': WA_New_Tab.INTRO,
    'tab_focus': WA_Switch_Tab.INTRO,
    'close_tab': WA_Close_Tab.INTRO,
    'goto': WA_Go_To.INTRO
}

ACTIONS_NAME = {
    'click': Click,
    'input_text': Type,
    'open_app': OpenApp,
    'keyborad_enter': KeyboradEnter,
    'scroll': Scroll,
    'navigate_back': Navigate_Back,
    'navigate_home': Navigate_Home,
    'wait': Wait,
}

WEBARENA_ACTIONS_NAME = {
    'click': WA_Click,
    'type': WA_Type,
    'hover': WA_Hover,
    'press': WA_Press,
    'scroll': WA_Scroll,
    'go_forward': WA_Navigate_Forward,
    'go_back': WA_Navigate_Back,
    'new_tab': WA_New_Tab,
    'tab_focus': WA_Switch_Tab,
    'close_tab': WA_Close_Tab,
    'goto': WA_Go_To
}