import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from lxml import etree, html
import json
import supervision as sv
from openai import OpenAI
import torch
from PIL import Image
import io
import os 
import random
import base64
import requests
from argparse import ArgumentParser
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def check_visiblility(coord):
    cond_x = coord['left'] >= 0 and coord['right'] - coord['left'] > 0 and coord['left'] < 1920
    cond_y = coord['top'] >= 0 and coord['down'] - coord['top'] > 0 and coord['top'] < 1080
    return cond_x and cond_y


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



class VEnvElement:
    def __init__(self, coord: dict, name, tag, description, is_checkable, is_checked, is_clickable, is_editable, is_draggable, attributes, caption=None):
        self.coord = coord
        self.name = name
        self.tag = tag
        self.content_description = description
        self.is_checkable = is_checkable
        self.is_checked = is_checked
        self.is_clickable = is_clickable
        self.is_editable = is_editable
        self.is_draggable = is_draggable
        self.attributes = attributes
        self.caption = caption
        self.elements = []


    def to_dict(self):
        element = {
            'coord': self.coord,
            'tag': self.tag
        }
        if self.name:
            element['name'] = self.name
        if self.content_description:
            element['content_description'] = self.content_description
        if self.is_checkable:
            element['is_checkable'] = self.is_checkable
        if self.is_checked:
            element['is_checked'] = self.is_checked
        if self.is_clickable:
            element['is_clickable'] = self.is_clickable
        if self.is_editable:
            element['is_editable'] = self.is_editable
        if self.is_draggable:
            element['is_draggable'] = self.is_draggable
        if self.attributes:
            element['attributes'] = self.attributes
        if self.caption:
            element['caption'] = self.caption
        if self.elements:
            element['elements'] = self.elements
        return element

"""
HTML Filtering Rule:
1. go through the HTML cleaner
"""

class BaseConverter:
    def __init__(self):
        pass

    def convert_to_venv(self, input):
        raise NotImplementedError
    


class URLConverter(BaseConverter):
    TAGS_OF_INTEREST = [
        'a',
        'button',
        'input',
        'select',
        'textarea',
        'text',
        'img',
        'source',
        'div',
    ]
    VALID_ATTRS = [
        'alt',
        'class',
        'checked',
        'inputmode',
        'input_checked',
        'input_value',
        'is_clickable',
        'label',
        'name',
        'option_selected',
        'role',
        'text_value',
        'title',
        'type',
        'value',
    ]

    def __init__(self):
        super().__init__()
        
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--start-maximized')
        chrome_options.add_argument('--disable-gpu')  # Necessary for some environments
        chrome_options.add_argument('--window-size=1920x1080')  # Optional, for setting the window size
        # Default: Chrome driver
        self.driver = webdriver.Chrome(options=chrome_options)
        self.device_pixel_ratio = self.driver.execute_script('return window.devicePixelRatio')


        self.client = OpenAI(
            organization=os.environ['OPENAI_ORG_ID'],
        )


    def openai_caption(self, screenshot_path, url=None, element_info=None, temperature=0.5):
        
        base64_image = encode_image(screenshot_path)
        # Log the size of the base64 encoded image
        # print(f"Base64 image length: {len(base64_image)}")  # 667256
        # Convert element_info dictionary to JSON string
        element_info_str = json.dumps(element_info, indent=4) if element_info else "No element information provided."

        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.environ["OPENAI_API_KEY"]}',
            'OpenAI-Organization': os.environ.get("OPENAI_ORG_ID") 
        }
        
        url_example = "https://gitlab.com/dashboard/"
        element_info_example = {
            'name': 'Merge requests',
            'tag': 'a',
            'content_description': 10
        }
        element_info_example_str = json.dumps(element_info_example, indent=4)
        encoded_img_example = encode_image('system_prompts/img_caption/annotated_img_example.png')
        # print(f"Base64 image length: {len(encoded_img_example)}")  # 102692

        payload = {
            'model': 'gpt-4o',
            'messages': [
                {
                    'role': 'system',
                    'content': '''
You are an AI assistant designed to describe the function and role of web elements found within a bounding box on a screenshot. Your goal is to help users understand the purpose and potential interactions with these elements given url, element information, and encoded screenshot.
Here are some requirements for output:
    - First, think about the meaning of element in the bounding box, considering the context of the given domain (e.g., Amazon, GitHub, etc.).
    - Second, provide a brief description of the element, explaining its role or purpose on the webpage, and what I can get by interacting with the element.
    - You cannot describe anything outside the bounding box.
    - Ensure your description is concise, informative, and relevant to the context of the web page.
I will give you an example containing Example Input and Example Output. You need to follow it and response like Example Output.

Example Input:
    text: {}
    text: {}
    image_url: {}
    
Example Output:
    Thought: The given url is the GitLab dashboard, and screenshot contains one of your projects. From provided element name and icon in the bounding box, it is a "Merge request" icon, and "10" means there are 10 merge requests for your projects if you click on the icon.
    Description: The icon represents "Merge requests". I can interact with the element by clicking, and it will show me 10 merge requests for my project.
                    '''.format(url_example, element_info_example_str, encoded_img_example)
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': url
                        },
                        {
                            'type': 'text',
                            'text': element_info_str
                        },
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:image/png;base64,{base64_image}'
                            }
                        }
                    ]
                }
            ],
            'temperature': temperature
        }
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload)

        try:
            response_json = response.json()
            content = response_json['choices'][0]['message']['content']
            return content
        except requests.exceptions.JSONDecodeError:
            print("Error: Failed to decode JSON response")
            return ""

    
    def draw_annotations(self, screenshot, elements):
        annotator = sv.BoundingBoxAnnotator()
        for e in elements:
            coord = e['coord']
            # draw the bounding box
            transformer_result = {
                'boxes': torch.tensor([[coord['left'], coord['top'], coord['right'], coord['down']]]) * self.device_pixel_ratio,
                'scores': torch.tensor([0.5]),
                'labels': torch.tensor([1]),
            }
            detection = sv.Detections.from_transformers(transformer_result)
            screenshot = annotator.annotate(screenshot, detection)

        # save the annotated image
        screenshot.save(f'imgs/annotated_screenshot_{time.time()}.png')


    """input: URL string of the website"""
    def convert_to_venv(self, input, add_visual_info=True, take_annotated_screen=False):
        self.driver.get(input)
        # print("open")
        
        # self.webarena_gitlab_login()

        # get the screenshot
        screenshot = self.driver.get_screenshot_as_png()
        screenshot = Image.open(io.BytesIO(screenshot))
        # save the screenshot
        screenshot.save('web_images/screenshot.png')

        venv_elements = []
        # only keeping the tags of interest
        for tag in self.TAGS_OF_INTEREST:
            elements = self.driver.find_elements(By.TAG_NAME, tag)
            for element in elements:
                # print(element)
                if tag == 'div' and (not element.get_attribute('jsaction') or 'click' not in element.get_attribute('jsaction')):
                    continue
                rect = element.rect
                # print("get coordinates")
                if tag == 'select' and not element.is_displayed():
                    rect = self.driver.execute_script('return arguments[0].parentNode;', element).rect
                # transform into coordinates
                coord = {
                    'top': rect['y'],
                    'left': rect['x'],
                    'down': rect['y'] + rect['height'],
                    'right': rect['x'] + rect['width']
                }
                # buggy: selenium is_displayed() sometimes filters out visible elements
                if (not element.is_displayed() or not check_visiblility(coord)) and tag != 'select':
                    continue
                
                name = element.get_attribute('title') if element.get_attribute('title') else element.get_attribute('name')
                tag = element.tag_name
                # print(f"name:{name}, tag:{tag}")
                if tag != 'select':
                    description = element.text
                else:
                    parent = self.driver.execute_script('return arguments[0].parentNode;', element)
                    try:
                        description = parent.find_element(By.TAG_NAME, 'label').text
                    except:
                        description = ''
                if tag == 'input' or tag == 'select':
                    if element.get_attribute('role'):
                        tag = tag + f'_{element.get_attribute("role")}'
                    if element.get_attribute('id'):
                        tag = tag + f'_{element.get_attribute("id")}'
                    if element.get_attribute('placeholder'):
                        description = element.get_attribute('placeholder')
                    if not description:
                        description = element.get_attribute("value")
                
                is_checkable = None
                is_checked = None
                if tag == 'input':
                    is_checkable = element.get_attribute('type') == 'checkbox' or element.get_attribute('type') == 'radio'
                    is_checked = element.is_selected()
                is_clickable = None
                if tag in ['a', 'button'] or element.get_attribute('onclick') or 'input' in tag:
                    # print(tag)
                    is_clickable = True
                is_editable = None
                if tag in ['input', 'textarea'] or element.get_attribute('contenteditable'):
                    is_editable = True
                is_draggable = None
                if element.get_attribute('ondragstart') or element.get_attribute('ondrag'):
                    is_draggable = True

                # Captioning the screenshot (with bounding box)
                if add_visual_info:
                    # print("start caption")
                    # if tag != 'img' and 'button' not in tag:
                    if tag != 'a':
                        # print("not caption")
                        continue
                    
                    # Create a copy of the original screenshot for annotation
                    screenshot_copy = screenshot.copy()
                    # draw the bounding box
                    transformer_result = {
                        'boxes': torch.tensor([[coord['left'], coord['top'], coord['right'], coord['down']]]) * self.device_pixel_ratio,
                        'scores': torch.tensor([0.5]),
                        'labels': torch.tensor([1]),
                    }
                    detection = sv.Detections.from_transformers(transformer_result)

                    annotator = sv.BoundingBoxAnnotator()
                    annotated_image = annotator.annotate(screenshot_copy, detection)

                    # save the annotated image
                    annotated_image.save(f'imgs/annotated_{len(venv_elements)}.png')
                    
                    # add element information to help caption
                    element_info = element = {
                        'name': name,
                        'tag': tag,
                        'content_description': description
                    }

                    # call LMM to caption the screenshot
                    caption = self.openai_caption(f'imgs/annotated_{len(venv_elements)}.png', input, element_info, temperature=0.5)
                    try:
                        if "Description :" in caption:
                            caption = caption.split("Description :")[1]
                        elif "Description:" in caption:
                            caption = caption.split("Description:")[1]
                        else:
                            caption = caption.split("Description")[1]
                    except Exception as e:
                        print("Error: ", e)

                else:
                    caption = None

                venv_element = VEnvElement(
                    coord,
                    name,
                    tag,
                    description,
                    is_checkable,
                    is_checked,
                    is_clickable,
                    is_editable,
                    is_draggable,
                    caption
                ).to_dict()
                print("element:", venv_element)
                venv_elements.append(venv_element)

        if take_annotated_screen:
            self.draw_annotations(screenshot, venv_elements)
        self.driver.quit()
        return venv_elements


class WebArenaConverter(BaseConverter):
    TAG_NAME_MAPPING = {
        'link': 'a',
        'textbox': 'textarea',
        'StaticText': 'text',
        'heading': 'text',
        'time': 'text',
        'generic': 'div',
    }
    INVERSE_TAG_NAME_MAPPING = {
        'a': 'link',
        'textarea': 'textbox',
        'text': 'StaticText',
        'div': 'generic',
    }
    def __init__(self):
        super().__init__()

    """
    line: a line corresponding to a web element
    in the form of :
    [1] RootWebArea 'Projects · Dashboard · Gitlab' focused: True
    """
    def parse_element(self, line: str) -> VEnvElement:
        # print(line)
        line = line.strip()
        
        # find the first whitespace
        first_space = line.find(' ')
        try:
            id = int(line[1:first_space - 1])
        except:
            id = random.randint(10000, 99999)

        second_space = line.find(' ', first_space + 1)
        tag = line[first_space + 1:second_space]
        if tag in self.TAG_NAME_MAPPING:
            tag = self.TAG_NAME_MAPPING[tag]
        
        # find the first quote
        first_quote = line.find('\'')
        # find the second quote
        second_quote = line.find('\'', first_quote + 1)
        content_description = line[first_quote + 1:second_quote]

        substr = line[second_quote + 2:]
        attributes = substr if substr else None

        is_checkable = None
        is_checked = None
        is_clickable = None
        if tag in ['a', 'button']:
            is_clickable = True
        is_editable = None
        if tag in ['input', 'textarea']:
            is_editable = True
        is_draggable = None

        if tag != 'img':
            return VEnvElement(
                coord={'id': id},
                name=None,
                tag=tag,
                description=content_description,
                is_checkable=is_checkable,
                is_checked=is_checked,
                is_clickable=is_clickable,
                is_editable=is_editable,
                is_draggable=is_draggable,
                attributes=attributes
            )
        else:
            return VEnvElement(
                coord={'id': id},
                name=None,
                tag=tag,
                description=None,
                caption=content_description,
                is_checkable=is_checkable,
                is_checked=is_checked,
                is_clickable=is_clickable,
                is_editable=is_editable,
                is_draggable=is_draggable,
                attributes=attributes
            )


    def webarena_gitlab_login(self):
        # login webarena gitlab(example 44)
        try:
            # elements_with_ids = self.driver.find_elements("xpath", '//*[@id]')
            # for element in elements_with_ids:
            #     print(f"Tag: {element.tag_name}, ID: {element.get_attribute('id')}")
            
            username_field = self.driver.find_element(By.ID, "user_login")
            password_field = self.driver.find_element(By.ID, "user_password")
            print("successfully find elements")
            username_field.send_keys("byteblaze")
            password_field.send_keys("hello1234")
            print("successfully send keys")

            sign_in_button = self.driver.find_element(By.XPATH, "//button[text()='Sign in']")
            sign_in_button.click()
            
            # Optionally, wait for the next page to load or confirm login success
            WebDriverWait(self.driver, 10).until(
                EC.url_changes(input)
            )
            print("Login successful.")

        except Exception as e:
            print(f"An error occurred during login: {e}")

        return

    """input: WebArena accessibility tree of the website"""
    def convert_to_venv(self, input: str):
        lines = input.split('\n')
        venv_elements = []
        for line in lines:
            if line:
                venv_element = self.parse_element(line)
                # print("finish")
                venv_elements.append(venv_element.to_dict())
        return venv_elements
    
    def unify_state_format(self, state: list):
        for i, line in enumerate(state):
            line = line.replace('\t', '    ')
            
            if '[' not in line:
                # add an element id to it
                # computer the number of indent in the prefix
                cnt = 0
                while line[cnt] == ' ':
                    cnt += 1
                id = 10000 + random.randint(0, 9999)
                state[i] = cnt * ' ' + f'[{id}] ' + line.strip()
            else:
                state[i] = line

        return state


    def convert_to_tree_venv(self, input):
        if isinstance(input, str):
            input = input.split('\n')

            if 'Tab' in input[0]:
                input = input[2:]
        
        cnt = 0
        root = self.parse_element(input[0]).to_dict()
        cnt += 1

        # get the number of spaces in the prefix of line for one level
        i = input[cnt].find('[')
        NUM_SPACES_PER_LEVEL = i
        MAX_LEN = len(input)

        def dfs(root, input_list, current_depth):
            nonlocal cnt
            
            while cnt < MAX_LEN:
                # find the number of spaces in the prefix of line
                line = input_list[cnt]
                num_spaces = line.find('[')
                if num_spaces // NUM_SPACES_PER_LEVEL <= current_depth:
                    # should return to the upper level
                    return root
                else:
                    venv_element = self.parse_element(line).to_dict()
                    cnt += 1
                    if 'elements' not in root:
                        root['elements'] = []
                    root['elements'].append(dfs(venv_element, input_list, current_depth + 1))
            return root
        
        return dfs(root, input, 0)
    
    def convert_flat_tree_venv_to_real_env(self, input: list[dict], depth_dict: dict):
        indent = ''
        out = ''
        for element in input:
            depth = depth_dict[element['coord']['id']]
            indent = ' ' * depth * 4
            if element['tag'] in self.INVERSE_TAG_NAME_MAPPING:
                tag = self.INVERSE_TAG_NAME_MAPPING[element['tag']]
            else:
                tag = element['tag']
            out += f"{indent}[{element['coord']['id']}] {tag} "
            if 'name' in element:
                out += f"{element['name']} "
            if 'content_description' in element:
                out += f"'{element['content_description']}' "
            if 'caption' in element:
                out += f"'{element['caption']}' "
            if 'attributes' in element:
                out += f" {element['attributes']}"
            out += '\n'

        return out
    
    def convert_tree_venv_to_real_env(self, input: dict):
        indent = ''
        out = ''
        # first order traversal
        def dfs(root, depth):
            nonlocal out
            indent = ' ' * depth * 4
            if root['tag'] in self.INVERSE_TAG_NAME_MAPPING:
                tag = self.INVERSE_TAG_NAME_MAPPING[root['tag']]
            else:
                tag = root['tag']
            out += f"{indent}[{root['coord']['id']}] {tag} "
            if 'name' in root:
                out += f"{root['name']} "
            if 'content_description' in root:
                out += f"'{root['content_description']}' "
            if 'caption' in root:
                out += f"'{root['caption']}' "
            if 'attributes' in root:
                out += f" {root['attributes']}"
            out += '\n'
            if 'elements' in root:
                for element in root['elements']:
                    dfs(element, depth + 1)

        dfs(input, 0)
        return out
    
    def convert_venv_to_real_env(self, input: list[dict]):
        out = ''
        for element in input:
            keys = set(element.keys())
            out += f"[{element['coord']['id']}] {element['tag']} "
            keys.remove('coord')
            keys.remove('tag')
            if 'name' in element:
                out += f"{element['name']} "
                keys.remove('name')
            if 'content_description' in element:
                out += f"'{element['content_description']}' "
                keys.remove('content_description')
            if 'caption' in element:
                out += f"'{element['caption']}' "
                keys.remove('caption')
            if 'attributes' in element:
                out += f" {element['attributes']}"
                keys.remove('attributes')
            for key in keys:
                out += f" {key}: {element[key]}"
            out += '\n'

        return out
    
    def convert_all_to_tree_venv(self, domain: str, converter):
        l = [f for f in os.listdir(f'init_states/webarena/{domain}') if f.endswith('.txt')]        
        for p in l:
            path = f'init_states/webarena/{domain}/{p}'
            with open(path, 'r') as f:
                content = f.read()
            try:
                venv_state = converter.convert_to_tree_venv(content.split('\n'))
                # save to json file
                with open(f'init_states/webarena/{domain}/{os.path.splitext(p)[0]}.json', 'w') as f:
                    json.dump(venv_state, f)
            except Exception as e:
                print("Error:", e)
                print(path)
        
