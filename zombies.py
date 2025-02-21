#!/usr/bin/env python3
import cv2
import numpy as np
import pyautogui
import time
import random
from mss import mss

# --- Global Game State ---
ammo = 10
max_ammo = 10
health = 100
kills = 0

# --- Adaptive Settings ---
learn_params = {
    'reaction_time': 0.06,
    'detection_threshold': 0.85,
    'aim_adjustment': 50,
    'headshot_priority': True
}

# --- Load Templates ---
zombie_template = cv2.imread('templates/zombie.png', 0)
headshot_template = cv2.imread('templates/head.png', 0)

def capture_screen():
    with mss() as sct:
        sct_img = sct.grab(sct.monitors[0])
        img = np.array(sct_img)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def match_template(screen_img, template):
    if template is None:
        return []
    result = cv2.matchTemplate(screen_img, template, cv2.TM_CCOEFF_NORMED)
    return list(zip(*np.where(result >= learn_params['detection_threshold'])[::-1]))

def detect_zombies(screen_img):
    zombies = match_template(screen_img, zombie_template)
    headshots = match_template(screen_img, headshot_template)
    return {'zombies': zombies, 'headshots': headshots}

def decide_action(detections):
    global ammo, kills
    actions = []
    if detections['zombies']:
        target = detections['headshots'][0] if detections['headshots'] and learn_params['headshot_priority'] else detections['zombies'][0]
        if ammo > 0:
            actions.append(('aim', target))
            actions.append(('shoot', target))
            ammo -= 1
            kills += 1
        else:
            actions.append(('reload', None))
        actions.append(('move_left', None) if random.random() > 0.5 else ('move_right', None))
    else:
        actions.append(('look_around', None))
        actions.append((random.choice(['move_forward', 'move_backward']), None))
    if ammo < 3:
        actions.append(('reload', None))
    return actions

def perform_action(action):
    action_type, pos = action
    rt = learn_params['reaction_time']
    if action_type == 'shoot':
        pyautogui.click()
    elif action_type == 'aim':
        pyautogui.moveTo(pos[0], pos[1], duration=rt)
    elif action_type == 'reload':
        pyautogui.press('r')
        global ammo
        ammo = max_ammo
    elif action_type == 'look_around':
        pyautogui.moveRel(random.randint(-100, 100), random.randint(-100, 100), duration=rt)
    else:
        pyautogui.keyDown(action_type[5:])
        time.sleep(rt)
        pyautogui.keyUp(action_type[5:])

def main():
    print("Starting Zombie Bot...")
    while True:
        screen_img = capture_screen()
        detections = detect_zombies(screen_img)
        actions = decide_action(detections)
        for action in actions:
            perform_action(action)
        print(f"Ammo: {ammo}, Health: {health}, Kills: {kills}, Actions: {actions}")
        time.sleep(learn_params['reaction_time'])

if __name__ == "__main__":
    main()






