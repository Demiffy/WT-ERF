import os, time, pyautogui, pytesseract, re
import utils

UI_MAP_VIEW_REGION = (255, 725, 100, 20) # left, top, width, height
UI_STATISTICS_REGION = (65, 75, 110, 40)
UI_MAINMENU_REGION = (300, 878, 1020, 20)
UI_INGAME_REGION = (15, 995, 105, 70)

main_menu_keywords = ["usa", "germany", "ussr", "great britain", "japan", "china", "italy", "france", "sweden", "israel"]

STATE_DIR = os.path.join("static", "screenshots", "state")
os.makedirs(STATE_DIR, exist_ok=True)
STATE_IMAGE_PATH = os.path.join(STATE_DIR, "latest_state.png")
latest_state_snapshot = "/static/screenshots/state/latest_state.png"
STATE_EXTRA_IMAGE_PATH = os.path.join(STATE_DIR, "latest_state_extra.png")
latest_state_extra_snapshot = "/static/screenshots/state/latest_state_extra.png"
STATE_MAINMENU_IMAGE_PATH = os.path.join(STATE_DIR, "latest_state_mainmenu.png")
latest_state_mainmenu_snapshot = "/static/screenshots/state/latest_state_mainmenu.png"
STATE_INGAME_IMAGE_PATH = os.path.join(STATE_DIR, "latest_state_ingame.png")
latest_state_ingame_snapshot = "/static/screenshots/state/latest_state_ingame.png"

ui_state_text = ""
ui_state = ""

def ocr_state_loop():
    global ui_state_text, ui_state
    while True:
        if not utils.is_aces_in_focus():
            time.sleep(1)
            continue

        try:
            screenshot1 = pyautogui.screenshot(region=UI_MAP_VIEW_REGION)
            screenshot1.save(STATE_IMAGE_PATH)
            
            screenshot2 = pyautogui.screenshot(region=UI_STATISTICS_REGION)
            screenshot2.save(STATE_EXTRA_IMAGE_PATH)
            
            mainmenu_screenshot = pyautogui.screenshot(region=UI_MAINMENU_REGION)
            mainmenu_screenshot.save(STATE_MAINMENU_IMAGE_PATH)
            
            ingame_screenshot = pyautogui.screenshot(region=UI_INGAME_REGION)
            ingame_screenshot.save(STATE_INGAME_IMAGE_PATH)
            
            raw_text1 = pytesseract.image_to_string(screenshot1)
            raw_text2 = pytesseract.image_to_string(screenshot2)
            main_menu_text = pytesseract.image_to_string(mainmenu_screenshot, lang="eng").strip()
            raw_text_ingame = pytesseract.image_to_string(ingame_screenshot)
            
            ui_state_text = f"{raw_text1.strip()} {raw_text2.strip()} {main_menu_text} {raw_text_ingame.strip()}"
            
            lower_text1 = raw_text1.lower()
            lower_text2 = raw_text2.lower()
            lower_mainmenu = main_menu_text.lower()
            lower_ingame = raw_text_ingame.lower()
            
            if any(keyword in lower_ingame for keyword in ["gear", "rpm", "spd", "km"]):
                ui_state = "ingame"
            elif any(keyword in lower_mainmenu for keyword in main_menu_keywords):
                ui_state = "main_menu"
            elif ("time left" in lower_text2 or "time" in lower_text2 or 
                  "left" in lower_text2 or "conditions" in lower_text2):
                ui_state = "statistics_view"
            elif "chat" in lower_text1 or "battle" in lower_text1 or "chat" in lower_text2 or "battle" in lower_text2:
                ui_state = "map_view"
            else:
                ui_state = "unknown"
            
            utils.log(f"UI State OCR: {ui_state} | Raw: {ui_state_text}", level="INFO", tag="OCR")
        except Exception as e:
            utils.log(f"UI State OCR Error: {e}", level="ERROR", tag="OCR")
        time.sleep(2)