import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class SeleniumController:
    def __init__(self, url="https://minesweeper.online/"):
        options = webdriver.FirefoxOptions()
        self.driver = webdriver.Firefox(options=options)
        self.driver.get(url)
        
        print("Siteye girildi, oyunun başlaması için zorluk seviyesi seçiliyor...")
        time.sleep(2)
        
        try:
            beginner_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(@data-id, '1') or contains(@class, 'level-1')] | //div[contains(text(), 'Beginner')]/.."))
            )
            beginner_btn.click()
            print("Beginner seviyesi seçildi!")
            time.sleep(2)
        except Exception as e:
            print("Zorluk seçimi butonu bulunamadı, site zaten doğrudan oyunu açmış olabilir.")
        
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "statetable"))
            )
            self.board_id = "statetable"
        except:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            self.board_id = None
            
        time.sleep(1)

    def get_board_screenshot(self, save_path="current_board.png"):
        if self.board_id:
            game_board = self.driver.find_element(By.ID, self.board_id)
        else:
            game_board = self.driver.find_element(By.TAG_NAME, "body")
            
        game_board.screenshot(save_path)
        return save_path

    def read_board_state(self, rows=9, cols=9):
        board = []

        for y in range(rows):
            row = []
            for x in range(cols):
                try:
                    cell = self.driver.find_element(By.ID, f"cell_{x}_{y}")
                    class_name = cell.get_attribute("class") or ""
                    row.append(self._cell_class_to_state(class_name))
                except Exception:
                    return None

            board.append(row)

        return board

    def _cell_class_to_state(self, class_name):
        if "hd_flag" in class_name or "flag" in class_name:
            return "flag"
        if "hd_closed" in class_name or "closed" in class_name:
            return "unsolved"
        if "hd_type10" in class_name or "hd_type11" in class_name:
            return "mine"
        if "hd_mine" in class_name or "mine" in class_name:
            return "mine"

        number_names = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
        }
        for number, name in number_names.items():
            if f"hd_type{number}" in class_name or f"type{number}" in class_name:
                return name

        return "unsolved"

    def reset_game(self):
        try:
            reset_button = self.driver.find_element(By.ID, "top_area_face")
            reset_button.click()
        except Exception:
            self.driver.refresh()

    def click_cell(self, x, y):
        try:
            try:
                cell_id = f"cell_{x}_{y}"
                cell = self.driver.find_element(By.ID, cell_id)
            except:
                cell_id = f"{x}_{y}"
                cell = self.driver.find_element(By.ID, cell_id)
                
            cell.click()
            return True
        except Exception as e:
            print(f"Hücre tıklama hatası ({x},{y}): {e}")
            return False

    def close(self):
        self.driver.quit()
