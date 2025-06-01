from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os
from tqdm import tqdm

from crawler.utils.webdriver_manager import initialize_driver
from crawler.utils.data_saver import save_csv

SAVE_DIR = 'data/courseinfo/'

class CourseInfoScraper:
    def __init__(self, base_url):
        self.base_url = base_url

    def wait_for_loading(self, driver):
        try:
            WebDriverWait(driver, 10).until(EC.invisibility_of_element_located((By.ID, "ur-loading-box")))
        except:
            pass

    def set_year_semester(self, driver, year, semester):
        wait = WebDriverWait(driver, 10)
        year_input = wait.until(EC.presence_of_element_located((By.ID, "WD2A")))
        year_input.click()
        year_options = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div[role='option']")))
        for opt in year_options:
            if opt.text.strip() == f"{year} 학년도":
                opt.click()
                break
        self.wait_for_loading(driver)
        time.sleep(1)

        semester_input = wait.until(EC.presence_of_element_located((By.ID, "WD4F")))
        semester_input.click()
        semester_options = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div[role='option']")))
        for opt in semester_options:
            if opt.text.strip() == semester:
                opt.click()
                break
        self.wait_for_loading(driver)
        time.sleep(1)

    def select_college(self, driver):
        wait = WebDriverWait(driver, 10)
        dropdown_input = wait.until(EC.presence_of_element_located((By.ID, "WD83")))
        dropdown_input.click()
        options = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div[role='option']")))
        for opt in options:
            if opt.text.strip() == "대학":
                opt.click()
                break
        self.wait_for_loading(driver)
        time.sleep(3)

    def search_courses(self, driver):
        wait = WebDriverWait(driver, 10)
        search_button = wait.until(EC.element_to_be_clickable((By.ID, "WDB9")))
        search_button.click()
        
        # 단계적으로 로딩 기다리기
        max_attempts = 3
        for attempt in range(max_attempts):
            time.sleep(5)
            try:
                # 로딩 박스가 없어졌는지 체크
                self.wait_for_loading(driver)
                break  # 로딩 끝나면 탈출
            except:
                if attempt == max_attempts - 1:
                    raise Exception("❌ 로딩이 너무 오래 걸립니다!")
                print(f"⏳ 로딩 중... {attempt + 1}번째 추가 대기")



    def scrape_courses(self, year, semester):
        driver = initialize_driver(headless=False)
        driver.get(self.base_url)

        self.set_year_semester(driver, year, semester)
        self.select_college(driver)
        self.search_courses(driver)

        table_body = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "tbody")))
        rows = table_body.find_elements(By.CSS_SELECTOR, "tr")

        all_data = []

        for row in tqdm(rows[1:], desc=f"📊 {year} {semester} 강좌 수집 중"):
            cells = row.find_elements(By.CSS_SELECTOR, "td")
            if len(cells) == 27:
                row_data = [cell.text.strip() for cell in cells]
                all_data.append(row_data)

        print(f"✅ {year} {semester} 수집 완료: {len(all_data)}개 강좌")
        driver.quit()
        return all_data

if __name__ == "__main__":
    BASE_URL = "https://sis109.sogang.ac.kr/sap/bc/webdynpro/sap/zcmw9016?sap-language=KO#"
    targets = [
        {"year": 2025, "semester": "1학기"},
        {"year": 2024, "semester": "1학기"},
        {"year": 2024, "semester": "하계학기"},
        {"year": 2024, "semester": "2학기"},
        {"year": 2024, "semester": "동계학기"},
        {"year": 2023, "semester": "1학기"},
        {"year": 2023, "semester": "하계학기"},
        {"year": 2023, "semester": "2학기"},
        {"year": 2023, "semester": "동계학기"},
    ]

    scraper = CourseInfoScraper(BASE_URL)

    for target in targets:
        scraped_data = scraper.scrape_courses(target['year'], target['semester'])
        if scraped_data:
            df = pd.DataFrame(scraped_data[1:], columns=scraped_data[0])
            filename = f"{target['year']}_{target['semester']}.csv"
            save_path = os.path.join(SAVE_DIR, filename)
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
