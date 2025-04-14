import requests
import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from io import BytesIO
import PyPDF2


class PDFExtractor:
    @staticmethod
    def extract_text_from_pdf(pdf_url):
        try:
            response = requests.get(pdf_url)
            pdf_file = BytesIO(response.content)

            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text.strip()
        except Exception as e:
            print(f"⚠️ PDF 텍스트 추출 실패: {e}")
            return ""


class PostDetailExtractor:
    def __init__(self, driver, base_url):
        self.driver = driver
        self.base_url = base_url

    def extract_post_detail_without_pdf(self):
        soup = BeautifulSoup(self.driver.page_source, "html.parser")

        title = self._extract_title(soup)
        author, date, views = self._extract_meta_info(soup)
        content = self._extract_content(soup)

        return {
            "title": title,
            "author": author,
            "date": date,
            "views": views,
            "content": content,
            "url": self.driver.current_url
        }

    def extract_post_detail_with_pdf(self):
        soup = BeautifulSoup(self.driver.page_source, "html.parser")

        title = self._extract_title(soup)
        author, date, views = self._extract_meta_info(soup)
        content = self._extract_content(soup)

        pdf_link = self._extract_pdf_link(soup)
        pdf_text = ""
        if pdf_link:
            print(f"📄 PDF 다운로드 중: {pdf_link}")
            pdf_text = PDFExtractor.extract_text_from_pdf(pdf_link)

        content += "\n\n[PDF 내용]\n" + pdf_text

        return {
            "title": title,
            "author": author,
            "date": date,
            "views": views,
            "content": content,
            "url": self.driver.current_url,
            "pdf_url": pdf_link if pdf_link else None
        }

    def _extract_title(self, soup):
        title_tag = soup.select_one("div#set-alt-image")
        return title_tag.text.strip() if title_tag else "제목 없음"

    def _extract_meta_info(self, soup):
        author, date, views = "", "", ""
        try:
            meta_box = soup.select_one("div.flex.justify-between.md\\:mt-6")
            if meta_box:
                text_blocks = meta_box.find_all("div", recursive=True)
                for i in range(len(text_blocks)):
                    label = text_blocks[i].text.strip()
                    if label == "작성자":
                        author = text_blocks[i + 2].text.strip()
                    elif label == "작성일":
                        date = text_blocks[i + 2].text.strip()
                    elif label == "조회수":
                        views = text_blocks[i + 2].text.strip()
        except Exception as e:
            print("⚠️ 메타 정보 파싱 실패:", e)
        return author, date, views

    def _extract_content(self, soup):
        content_container = soup.select_one("div.break-words.custom-css-tag-a.tiptap")
        return "\n".join([p.text.strip() for p in content_container.select("p.zoom-text")]) if content_container else ""

    def _extract_pdf_link(self, soup):
        pdf_tag = soup.select_one("a[href$='.pdf']")
        if pdf_tag:
            pdf_link = pdf_tag['href']
            if not pdf_link.startswith("http"):
                pdf_link = self.base_url + pdf_link  # 상대 경로를 절대 경로로 변경
            return pdf_link
        return None


class NoticeScraper:
    def __init__(self, driver, base_url, start_page=1, end_page=1):
        self.driver = driver
        self.base_url = base_url
        self.start_page = start_page
        self.end_page = end_page
        self.post_detail_extractor = PostDetailExtractor(driver, base_url)

    def scrape_notice_pages(self):
        all_data = []

        for page_num in range(self.start_page, self.end_page + 1):
            print(f"\n📄 페이지 {page_num} 접속 중...")
            self.driver.get(f"{self.base_url}?page={page_num}")
            time.sleep(3)

            posts = self.driver.find_elements(By.CSS_SELECTOR, "div.cursor-pointer.border-b-\\[1px\\]")

            for i in range(len(posts)):
                try:
                    post_elements = self.driver.find_elements(By.CSS_SELECTOR, "div.cursor-pointer.border-b-\\[1px\\]")
                    post = post_elements[i]
                    preview_text = post.text.split("\n")[0].strip()
                    print(f"  - 게시글 클릭 중: {preview_text}")

                    self.driver.execute_script("arguments[0].click();", post)
                    time.sleep(1)

                    if self.driver.current_url.endswith('.pdf'):
                        data = self.post_detail_extractor.extract_post_detail_with_pdf()
                    else:
                        data = self.post_detail_extractor.extract_post_detail_without_pdf()

                    all_data.append(data)

                    self.driver.back()
                    time.sleep(1)

                except Exception as e:
                    print(f"    ❌ 게시글 크롤링 실패: {e}")
                    continue

        return all_data


class WebDriverManager:
    @staticmethod
    def initialize_driver():
        options = webdriver.ChromeOptions()
        # options.add_argument("--headless")  # 창 안 띄우는 옵션
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        return driver


if __name__ == "__main__":
    driver = WebDriverManager.initialize_driver()
    BASE_URL = "https://sogang.ac.kr/ko/academic-support/notices"

    scraper = NoticeScraper(driver, BASE_URL, start_page=1, end_page=1)
    scraped = scraper.scrape_notice_pages()

    with open("data/raw/aca-support_test.json", "w", encoding="utf-8") as f:
        json.dump(scraped, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 총 {len(scraped)}개의 게시글 저장 완료!")
    driver.quit()
