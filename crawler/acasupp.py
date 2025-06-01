import requests
import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from io import BytesIO
import PyPDF2


# PDF 텍스트 추출 클래스
class PDFExtractor:
    @staticmethod
    def extract_text_from_pdf(pdf_url):
        try:
            response = requests.get(pdf_url, verify=False)
            pdf_file = BytesIO(response.content)

            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text.strip()
        except Exception as e:
            print(f"⚠️ PDF 텍스트 추출 실패: {e}")
            return ""


# 게시글 세부 내용 추출 클래스
class PostDetailExtractor:
    def __init__(self, driver, base_url):
        self.driver = driver
        self.base_url = base_url

    # 게시글에서 PDF 링크를 추출
    def extract_pdf_link(self, soup):
        # PDF 링크가 iframe 내에 포함되어 있을 수 있음
        iframe_tag = soup.select_one("iframe[src*='viewer.html?file']")
        if iframe_tag:
            iframe_src = iframe_tag.get('src')
            pdf_link = iframe_src.split("file=")[-1]
            pdf_link = requests.utils.unquote(pdf_link)  # URL 디코딩
            if not pdf_link.startswith("http"):
                pdf_link = self.base_url + pdf_link
            return pdf_link
        
        # 일반적으로 a 태그에서 .pdf 파일 링크 추출
        pdf_tag = soup.select_one("a[href$='.pdf']")
        if pdf_tag:
            pdf_link = pdf_tag['href']
            if not pdf_link.startswith("http"):
                pdf_link = self.base_url + pdf_link  # 상대 경로를 절대 경로로 변경
            return pdf_link
        return None

    # 게시글에서 이미지 URL을 추출 (특정 img 태그만 추출)
    def extract_image_urls(self, soup):
        img_urls = []
        img_tags = soup.select("p img")  # 원하는 <img> 태그만 선택
        
        for img in img_tags:
            img_url = img.get("src")
            
            # .svg 이미지는 제외
            if img_url and not img_url.endswith('.svg'):
                # 상대 경로일 경우 절대 경로로 변경
                if not img_url.startswith("http"):
                    img_url = self.base_url + img_url
                img_urls.append(img_url)
        
        return img_urls


    # 게시글 내용 추출 (PDF 있는 경우, 없는 경우 모두 처리)
    def extract_post_detail(self):
        soup = BeautifulSoup(self.driver.page_source, "html.parser")

        title = self._extract_title(soup)
        author, date, views = self._extract_meta_info(soup)
        content = self._extract_content(soup)

        # PDF 링크 추출
        pdf_link = self.extract_pdf_link(soup)
        pdf_text = ""
        if pdf_link:
            print(f"📄 PDF 다운로드 중: {pdf_link}")
            pdf_text = PDFExtractor.extract_text_from_pdf(pdf_link)
            content += "\n\n[PDF 내용]\n" + pdf_text

        # 이미지 URL 추출
        image_urls = self.extract_image_urls(soup)

        return {
            "title": title,
            "author": author,
            "date": date,
            "views": views,
            "content": content,
            "url": self.driver.current_url,
            "pdf_url": pdf_link if pdf_link else None,
            "image_urls": image_urls  # 이미지 URL 리스트 추가
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


# 웹 드라이버 관리 클래스
class WebDriverManager:
    @staticmethod
    def initialize_driver():
        options = webdriver.ChromeOptions()
        #options.add_argument("--headless")  # 창 안 띄우는 옵션
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        return driver

# 메인 크롤러 클래스
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

                    # 게시글 세부 정보 추출, 최대 3번 시도
                    retry_count = 0
                    data = None
                    while retry_count < 3:
                        try:
                            data = self.post_detail_extractor.extract_post_detail()

                            # 디버깅용으로 텍스트 일부 출력
                            print(f"📄 제목: {data['title']}")
                            print(f"📄 내용 일부: {data['content'][:200]}...")  # 본문 내용의 앞 200자 출력
                            
                            # 제목과 본문 내용이 있으면 바로 저장
                            if data['title'] != "제목 없음" and (data['content'] or data['image_urls']):
                                break  # 본문이나 이미지가 있다면 멈추고 저장

                        except Exception as e:
                            print(f"    ❌ 게시글 크롤링 실패: {e}")
                            retry_count += 1
                            if retry_count >= 3:
                                print("⚠️ 최대 시도 횟수 초과, 다음 게시글로 넘어갑니다.")
                            time.sleep(2)

                    if data:
                        all_data.append(data)
                    self.driver.back()
                    time.sleep(1)

                except Exception as e:
                    print(f"    ❌ 게시글 크롤링 실패: {e}")
                    continue

        return all_data


if __name__ == "__main__":
    driver = WebDriverManager.initialize_driver()
    BASE_URL = "https://sogang.ac.kr/ko/academic-support/notices"

    scraper = NoticeScraper(driver, BASE_URL, start_page=1, end_page=11)
    scraped = scraper.scrape_notice_pages()

    with open("data/acasupport/raw/acasupp.json", "w", encoding="utf-8") as f:
        json.dump(scraped, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 총 {len(scraped)}개의 게시글 저장 완료!")
    driver.quit()
