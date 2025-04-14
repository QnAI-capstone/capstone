## 📌 가상 환경 설정 방법

### 1. Python 가상 환경 생성
프로젝트 폴더 내에서 Python 가상 환경을 생성합니다. 아래 명령어를 터미널에 입력하세요.

#### (macOS / Linux)
```bash
python3 -m venv .venv
```
#### (Windows)
```bash
python -m venv .venv
```

### 2. 가상 환경 활성화
(macOS / Linux)
```bash
source .venv/bin/activate
```

(Windows)
```bash
.venv\Scripts\activate
```
위 명령어를 실행하면 가상 환경이 활성화됩니다. 활성화된 후, 터미널의 프롬프트에 .venv가 표시됩니다.

### 3. 가상 환경 비활성화
가상 환경을 비활성화하려면 아래 명령어를 입력합니다.

```bash
deactivate
```

### 4. 필수 패키지 설치
프로젝트에 필요한 모든 패키지를 설치하려면, 아래 명령어를 입력하세요.

```bash
pip install -r requirements.txt
```

### 5. requirements.txt 업데이트
새로운 패키지를 설치한 후, requirements.txt 파일을 최신 상태로 업데이트하려면 아래 명령어를 사용하세요.

```bash
pip freeze > requirements.txt
```
### 6. .venv 가상 환경을 다른 팀원과 공유하기
.venv 폴더는 버전 관리에 포함시키지 않도록 .gitignore 파일에 추가되어 있습니다. 따라서 다른 팀원들은 위의 방법을 따라 가상 환경을 설정하면 됩니다.

## 📌 프로젝트 실행
프로젝트를 실행하려면, 가상 환경을 활성화한 후, 필요한 스크립트를 실행하세요.

예시:

```bash
python modules/scraper.py
```