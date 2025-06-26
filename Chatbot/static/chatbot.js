/* 챗봇 동작 */
const messageList = document.querySelector('.messages-list');
const messageForm = document.querySelector('.message-form');
const messageInput = document.querySelector('.message-input');

function formatCurrentTime() {
  const now = new Date();
  return now.getHours().toString().padStart(2, '0') + ':' +
    now.getMinutes().toString().padStart(2, '0');
}

function formatTodayDate() {
  const now = new Date();
  return now.getFullYear() + '-' +
    (now.getMonth() + 1).toString().padStart(2, '0') + '-' +
    now.getDate().toString().padStart(2, '0');
}

function insertDateIfNeeded() {
  const today = formatTodayDate();
  const lastDate = localStorage.getItem('lastMessageDate');

  if (lastDate !== today) {
    const dateItem = document.createElement('li');
    dateItem.classList.add('date-separator');
    dateItem.textContent = '📅 ' + today;
    messageList.appendChild(dateItem);
    localStorage.setItem('lastMessageDate', today);
  }
}

const chatbotResponses = {
  "이수요건": {
    message: "이수요건 알려주세요"
  },
  "과목정보": {
    message: "과목정보 알려주세요"
  },
  "학사공지": {
    message: "학사공지 알려주세요"
  }
};

/* 카테고리 버튼 응답 */
document.querySelectorAll('.category-btn').forEach(btn => {
  btn.addEventListener('click', (event) => {
    event.preventDefault();

    const type = btn.dataset.type;
    const config = chatbotResponses[type];
    const message = config.message;
    const category = btn.value;
    const time = formatCurrentTime();
    insertDateIfNeeded();

    // 사용자 메시지 추가
    const userMessageItem = document.createElement('li');
    userMessageItem.classList.add('message', 'sent');
    userMessageItem.innerHTML = `
      <div class="message-text">
        <div class="message-content">${message}</div>
        <div class="message-time">${time}</div>
      </div>`;
    messageList.appendChild(userMessageItem);

    // 챗봇 응답 처리
    fetch('', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        'csrfmiddlewaretoken': document.querySelector('[name=csrfmiddlewaretoken]').value,
        'message': message,
        'category': category
      })
    })
      .then(response => response.json())
      .then(data => {
        const response = data.response;
        const botMessageItem = document.createElement('li');
        botMessageItem.classList.add('message', 'received');
        botMessageItem.innerHTML = `
        <div class="message-text">
          <div class="message-content">${response}</div>
          <div class="message-time">${time}</div>
        </div>`;
        messageList.appendChild(botMessageItem);
        scrollToBottom()
      })
      .catch(error => {
        console.error('Error:', error);
      });
  });
});

/* 일반 질문 응답 */
messageForm.addEventListener('submit', (event) => {
  event.preventDefault();

  const message = messageInput.value.trim();
  if (message.length === 0) {
    return;
  }

  const time = formatCurrentTime();
  insertDateIfNeeded();  // 오늘 날짜 표시

  // 사용자 메시지 추가
  const userMessageItem = document.createElement('li');
  userMessageItem.classList.add('message', 'sent');
  userMessageItem.innerHTML = `
      <div class="message-text">
        <div class="message-content">${message}</div>
        <div class="message-time">${time}</div>
      </div>`;
  messageList.appendChild(userMessageItem);

  messageInput.value = '';

  // 로딩 메시지 요소
  const loadingMessageHTML = `
  <div class="message-text">
    <div class="message-content">
      <span class="dot dot1"></span>
      <span class="dot dot2"></span>
      <span class="dot dot3"></span>
      </div>
  </div>
`;
  // 로딩중
  const loadingMessage = document.createElement('li');
  loadingMessage.classList.add('message', 'received');
  loadingMessage.innerHTML = loadingMessageHTML;
  messageList.appendChild(loadingMessage);

  // 챗봇 응답 처리
  fetch('', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      'csrfmiddlewaretoken': document.querySelector('[name=csrfmiddlewaretoken]').value,
      'message': message
    })
  })
    .then(response => response.json())
    .then(data => {
      const response = data.response;
      const questions = data.questions || [];
      const time = formatCurrentTime();

      // 로딩 메시지 제거
      loadingMessage.remove();

      const botMessageItem = document.createElement('li');
      botMessageItem.classList.add('message', 'received');
      botMessageItem.innerHTML = `
        <div class="message-text">
          <div class="message-content">${response}</div>
          <div class="message-time">${time}</div>
        </div>`;
      messageList.appendChild(botMessageItem);
      if (questions.length > 0) {
        const questionContainer = document.createElement('li');
        questionContainer.classList.add('follow-up-questions');

        questions.forEach(question => {
          const button = document.createElement('button');
          button.classList.add('category-btn');
          button.innerText = question;

          // 버튼 클릭 시 입력창에 질문 넣고 폼 자동 제출
          button.addEventListener('click', () => {
            messageInput.value = question;
            messageForm.dispatchEvent(new Event('submit')); // form 자동 제출
          });

          questionContainer.appendChild(button);
        });

        botMessageItem.querySelector('.message-text').appendChild(questionContainer);
      }
      scrollToBottom()
    })
    .catch(error => {
      console.error('Error:', error);  // 오류 처리
      // 로딩 메시지 제거
      loadingMessage.remove();

      // 오류 메시지 표시
      const errorMessageItem = document.createElement('li');
      errorMessageItem.classList.add('message', 'received');
      errorMessageItem.innerHTML = `
        <div class="message-text">
          <div class="message-content">오류가 발생했습니다. 다시 시도해주세요.</div>
        </div>`;
      messageList.appendChild(errorMessageItem);
      messageList.scrollTop = messageList.scrollHeight;
    });
});

/* 스크롤 */
function scrollToBottom() {
  const messageBox = document.querySelector('.chat-container');
  if (messageBox) {
    messageBox.scrollTop = messageBox.scrollHeight;
  }
}

// 1. 페이지 로딩 후 스크롤
window.addEventListener('load', scrollToBottom);

// 2. 메시지 전송 후 (폼 전송 후) 스크롤
const messageFormScroll = document.querySelector('.message-form');
messageFormScroll.addEventListener('submit', function (e) {
  // 이 코드는 Ajax 없이 동기 submit일 경우에만 유효
  // Ajax라면 Ajax 완료 콜백 안에서 호출해야 함
  setTimeout(scrollToBottom, 100); // 약간의 지연 후 스크롤
});

/* 버튼 재전송 */
const reButton = document.querySelector(".btn-re");

reButton.addEventListener("click", function () {
  reloadScript("/static/button.js?v=${Date.now()}`")
  const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value;

  const defaultHTML = `
      <div class="message-text">
        <div class="message-content">
          <p style="margin:0;">질문의 카테고리를 선택해주세요.</p>
        </div>
      </div>
    `;

  const defaultbtnHTML = `
      <div class="message-text">
        <div id="category_buttons" class="chat-box">
          <form id="category_form" method="post">
            <input type="hidden" name="csrfmiddlewaretoken" value="${csrfToken}">
            <button class="cat-btn" type="button" name="message" value="1" data-type="이수요건">
              <span class="m_txt"><span>이수요건<br></span></span>
            </button>
            <button class="cat-btn" type="button" name="message" value="2" data-type="과목정보">
              <span class="m_txt"><span>과목정보<br></span></span>
            </button>
            <button class="cat-btn" type="button" name="message" value="3" data-type="학사공지">
              <span class="m_txt"><span>학사공지<br></span></span>
            </button>
          </form>
        </div>
      </div>
    `;
  const reMessage1 = document.createElement('li');
  reMessage1.classList.add('message', 'received');
  reMessage1.innerHTML = defaultHTML;
  messageList.appendChild(reMessage1);
  const reMessage2 = document.createElement('li');
  reMessage2.classList.add('message', 'received');
  reMessage2.innerHTML = defaultbtnHTML;
  messageList.appendChild(reMessage2);
  scrollToBottom()
});

function reloadScript(src) {
  // 기존 script 제거 (src로 비교)
  const oldScript = document.querySelector(`script[src="${src}"]`);
  if (oldScript) {
    oldScript.remove();
  }

  // 새 script 생성
  const newScript = document.createElement("script");
  newScript.src = src;
  newScript.onload = () => {
    console.log(`✅ ${src} reloaded`);
  };
  newScript.onerror = () => {
    console.error(`❌ Failed to load ${src}`);
  };

  document.body.appendChild(newScript);
}