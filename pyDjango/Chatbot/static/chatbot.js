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
  messageList.scrollTop = messageList.scrollHeight;

  messageInput.value = '';

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
      const time = formatCurrentTime();

      const botMessageItem = document.createElement('li');
      botMessageItem.classList.add('message', 'received');
      botMessageItem.innerHTML = `
        <div class="message-text">
          <div class="message-content">${response}</div>
          <div class="message-time">${time}</div>
        </div>`;
      messageList.appendChild(botMessageItem);
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

/* 기본 버튼 */
// 버튼 클릭에 대한 응답 매핑
const chatbotResponses = {
  "학사공지": {
    message: "학사공지 알려주세요",
    botResponse: `어떤 학사 공지가 궁금하신가요?`
  },
  "학과정보": {
    message: "학과정보 알려주세요",
    botResponse: `어떤 학과 정보가 궁금하신가요?`,
    showSubButtons: true
  }
};

// 버튼 클릭 이벤트 처리
document.querySelectorAll('.chatbot-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const type = btn.dataset.type;
    const config = chatbotResponses[type];

    if (!config) return;

    const time = formatCurrentTime();
    insertDateIfNeeded();

    // 사용자 메시지
    const userMessageItem = document.createElement('li');
    userMessageItem.classList.add('message', 'sent');
    userMessageItem.innerHTML = `
      <div class="message-text">
        <div class="message-content">${config.message}</div>
        <div class="message-time">${time}</div>
      </div>`;
    messageList.appendChild(userMessageItem);

    // 챗봇 응답 메시지
    const botMessageItem = document.createElement('li');
    botMessageItem.classList.add('message', 'received');

    let contentHTML = `
      <div class="message-text">
        <div class="message-content">${config.botResponse}`;

    // 기존 메시지 생성 (학사일정 등 단순 버튼용)
    if (!config.showSubButtons) {
      const botMessageItem = document.createElement('li');
      botMessageItem.classList.add('message', 'received');
      let contentHTML = `
    <div class="message-text">
      <div class="message-content">${config.botResponse}<br></div>
      <div class="message-time">${time}</div>
    </div>`;

      botMessageItem.innerHTML = contentHTML;
      messageList.appendChild(botMessageItem);
      scrollToBottom();
    }
    // 학과정보처럼 버튼이 따로 필요한 경우
    else {
      // 텍스트만 따로 말풍선 출력
      const botTextOnly = document.createElement('li');
      botTextOnly.classList.add('message', 'received');
      botTextOnly.innerHTML = `
    <div class="message-text">
      <div class="message-content">${config.botResponse}</div>
      <div class="message-time">${time}</div>
    </div>`;
      messageList.appendChild(botTextOnly);

      // 버튼만 따로 출력
      const botButtons = document.createElement('li');
      botButtons.classList.add('message', 'received');
      botButtons.innerHTML = `
    <div class="message-text">
      <div class="chat-box">
        <a href="#none" class="icon_box sub-btn" data-sub="과목">
          <img class="notice-image" src="/static/image/과목.png">
          <span class="m_txt"><span>과목<br></span></span>
        </a>
        <a href="#none" class="icon_box sub-btn" data-sub="학과">
          <img class="notice-image" src="/static/image/학과.png">
          <span class="m_txt"><span>학과<br></span></span>
        </a>
        <a href="#none" class="icon_box sub-btn" data-sub="이수요건">
          <img class="notice-image" src="/static/image/이수요건.png">
          <span class="m_txt"><span>이수요건<br></span></span>
        </a>
      </div>
      <div class="message-time">${time}</div>
    </div>`;
      messageList.appendChild(botButtons);
      scrollToBottom();
    }

    messageList.scrollTop = messageList.scrollHeight;

    // 서브 버튼 클릭 이벤트 (동적 요소이므로 나중에 바인딩)
    setTimeout(() => {
      document.querySelectorAll('.sub-btn').forEach(sub => {
        sub.addEventListener('click', () => {
          const subType = sub.dataset.sub;
          const time = formatCurrentTime();

          const userMsg = document.createElement('li');
          userMsg.classList.add('message', 'sent');
          userMsg.innerHTML = `
            <div class="message-text">
              <div class="message-content">${subType} 정보 알려줘</div>
              <div class="message-time">${time}</div>
            </div>`;
          messageList.appendChild(userMsg);

          const botMsg = document.createElement('li');
          botMsg.classList.add('message', 'received');
          botMsg.innerHTML = `
            <div class="message-text">
              <div class="message-content">${subType}에 대한 정보를 준비 중입니다!</div>
              <div class="message-time">${time}</div>
            </div>`;
          messageList.appendChild(botMsg);
          scrollToBottom();

          messageList.scrollTop = messageList.scrollHeight;
        });
      });
    }, 100);
  });
});

/* 로딩 메시지 추가 */
function addLoadingMessage() {
  const messagesList = document.querySelector('.messages-list');
  const loadingMessage = document.createElement('li');
  loadingMessage.className = 'message received loading';
  loadingMessage.innerHTML = `
    <div class="message-text">
      <div class="message-content">
        <span></span><span></span><span></span>
      </div>
      <div class="message-time">입력 중...</div>
    </div>
  `;
  messagesList.appendChild(loadingMessage);
  scrollToBottom();
}

function replaceLoadingMessageWithResponse(text, time) {
  const loadingMessage = document.querySelector('.message.received.loading');
  if (loadingMessage) {
    loadingMessage.innerHTML = `
      <div class="message-text">
        <div class="message-content">${text}</div>
        <div class="message-time">${time}</div>
      </div>
    `;
    loadingMessage.classList.remove('loading');
    scrollToBottom();
  }
}