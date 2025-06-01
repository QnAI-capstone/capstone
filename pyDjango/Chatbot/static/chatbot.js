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

// 로딩 메시지 요소
const loadingMessageHTML = `
  <li class="message received" id="loadingMessage">
    <div class="message-text">
      <div class="message-content">로딩 중...</div>
    </div>
  </li>
`;

const chatbotResponses = {
  "이수요건": {
    message: "이수요건 알려주세요"
  },
  "과목정보": {
    message: "과목정보 알려주세요"
  }
};

document.querySelectorAll('.category-btn').forEach(btn => {
  btn.addEventListener('click', (event) => {
    event.preventDefault();
    
    const type = btn.dataset.type;
    const config = chatbotResponses[type];
    const message = btn.value;
    const time = formatCurrentTime();
    insertDateIfNeeded();

    // 사용자 메시지 추가
    const userMessageItem = document.createElement('li');
    userMessageItem.classList.add('message', 'sent');
    userMessageItem.innerHTML = `
      <div class="message-text">
        <div class="message-content">${config.message}</div>
        <div class="message-time">${time}</div>
      </div>`;
    messageList.appendChild(userMessageItem);

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
      const response = data.response;  // 챗봇의 응답 받기
      const botMessageItem = document.createElement('li');
      botMessageItem.classList.add('message', 'received');
      botMessageItem.innerHTML = `
        <div class="message-text">
          <div class="message-content">${response}</div>
          <div class="message-time">${time}</div>
        </div>`;
      messageList.appendChild(botMessageItem);
    })
    .catch(error => {
      console.error('Error:', error);
    });
  });
});


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

  // 로딩중
  const loadingMessage = document.createElement('li');
  loadingMessage.classList.add('message', 'received');
  loadingMessage.innerHTML = `
    <div class="message-text">
      <div class="message-content">
        <p class="loading">로딩 중...</p>
      </div>
    </div>
  `;
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