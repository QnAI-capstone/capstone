/* 카테고리 버튼 응답 */
document.querySelectorAll('.cat-btn').forEach(btn => {
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
    });
});