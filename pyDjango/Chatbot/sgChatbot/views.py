from django.shortcuts import render
from django.http import JsonResponse
from django.core.cache import cache
from datetime import datetime

weekday_korean = ['월', '화', '수', '목', '금', '토', '일']

def chatbot(request):
    # chatgpt api 들어갈 부분
    default_response = '챗봇 답변'

    if request.method == 'POST':
        message = request.POST.get('message')
        response = default_response

        if request.user.is_authenticated:
            user_key = f"chat_history:{request.user.id}"
            chat_history = cache.get(user_key, [])

            now = datetime.now()
            today_label = f"{now.year}년 {now.month}월 {now.day}일 ({weekday_korean[now.weekday()]})"

            # 마지막 저장된 날짜 확인, 구분선 추가
            last_date = None
            for entry in reversed(chat_history):
                if entry['type'] == 'date_separator':
                    last_date = entry['date']
                    break

            if last_date != today_label:
                chat_history.append({
                    'type': 'date_separator',
                    'date': today_label
                })

            chat_history.append({
                'type': 'user',
                'text': message,
                'time': now.strftime('%H:%M')
            })

            chat_history.append({
                'type': 'bot',
                'text': response,
                'time': now.strftime('%H:%M')
            })

            # 저장 기간: 3달(90일)
            cache.set(user_key, chat_history, timeout=60 * 60 * 24 * 90)

        return JsonResponse({'message': message, 'response': response})

    # 사용자 로그인 -> chat_history 불러오기
    if request.user.is_authenticated:
        user_key = f"chat_history:{request.user.id}"
        chat_history = cache.get(user_key, [])
    else:
        chat_history = []

    now = datetime.now()
    today_label = f"{now.year}년 {now.month}월 {now.day}일 ({weekday_korean[now.weekday()]})"

    return render(request, 'chatbot.html', {
        'chat_history': chat_history,
        'now': now.strftime('%H:%M'),
        'today': today_label
    })
