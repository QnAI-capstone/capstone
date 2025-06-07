from django.shortcuts import render
from django.http import JsonResponse
from django.core.cache import cache
from datetime import datetime
import markdown
from .build_db import *
from .hybrid_rag import initialize_rag, get_categories, category_to_collection, get_response_from_retriever, auto_linkify

# chatbot
weekday_korean = ['월', '화', '수', '목', '금', '토', '일']

initialize_rag()

def chatbot(request):
    default_response = "답변"
    now = datetime.now()
    today_label = f"{now.year}년 {now.month}월 {now.day}일 ({weekday_korean[now.weekday()]})"
    chat_log = cache.get('chat_log')

    if chat_log is None:  # 서버 시작 후 초기화된 경우
        chat_log = []  # 대화 로그 초기화
    
    if request.method == 'POST':
        message = request.POST.get('message')
        category = request.POST.get('category')
        response = default_response

        chat_log.append({"role": "user", "content": message})

        # 사용자가 카테고리를 선택한 경우 처리
        if category in category_to_collection:
            selected_category = category_to_collection[category]
            cache.set(f"chat_selected:{request.user.id}", selected_category, timeout=60*60*24*30)
            response = f"✅ '{get_categories()[category]}' 카테고리가 선택되었습니다."

        # 사용자가 질문을 입력한 경우
        elif message.strip():
            selected_category = cache.get(f"chat_selected:{request.user.id}")
            if selected_category:
                raw_response = get_response_from_retriever(message, selected_category, chat_log)
                chat_log.append({"role": "assistant", "content": raw_response})
                # 3개의 대화만 저장
                if len(chat_log) > 6:
                    chat_log = chat_log[-6:]
                cache.set('chat_log', chat_log)
                raw_response = auto_linkify(raw_response)
                response = convert_markdown_to_html(raw_response)
            else:
                response = "⚠️ 먼저 카테고리를 선택해주세요."

        if request.user.is_authenticated:
            user_key = f"chat_history:{request.user.id}"
            chat_history = cache.get(user_key, [])

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

    return render(request, 'chatbot.html', {
        'chat_history': chat_history,
        'now': now.strftime('%H:%M'),
        'today': today_label
    })

def convert_markdown_to_html(markdown_text):
    return markdown.markdown(markdown_text)