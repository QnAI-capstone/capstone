from django.core.cache import cache

def save_user_message(user_id, message):
    key = f"chat_history:{user_id}"
    history = cache.get(key, [])
    history.append(message)
    cache.set(key, history)

def get_user_chat_history(user_id):
    key = f"chat_history:{user_id}"
    return cache.get(key, [])

def clear_user_chat_history(user_id):
    key = f"chat_history:{user_id}"
    cache.delete(key)
