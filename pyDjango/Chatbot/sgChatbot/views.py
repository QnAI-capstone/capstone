from django.shortcuts import render
from django.http import JsonResponse

# Create your views here.
def chatbot(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        response = '무엇을 도와드릴까요?'
        return JsonResponse({'message': message, 'response': response})
    return render(request, 'chatbot.html')