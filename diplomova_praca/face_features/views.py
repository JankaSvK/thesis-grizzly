from django.shortcuts import render

# Create your views here.
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def face_features(request):
    return render(request, 'position_similarity/index.html', {})

