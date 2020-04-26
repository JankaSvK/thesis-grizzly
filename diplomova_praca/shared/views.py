from pathlib import Path

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from shared.utils import dir_files


@csrf_exempt
def video_images(request):
    src = request.POST.get('src', '')
    print(src)
    files = [{"img_src": '/' + str(path)} for path in dir_files(Path(src[1:]).parent)]
    return JsonResponse({'files': files}, status=200)
