from pathlib import Path
from typing import Set

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from diplomova_praca_lib.position_similarity.models import PositionMethod
from diplomova_praca_lib.position_similarity.position_similarity_request import available_images
from shared.utils import dir_files


@csrf_exempt
def video_images(request):
    src = request.POST.get('src', '')
    print(src)
    files = [{"img_src": '/' + str(path)} for path in dir_files(Path(src[1:]).parent)]
    return JsonResponse({'files': files}, status=200)

@csrf_exempt
def images_loaded_in_dataset(method:PositionMethod) -> Set[str]:
    return available_images(method)

