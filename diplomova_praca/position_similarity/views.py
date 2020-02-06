from django.http import HttpResponse
from django.shortcuts import render
import json

# Create your views here.
from django.views.decorators.csrf import csrf_exempt


def index(request):
    context = {
        "ranking_results": [
            {"img_src": "/static/images/lookup/00001040.jpg"},
            {"img_src": "https://www.galeje.sk/web_object/12510_s.png"},
            {"img_src": "https://www.matfyz.cz/files/hlrljg-mathematics-1550844-639x453.jpg"},
            {"img_src": "https://www.galeje.sk/web_object/12510_s.png"},
            {"img_src": "https://www.matfyz.cz/files/hlrljg-mathematics-1550844-639x453.jpg"},
            {"img_src": "https://www.galeje.sk/web_object/12510_s.png"},
            {"img_src": "https://www.matfyz.cz/files/hlrljg-mathematics-1550844-639x453.jpg"},
            {"img_src": "https://www.galeje.sk/web_object/12510_s.png"},
            {"img_src": "https://www.matfyz.cz/files/hlrljg-mathematics-1550844-639x453.jpg"},
            {"img_src": "https://www.matfyz.cz/files/hlrljg-mathematics-1550844-639x453.jpg"},
            {"img_src": "https://www.matfyz.cz/files/hlrljg-mathematics-1550844-639x453.jpg"},
            {"img_src": "https://www.matfyz.cz/files/hlrljg-mathematics-1550844-639x453.jpg"},
            {"img_src": "https://www.matfyz.cz/files/hlrljg-mathematics-1550844-639x453.jpg"},
        ]
    }
    return render(request, 'position_similarity/index.html', context)


@csrf_exempt
def position_similarity(request):
    # 1. Spracuje poziadavok
    # 2. Vyrenderuje novu stranku s novymi obrazkami
    # 3. Zachova ten request v tom platne

    request_data = request.POST
    request_data_list = json.loads(request_data['json_data'])

    context = {
        "ranking_results": [
            {"img_src": "/static/images/lookup/00001040.jpg"},
            {"img_src": "https://www.galeje.sk/web_object/12510_s.png"}]}
    return render(request, 'position_similarity/index.html', context)
