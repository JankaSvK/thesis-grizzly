from typing import List

from django.db import models

# Create your models here.
from diplomova_praca_lib.face_features.models import FaceCrop
from diplomova_praca_lib.position_similarity.models import Crop


class FaceFeaturesSubmission(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    request = models.TextField()
    selected = models.TextField()
    num_hints = models.IntegerField()


