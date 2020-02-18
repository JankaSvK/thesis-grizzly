from Lib import uuid
from django.db import models

# Create your models here.
class PositionRequest(models.Model):
    json_request = models.TextField()
    response = models.TextField()