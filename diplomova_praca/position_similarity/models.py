from django.db import models

# Create your models here.
class PositionRequest(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    json_request = models.TextField()
    response = models.TextField()
