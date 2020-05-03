from django.db import models

# Create your models here.
class PositionRequest(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    json_request = models.TextField()
    response = models.TextField()


class Collage(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    overlay_image = models.TextField()
    images = models.TextField()