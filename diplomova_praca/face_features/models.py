from django.db import models

# Create your models here.
class FaceFeaturesSubmission(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    request = models.TextField()
    selected = models.TextField()
    num_hints = models.IntegerField()