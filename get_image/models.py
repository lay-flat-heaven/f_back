from enum import Flag
from django.db import models

# Create your models here.

class ImageFileModels(models.Model):
    file = models.FileField(blank=False, null=False)
    remark = models.CharField(max_length=20)
    timestamp = models.DateTimeField(auto_now_add=True)
