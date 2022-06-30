from rest_framework import serializers
import models

class FileModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ImageFileModels
        fields = ('file', 'remark', 'timestamp')