from rest_framework import serializers
from .models import DetectionAlert

class DetectionAlertSerializer(serializers.ModelSerializer):
    class Meta:
        model = DetectionAlert
        fields = '__all__'