from rest_framework import viewsets
from .models import DetectionAlert
from .serializers import DetectionAlertSerializer

class DetectionAlertViewSet(viewsets.ModelViewSet):
    queryset = DetectionAlert.objects.all().order_by('-timestamp')
    serializer_class = DetectionAlertSerializer