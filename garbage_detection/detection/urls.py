from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DetectionAlertViewSet

router = DefaultRouter()
router.register(r'alerts', DetectionAlertViewSet)

urlpatterns = [
    path('', include(router.urls)),
]