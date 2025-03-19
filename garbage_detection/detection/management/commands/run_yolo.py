from django.core.management.base import BaseCommand
from detection.test5 import run_detection_system

class Command(BaseCommand):
    help = "Запускает YOLOv5 для обнаружения мусора"

    def handle(self, *args, **kwargs):
        run_detection_system()