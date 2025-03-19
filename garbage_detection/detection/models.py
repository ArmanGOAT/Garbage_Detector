
from django.db import models

class DetectionAlert(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    category = models.CharField(max_length=100)  # Например, 'plastic', 'garbage'
    confidence = models.FloatField()
    image = models.ImageField(upload_to='detections/', blank=True, null=True)

    def __str__(self):
        return f"{self.category} - {self.confidence:.2f}"