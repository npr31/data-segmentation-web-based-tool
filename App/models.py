from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class Profile(models.Model):
     user = models.OneToOneField(User, on_delete=models.CASCADE)
     phone_number = models.IntegerField(null=True)
     phone_number = models.IntegerField(null=True)
     address = models.CharField(max_length=400)
     branch = models.CharField(max_length=200)
     is_branchmanager = models.BooleanField(default=False)
     is_salesmanager = models.BooleanField(default=False)

     def __str__(self) -> str:
          return self.user.username + " profile"

class BranchManager(models.Model):
     user = models.OneToOneField(User, on_delete=models.CASCADE,null=True)
     phone_number = models.IntegerField(null=True)
     address = models.CharField(max_length=400)
     branch = models.CharField(max_length=200)

class sales(models.Model):
     user = models.OneToOneField(User, on_delete=models.CASCADE,null=True)
     phone_number = models.IntegerField(null=True)
     address = models.CharField(max_length=400)
     branch = models.CharField(max_length=200)

class Dataset(models.Model):
     branch = models.CharField(max_length=400)
     dataset = models.FileField(upload_to='datasets')