from django.contrib import admin
from .models import Profile
from .models import *
# Register your models here.
admin.site.register(Profile)
admin.site.register([sales,Dataset])