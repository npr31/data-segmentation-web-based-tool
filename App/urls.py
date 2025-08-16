from django.contrib import admin
from django.urls import path,include
from . import views
urlpatterns = [
    path('', views.upload_dataset, name='home'),
    path('upload/', views.upload_dataset2, name='upload_datasetsale'),
    path('select_columns/', views.generate_chart, name='generate_chart'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('signup/', views.sign_up, name='signup'),
    path('sales/', views.Sales, name='sales'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('prediction/', views.prediction_view, name='prediction'),
    path('download-high-value-customers/', views.download_high_value_customers, name='download_high_value_customers'),
    path('segmentation/', views.segmentation_view, name='segmentation'),
    path('segmentation-result/', views.segmentation_result_view, name='segmentation_result'),
    path('help/', views.help_view, name='help'),
    path('contact/', views.contact_view, name='contact'),
    path('update_clusters/', views.update_clusters, name='update_clusters'),
    path('analyze-clusters/', views.analyze_clusters, name='analyze_clusters'),
]


