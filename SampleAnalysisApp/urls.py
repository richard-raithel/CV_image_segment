from django.urls import path
from . import views
from django.contrib.auth import views as auth_views


urlpatterns = [
    path('', views.entry, name='entry'),
    path('base_template', views.base_template, name='base_template'),
    path('index', views.index, name='index'),
    path('coming_soon', views.coming_soon, name='coming_soon'),
    path('gallery', views.gallery, name='gallery'),
    path('analyze_test', views.analyze_test, name='analyze_test'),
    path('analyze_user', views.analyze_user, name='analyze_user'),
    path('logout', auth_views.LogoutView.as_view(), name='logout'),
    path('batch', views.batch, name='batch'),
    path('batchSubmit', views.batchSubmit, name='batchSubmit'),
    path('reset_session_data', views.reset_session_data, name='reset_session_data'),
    # path('corrosion_detection', views.corrosion_detection, name='corrosion_detection'),
    path('chip_corrosion_detection', views.chip_corrosion_detection, name='chip_corrosion_detection'),
    path('bearing_corrosion_detection', views.bearing_corrosion_detection, name='bearing_corrosion_detection'),
    path('livestream', views.livestream, name='livestream'),
    path('signup', views.signup, name='signup'),
    path('export_to_csv', views.export_to_csv, name='export_to_csv'),
    path('create_report', views.create_report, name='create_report'),
]
