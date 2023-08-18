from django.contrib import admin
from django.urls import path 
from . import views

urlpatterns = [
    path('admin',  admin.site.urls),
    path('', views.INDEX, name = 'home'),
    path('csvfiles', views.showallCSV, name ="csvfiles" ),
    path('csv/<str:id>', views.showaCsv, name ="showaCsv" ),
    path('add', views.ADD, name = 'add'),
    path('edit', views.EDIT, name = 'edit'),
    path('update/<str:id>', views.UPDATE   , name = 'update'),
    path('updatecsv/<str:id>', views.UPDATECSV   , name = 'updatecsv'),
    path('delete/<str:id>', views.DELETE   , name = 'delete'),
    path('deletecsv/<str:id>', views.DELETECSV   , name = 'deletecsv'),
    path('seegraph/<str:id>', views.runSarimaView, name = 'seegraph'),
    path('sarimaAnalysis/<str:id>', views.runSarimaAnalysis, name = "sarimaAnalysis"),
    path('varPlot/<str:id>', views.varDifPlot, name = "sarimaAnalysis")

]




handler404 = 'CRUD.views.error_404_view'