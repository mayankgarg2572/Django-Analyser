from django.shortcuts import render
# from django.http import HttpResponse
from .forms import CsvModelForm
from .models import Csv
import csv
from polls.models import Sale
# Create your views here.


def upload_file_view(req):
    form = CsvModelForm(req.POST or None, req.FILES or None)
    if form.is_valid():
        form.save()
        form = CsvModelForm()
    return render(req, 'csvs/upload.html', {'form': form})


def csv_list_view(req):
    qs = Csv.objects.all()
    return render(req, 'csvs/list.html', {'qs': qs})


