from django.shortcuts import render

# Create your views here.

from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.shortcuts import redirect
 


@login_required(login_url="/accounts/login/")
def aboutUs(req):
  return render(request, 'base/about.html', context)