from django.http import HttpResponse
from django.shortcuts import render,render_to_response
from django import forms
from .. import models
import json
class uploadFile(forms.Form):
    fileupload = forms.FileField()

def home(request):


    return render(request=request,template_name='home.html',context={'pagename':'train'})


class TrainModelForm(forms.Form):
    name = forms.CharField()
    num_of_kernel = forms.IntegerField()
    kernel_size = forms.IntegerField()

    local_1 = forms.IntegerField()

    local_2 = forms.IntegerField()
    learning_rate = forms.FloatField()

    batch_size = forms.IntegerField()

    ema = forms.FloatField()

    max_step = forms.IntegerField()
    data_set = forms.IntegerField()

def train(request):
    if request.method == "POST":
        form = TrainModelForm(request.POST)
        print form.is_valid()

        if form.is_valid():
            print(form.cleaned_data)
            tp = models.TrainProject( form.cleaned_data)
            tp.save()
            print tp.id

    return HttpResponse('success')
def upload(request):

    return render(request = request,template_name='base.html',context={'pagename':'upload'})
def eval_single(request):
    pass
def eval_batch(request):
    pass
def shortcut(request):
    str = '''
    <nav class="navbar navbar-default top-navbar" role="navigation">
            <div class="navbar-header">
                <button type="button" class="navbar-toggle" data-toggle="collapse" data-target=".sidebar-collapse">
                    <span class="sr-only">Toggle navigation</span>
                    <span class="icon-bar"></span>
                    <span class="icon-bar"></span>
                    <span class="icon-bar"></span>
                </button>
                <a class="navbar-brand waves-effect waves-dark" href="index.html"><i class="large material-icons">insert_chart</i> <strong>TRACK</strong></a>

		<div id="sideNav" href=""><i class="material-icons dp48">toc</i></div>
            </div>

            <ul class="nav navbar-top-links navbar-right">
				<li><a class="dropdown-button waves-effect waves-dark" href="#!" data-activates="dropdown4"><i class="fa fa-envelope fa-fw"></i> <i class="material-icons right">arrow_drop_down</i></a></li>
				<li><a class="dropdown-button waves-effect waves-dark" href="#!" data-activates="dropdown3"><i class="fa fa-tasks fa-fw"></i> <i class="material-icons right">arrow_drop_down</i></a></li>
				<li><a class="dropdown-button waves-effect waves-dark" href="#!" data-activates="dropdown2"><i class="fa fa-bell fa-fw"></i> <i class="material-icons right">arrow_drop_down</i></a></li>
				  <li><a class="dropdown-button waves-effect waves-dark" href="#!" data-activates="dropdown1"><i class="fa fa-user fa-fw"></i> <b>John Doe</b> <i class="material-icons right">arrow_drop_down</i></a></li>
            </ul>
        </nav>'''
def dataset(request):
    datasets = []
    for i in range(3):
        ds = {'name':"DS" + str(i),'intro': "this is the test dataset" + str(i)}
        datasets.append(ds)
    return HttpResponse(json.dumps(datasets),content_type='application/json')