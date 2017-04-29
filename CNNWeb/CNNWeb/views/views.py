# coding:utf-8
import json
import threading
import time
from django import forms
from django.http import HttpResponse
from django.shortcuts import render
from django.utils.timezone import now
from .. import models
from tf_models import Train

threadlist = {}

class uploadFile(forms.Form):
    fileupload = forms.FileField()
    name = forms.CharField(max_length=20)
    source = forms.CharField(max_length=50)

def home(request):


    return render(request=request,template_name='home.html',context={'pagename':'run_state'})


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
    data_set = forms.CharField(max_length=40)

def train(request,id):
    print(id)
    if id is not None:

    #给定id了，说明是查看相应的train项目

        return  render(request = request,template_name='train_single.html',context={'pagename':'train','train_id': id})

    if request.method == "GET":
        return render(request,'train.html',{'pagename':'train'})
    if request.method == "POST":
        form = TrainModelForm(request.POST)

        if form.is_valid():
            data = form.cleaned_data
            try:
                tp = models.TrainProject(id=request.POST['id'])
            except KeyError:
                tp = models.TrainProject()
            tp.get_data(data)
            tp.save()

            return HttpResponse('success')
        else:
            return HttpResponse('Fail')
def upload(request):

    if request.method == 'POST':
        form = uploadFile(request.POST,request.FILES)
        if form.is_valid():
            tmp = open('tmp.txt','wb')
            for chunk in request.FILES['fileupload'].chunks():
                tmp.write(chunk)
            tmp.close()
            try:
                tmp = open('tmp.txt','r',encoding='utf-8')
            except TypeError:
                tmp = open('tmp.txt', 'r')
            n = 0
            p = 0
            for line in tmp:
                if line[0]=='1':
                    n+=1
                    continue
                elif line[0]=='0':
                    p+=1
            print n+p
            instance = models.Dataset(file = request.FILES['fileupload'],name = request.POST['name'],source=request.POST['source'],size = n+p,np = n/p,date_time=now())
            instance.save()
            return HttpResponse('success')

        return HttpResponse('failed')
    return render(request = request,template_name='upload.html',context={'pagename':'upload_file'})
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
    query_set_ds = models.Dataset.objects.all()
    for q in query_set_ds:
        ds = {}
        print q.file
        ds['name'] = str(q.file).split('/')[-1]
        ds['size'] = q.size
        ds['date_time'] = str(q.date_time).split('.')[0]
        ds['source'] = q.source
        ds['intro'] = q.intro
        ds['np'] = q.np
        datasets.append(ds)
    return HttpResponse(json.dumps(datasets),content_type='application/json')
def trainset(request,id):

    if id is None:
        data = request.POST
        trainsets = []
        query_ts = models.TrainProject.objects.all()
        print(query_ts)
        for t in query_ts:
            ds = {}
            ds['id'] = t.id
            ds['name'] = t.name
            ds['date_time'] = str(t.date_time).split('.')[0]
            ds['data_set'] = t.data_set
            ds['state'] = t.train_state
            trainsets.append(ds)
        return HttpResponse(json.dumps(trainsets),content_type='application/json')
    else:
        query_ts = models.TrainProject.objects.get(id=id)
        t_ins ={}
        t_ins['name'] = query_ts.name

        t_ins['data_set'] = query_ts.data_set

        t_ins['num_of_kernel'] = query_ts.num_of_kernel

        t_ins['kernel_size'] = query_ts.kernel_size

        t_ins['local_1'] = query_ts.local_1

        t_ins['local_2'] = query_ts.local_2

        t_ins['learning_rate'] = query_ts.learning_rate

        t_ins['ema'] = query_ts.ema
        t_ins['batch_size'] = query_ts.batch_size
        t_ins['max_step'] = query_ts.max_step
        t_ins['data_time'] = str(query_ts.date_time).split('.')[0]
        return HttpResponse(json.dumps(t_ins), content_type='application/json')

class train_thread(threading.Thread):
    state = 'file_t'
    progress = 0
    message = ''
    file_size = 1
    max_step= 1
    def change_state(self,state,step,message):
        self.state = state
        self.message = message
        if state == 'file_t':
            self.progress= step*100.0/self.file_size
        elif state == 'train':
            self.progress= step*100.0/self.max_step
        print self.progress

    def __init__(self,model):
        threading.Thread.__init__(self)
        self.model = model
        file_name = model.data_set
        file_name = 'data/labeled/'+file_name
        DS = models.Dataset.objects.get(file = file_name)
        self.file_size = DS.size
        self.max_step = model.max_step
    def run(self):
        Train.run(self.model,self)
        q_t_p = models.TrainProject.objects.filter(id=self.model.id)
        q_t_p.update(train_state = 'finished')


def run_train(request):
    if request.method=='POST':
        data = request.POST
        id = data['id']
        ts = models.TrainProject.objects.filter(id = id)

        ts_m = models.TrainProject.objects.get(id = data['id'])
        t_thread = train_thread(ts_m)
        threadlist[id] = t_thread
        t_thread.start()
        ts.update(train_state= 'running')
        return HttpResponse('success')
def train_state(request,id):
    if request.method != 'POST':
        return render(request, template_name='train_state.html', context={'pagename': 'train_state'})
    else:
        ts_thread = threadlist[id]
        state = ts_thread.state
        message= ts_thread.message
        progress = ts_thread.progress
        mes = {'state':state,'progress':progress,'message':message}
        return HttpResponse(json.dumps(mes),'application/json')
