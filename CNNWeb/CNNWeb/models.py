from  django.db import models
import datetime
from django.utils.timezone import now
class Dataset(models.Model):
    def __unicode__(self):
        return self.name
    name = models.CharField(max_length=100)
    size = models.IntegerField()
    np = models.FloatField()
    date_time = models.DateTimeField(default=now())
    source = models.CharField(max_length=50,blank=True,null=True)
    intro = models.CharField(max_length=200,blank=True,null=True)
    file = models.FileField(upload_to='data/labeled/')
class TrainProject(models.Model):
    def get_data(self,dict):
        try:
            self.id = dict['id']
        except KeyError:
            pass
        self.name = dict['name']
        self.batch_size = dict['batch_size']
        self.num_of_kernel = dict['num_of_kernel']
        self.kernel_size = dict['kernel_size']
        self.local_1 = dict['local_1']
        self.local_2 = dict['local_2']
        self.learning_rate = dict['learning_rate']
        self.ema = dict['ema']
        self.max_step = dict['max_step']
        self.data_set = dict['data_set']
        self.train_state = 'ready'
        # ready/runnning/finished/evaluating
        self.date_time = now()+datetime.timedelta(hours=8)
    def __unicode__(self):
        return self.name
    name = models.CharField(max_length=100)
    num_of_kernel =models.IntegerField()
    kernel_size = models.IntegerField()
    local_1 = models.IntegerField()
    local_2 = models.IntegerField()
    learning_rate = models.FloatField()
    batch_size = models.IntegerField()
    ema = models.FloatField()
    max_step = models.IntegerField()
    data_set = models.CharField(default='1',max_length=40)
    train_state = models.CharField(max_length=10)
    date_time = models.DateTimeField(default=now())