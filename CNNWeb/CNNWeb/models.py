from  django.db import models

class Dataset(models.Model):
    def __unicode__(self):
        return self.name
    name = models.CharField(max_length=100)
    intro = models.CharField(max_length=200,blank=True)
    dir = models.CharField(max_length=100,unique=True)
class TrainProject(models.Model):
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
    data_set = models.IntegerField()
    train_state = models.IntegerField()