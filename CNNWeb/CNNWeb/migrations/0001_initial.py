# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2017-04-29 16:21
from __future__ import unicode_literals

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Dataset',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('size', models.IntegerField()),
                ('np', models.FloatField()),
                ('date_time', models.DateTimeField(default=datetime.datetime(2017, 4, 29, 16, 21, 28, 778619, tzinfo=utc))),
                ('source', models.CharField(blank=True, max_length=50, null=True)),
                ('intro', models.CharField(blank=True, max_length=200, null=True)),
                ('file', models.FileField(upload_to='data/labeled/')),
            ],
        ),
        migrations.CreateModel(
            name='TrainProject',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('num_of_kernel', models.IntegerField()),
                ('kernel_size', models.IntegerField()),
                ('local_1', models.IntegerField()),
                ('local_2', models.IntegerField()),
                ('learning_rate', models.FloatField()),
                ('batch_size', models.IntegerField()),
                ('ema', models.FloatField()),
                ('max_step', models.IntegerField()),
                ('data_set', models.CharField(default='1', max_length=40)),
                ('data_set_eval', models.CharField(default='1', max_length=40)),
                ('train_state', models.CharField(max_length=10)),
                ('date_time', models.DateTimeField(default=datetime.datetime(2017, 4, 29, 16, 21, 28, 778619, tzinfo=utc))),
            ],
        ),
    ]