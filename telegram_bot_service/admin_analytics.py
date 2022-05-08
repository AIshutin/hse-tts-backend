import base64
from io import BytesIO
from pkgutil import get_data

import gevent
from locust import HttpUser, task, constant_pacing, constant
from locust.env import Environment
from locust.stats import stats_printer, stats_history,  StatsCSVFileWriter
from locust.log import setup_logging
from datetime import datetime, timezone, timedelta
from math import floor
import random
import matplotlib.pyplot as plt
import pandas as pd
log = "-"
def parse_analytics(data, prefix=""):
    """
    Prepare analitics for sending to admin.
    data -- json with analitics
    prefix -- tab
    """
    IMG_THR = 100
    imgs = []
    text = []
    for el in data:
        if isinstance(data[el], str):
            if len(data[el]) > IMG_THR:
                imgs.append(BytesIO(base64.b64decode(data[el])))
                continue
        if isinstance(data[el], dict):
            text.append(prefix + el + ":\n")
            sub_imgs, sub_text = parse_analytics(data[el], prefix=prefix + "\t")
            imgs.extend(sub_imgs)
            text.extend(sub_text)
        else:
            text.append(prefix + el + ":\t" + str(data[el]) + "\n")
    return imgs, ''.join(text)

class User(HttpUser):
    """
    User that makes request for latency test
    """
    wait_time = constant_pacing(1) # min time of seconds between requests
    host = "http://dispatcher_service:7000"

    @task
    def my_task(self):
        """
        request
        """
        resp = "/?text=test+text+new+"+str(abs(floor(random.random()*100000000)))
        self.client.get(resp)

    @task
    def task_404(self):
        """
        connection error
        """
        print("/non-existing-path")


def latency_test(users_count = 1,test_time = 10):
    """
        latency test 
        user_count -- max count of paralel requests 
        text_time -- time of test
    """
    #launch without cash
    # setup Environment and Runner
    env = Environment(user_classes=[User])
    #my_stat_to_csv = StatsCSVFileWriter(env, "14", "stat_tests.csv", full_history=True)
    env.create_local_runner()
    
    logs = SafeMassage(env)
    # start a greenlet that periodically outputs the current stats
    gevent.spawn(stats_printer(env.stats))

    # start a greenlet that save current stats to history
    gevent.spawn(stats_history, env.runner)

    # start the test
    env.runner.start(users_count, spawn_rate=1)

    # in 60 seconds stop the runner
    gevent.spawn_later(test_time, lambda: env.runner.quit())

    # wait for the greenlets
    env.runner.greenlet.join()
    
    return logs.get_message(users_count,test_time),logs.get_image(users_count,test_time)

class SafeMassage:
    """
    Save data about test
    """
    def __init__(self,env) -> None:
        """
        env -- Envairenment 
        """
        env.events.request.add_listener(self.on_request)
        self.data = {'name':[],'start_time':[], 'request_type':[],'response_length':[],'exception':[],'response_time':[]}
    def get_message(self, users_count, test_time):
        """
        get log message
        """
        data = self.get_data()
        self.log = "Test results:\n count of users: {} \n test time: {} s\n total count of requets: {}\n count of fails: {}\n mean response time: {:3f} ms \n"\
        " max response time: {:3f} ms \n min response time: {:3f} ms \n ".format(users_count,test_time,data.shape[0],data.loc[data['exception'] != None].shape[0],
        data['response_time'].mean(), data['response_time'].max(),data['response_time'].min())
        
        max_len = 2000
        if(len(self.log)>max_len):
            return "...\n"+self.log[-max_len::]
        return self.log

    def get_data(self):
        """
        return data about requests
        """
        return pd.DataFrame.from_dict(self.data)

    def get_image(self,users_count,time):
        """
        image of test
        """
        data = self.get_data()
        
        fig = plt.figure()
        plt.plot(self.data['start_time'], data['response_time'])
        plt.title("Test {} paralel user {} s.".format(users_count,time))
        plt.xlabel("start time")
        plt.ylabel("response time ms")
        figdata = BytesIO()
        fig.savefig(figdata, format='png')
        return figdata.getvalue()

    def on_request(self,request_type, name, response_time, response_length, exception, context: dict, start_time=None, **kwargs):
        """
        safe statictics about request
        """
        self.data['name'].append(name)
        self.data['request_type'].append(request_type)
        self.data['response_time'].append(response_time)
        self.data['exception'].append(exception)
        self.data['start_time'].append(datetime.fromtimestamp(start_time, tz=timezone.utc))
        self.data['response_length'].append(response_length)
        #self.log+='Name: {}, request_type: {},response_time: {},response_length: {},exception: {},context: {}, start_time: {}'\
        # '\n'.format(name,request_type,response_time,response_length,exception,context,datetime.fromtimestamp(start_time, tz=timezone.utc))
