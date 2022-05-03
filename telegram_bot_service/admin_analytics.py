import base64
from io import BytesIO
import gevent
from locust import HttpUser, task, constant_pacing
from locust.env import Environment
from locust.stats import stats_printer, stats_history,  StatsCSVFileWriter
from locust.log import setup_logging
from math import floor
import random
log = "-"
def parse_analytics(data, prefix=""):
    """
    Prepare analitics for sending to admin.
    data -- json with analitics
    prefix -- separator
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
    wait_time = constant_pacing(1)
    host = "http://dispatcher_service:7000"

    @task
    def my_task(self):
        resp = "/?text=test+text+"+str(abs(floor(random.random()*100000000)))
        self.client.get(resp)

    @task
    def task_404(self):
        print("/non-existing-path")


def latency_test():
    #launch without cash
    ans = "-"
    # setup Environment and Runner
    env = Environment(user_classes=[User])
    #my_stat_to_csv = StatsCSVFileWriter(env, "14", "stat_tests.csv", full_history=True)
    env.create_local_runner()
    
    #env.events.request.add_listener(on_request)
    # start a WebUI instance
    #env.create_web_ui("0.0.0.0", 8089)
    logs = safe_massage(env)
    # start a greenlet that periodically outputs the current stats
    gevent.spawn(stats_printer(env.stats))

    # start a greenlet that save current stats to history
    gevent.spawn(stats_history, env.runner)

    # start the test
    env.runner.start(1, spawn_rate=10)

    # in 60 seconds stop the runner
    gevent.spawn_later(3, lambda: env.runner.quit())

    # wait for the greenlets
    env.runner.greenlet.join()

    # stop the web server for good measures
    #env.web_ui.stop()
    return logs.get_message()

class safe_massage:
    def __init__(self,env) -> None:
        env.events.request.add_listener(self.on_request)
        self.log = "Latency test Results\n"
    def get_message(self):
        return self.log
    def on_request(self,request_type, name, response_time, response_length, exception, context: dict, start_time=None, **kwargs):
        #print("request_type "+str(request_type)+"time "+str(response_time))
        self.log+='Name: {}, request_type: {},response_time: {},response_length: {},exception: {},context: {}, start_time: {}'\
         '\n'.format(name,request_type,response_time,response_length,exception,context,start_time)