import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluators.metrics import Metrics
import evaluators.metrics_functions as mf
import loggers.logger_agents as logger

def create_line_plot(data, labels=[""], xlim=None, ylim=None):
    df = pd.DataFrame(data, index=labels).T
    if xlim != None and ylim != None:
        df.plot.line(xlim=xlim, ylim=ylim)
    elif xlim != None:
        df.plot.line(xlim=xlim)
    elif ylim != None:
        df.plot.line(ylim=ylim)
    else:
        df.plot.line()

def create_pie_plot( data, labels):
    patches, _ = plt.pie(data, labels=[f"{int(p*100)}%" for p in data])
    plt.legend(patches, labels, loc="best")

def create_bar_plot( data, labels):
    plt.bar(x=[1, 3, 5], height=data, tick_label=labels)

def plot(metric, base_save_path=None, x_label=None, y_label=None, subtitle=None, xlim=None, ylim=None):
    ax = plt.gca()
    # reset axis to start at (0.0)
    if xlim == None:
        xlim = ax.get_xlim()
        ax.set_xlim((0, xlim[1]))
    if ylim == None:
        ylim = ax.get_ylim()
        ax.set_ylim((0, ylim[1]))

    if x_label != None:
        plt.xlabel(x_label)
    if y_label != None:
        plt.ylabel(y_label)
    if subtitle != None:
        plt.title(f"""{subtitle}""")
    else:
        plt.title(f"{metric.name}")
    if base_save_path != None:
        plt.savefig(f"{base_save_path}_{metric.value}.jpeg")
    plt.show()
    plt.close()