import numpy as np
import pandas
import plotly as pl
import plotly.io as pio
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

def baseline(x, nsteps=10):
    n = x.shape[0]
    b = []
    for i in range(n):
        b.append(x[np.random.choice(nsteps), :])
    return np.vstack(b)

def moving_average(x, step=1, window=10):
    seq = []
    n = x.shape[0]
    for i in np.arange(0, n, step):
        idx = np.arange(np.maximum(0, i - window), np.minimum(n - 1, i + window + 1))
        seq.append(np.mean(x[idx, :], axis=0))
    return np.vstack(seq)

def prepare_traces(data, trace_data, n=320):
    dx = data[2, 0] - data[1,0]
    if 'baseline' in trace_data['name'].lower():
        ma = baseline(data[:, 1:])
    else:
        ma = data[:, 1:] #moving_average(data[:, 1:])
    print(ma)
    x = np.arange(n) * dx
    x_rev = x[::-1]
    y_upper = ma[:n, 2]
    y_lower = ma[:n, 1]
    y_lower = y_lower[::-1]
    y_avg = ma[:n, 0]
    lu_trace = go.Scatter(
        x=x+x_rev,
        y=y_upper+y_lower,
        fill='tozerox',
        fillcolor=trace_data['alpha_color'],
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name=trace_data['name']
    )
    main_trace = go.Scatter(
        x=x,
        y=y_avg,
        line=dict(color=trace_data['color']),
        mode='lines',
        name=trace_data['name'],
    )
    return main_trace, lu_trace

if __name__ == '__main__':

    #attack = 'botnet_attack'
    #attack = 'exfiltration_attack'
    attack = 'slowloris_attack'
    fname = 'logs/{0}/{1}/mlp/progress.csv'
    algs = ['dqn', 'ppo']
    figname = 'figs/{0}.pdf'.format(attack)

    methods = [
        {
            'name': 'Baseline (do nothing)',
            'color': 'rgb(0,100,80)',
            'alpha_color': 'rgba(0,100,80,0.2)',
            'subdir': 'dqn'
        },
        {
            'name': 'DQN',
            'color': 'rgb(253,106,2)',
            'alpha_color': 'rgba(253,106,2,0.2)',
            'subdir': 'dqn'
        },
        {
            'name': 'PPO',
            'color': 'rgb(0,176,246)',
            'alpha_color': 'rgba(0,176,246,0.2)',
            'subdir': 'ppo'
        }
    ]

    colnames = [['steps', 'stats/timestamps'], ['reward', 'stats/reward'], ['reward_min', 'stats/reward_min'], ['reward_max', 'stats/reward_max'], ['steps', 'stats/timesteps']]
    main_traces = []
    lu_traces = []
    for method in methods[2:]:
        p = pandas.read_csv(fname.format(attack, method['subdir']), delimiter=',', dtype=float)
        keys = [item for item in p.keys()]
        data = np.zeros((p.values.shape[0], len(colnames)))
        for i,colname in enumerate(colnames):
            if colname[0] in keys:
                data[:, i] = p.values[:, keys.index(colname[0])]
            elif colname[1] in keys:
                data[:, i] = p.values[:, keys.index(colname[1])]
        print(data)
        main_trace, lu_trace = prepare_traces(data, method)
        main_traces.append(main_trace)
        lu_traces.append(lu_trace)

    data = lu_traces + main_traces

    layout = go.Layout(
        # paper_bgcolor='rgb(255,255,255)',
        # plot_bgcolor='rgb(229,229,229)',
        xaxis=dict(
            title='False Positive Rate',
            # gridcolor='rgb(255,255,255)',
            showgrid=True,
            showline=False,
            showticklabels=True,
            # tickcolor='rgb(127,127,127)',
            ticks='outside',
            zeroline=False
        ),
        yaxis=dict(
            title='True Positive Rate',
            # gridcolor='rgb(255,255,255)',
            showgrid=True,
            showline=False,
            showticklabels=True,
            # tickcolor='rgb(127,127,127)',
            ticks='outside',
            zeroline=False
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    pio.write_image(fig, figname)