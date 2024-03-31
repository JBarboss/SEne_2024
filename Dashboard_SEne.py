import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

import numpy as np
import os
import pandas as pd
import pickle
import plotly.express as px

from sklearn import  metrics
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css'
]


def load_data():
    current_dir = os.path.dirname(__file__)

    file_2017_name = 'IST_Civil_Pav_2017.csv'
    file_2018_name = 'IST_Civil_Pav_2018.csv'
    file_weather_name = 'IST_meteo_data_2017_2018_2019.csv'
    file_clean_name = 'IST_Civil_Pav_DataHourly_Clean.csv'
    file_test_name = 'testData_2019_Civil.csv'

    file_names = [file_2017_name, file_2018_name, file_weather_name, file_clean_name, file_test_name]

    file_2017_dir = os.path.join(current_dir, 'data', file_2017_name)
    file_2018_dir = os.path.join(current_dir, 'data', file_2018_name)
    file_weather_dir = os.path.join(current_dir, 'data', file_weather_name)
    file_clean_dir = os.path.join(current_dir, 'data', file_clean_name)
    file_test_dir = os.path.join(current_dir, 'data', file_test_name)

    df_2017 = pd.read_csv(file_2017_dir)
    df_2018 = pd.read_csv(file_2018_dir)
    df_weather = pd.read_csv(file_weather_dir)
    df_clean = pd.read_csv(file_clean_dir)
    df_test = pd.read_csv(file_test_dir)

    df_clean['Date'] = pd.to_datetime(df_clean['Date'])     # Turning date from object to datetime
    df_clean = df_clean.set_index ('Date', drop = True)

    dataframes = {}

    dataframes['raw_2017'] = df_2017
    dataframes['raw_2018'] = df_2018
    dataframes['weather'] = df_weather
    dataframes['clean'] = df_clean
    dataframes['test_raw'] = df_test

    return file_names, dataframes


def new_train_features(dataframe):
    dataframe['Power (-1 hour)'] = dataframe['Power (kWh)'].shift(1)
    dataframe['Power (-2 hours)'] = dataframe['Power (kWh)'].shift(2)
    dataframe['Power (-1 day)'] = dataframe['Power (kWh)'].shift(24)
    dataframe['Power (-1 week)'] = dataframe['Power (kWh)'].shift(168)

    dataframe['Hour (sin)'] = np.sin(2 * np.pi * dataframe.index.hour / 24)
    dataframe['Hour (cos)'] = np.cos(2 * np.pi * dataframe.index.hour / 24)

    dataframe['Week (sin)'] = np.sin(2 * np.pi * dataframe.index.weekday / 7)
    dataframe['Week (cos)'] = np.cos(2 * np.pi * dataframe.index.weekday / 7)

    dataframe = dataframe.dropna()

    return dataframe


def clean_test_data(dataframe):

    dataframe['Date'] = pd.to_datetime(dataframe['Date'])     # Turning date from object to datetime
    dataframe = dataframe.set_index ('Date', drop = True)

    dataframe.rename(columns = {'Civil (kWh)': 'Power (kWh)',
                                'temp_C': 'Temperature (ºC)',
                                'HR': 'Humidity (%)',
                                'windSpeed_m/s': 'Wind Speed (m/s)',
                                'windGust_m/s': 'Wind Gust (m/s)',
                                'pres_mbar': 'Pressure (mbar)',
                                'solarRad_W/m2': 'Solar Irradiance (W/m2)',
                                'rain_mm/h': 'Rainfall (mm/h)',
                                'rain_day': 'Rain (day)'},
                     inplace = True)
    
    dataframe = dataframe.dropna()

    dataframe_complete = dataframe

    dataframe_complete['Power (-1 hour)'] = dataframe['Power (kWh)'].shift(1)      # Create new column with 1 hour shift
    dataframe_complete['Power (-2 hours)'] = dataframe['Power (kWh)'].shift(2)     # Create new column with 2 hour shift
    dataframe_complete['Power (-1 day)'] = dataframe['Power (kWh)'].shift(24)      # Create new column with 24 hour ( 1 day ) shift

    dataframe_complete['Hour (sin)'] = np.sin(2 * np.pi * dataframe.index.hour / 24)       # Create new column with sin of hour
    dataframe_complete['Hour (cos)'] = np.cos(2 * np.pi * dataframe.index.hour / 24)       # Create new column with cos of hour

    dataframe_complete = dataframe_complete.dropna()

    dataframe_default = dataframe_complete[['Power (kWh)','Power (-1 hour)','Power (-2 hours)','Power (-1 day)','Hour (sin)','Hour (cos)']]

    return dataframe, dataframe_complete, dataframe_default


def load_models():
    current_dir = os.path.dirname(__file__)
    models_dir = os.path.join(current_dir, 'models')

    model_files = [
        'reg_model_LR.pkl',
        'reg_model_DT.pkl',
        'reg_model_RF.pkl',
        'reg_model_GB.pkl',
        'reg_model_XGB.pkl',
        'reg_model_BT.pkl',
        'reg_model_NN.pkl'
    ]

    models = {}

    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        with open(model_path, 'rb') as file:
            model_name = os.path.splitext(model_file)[0]  # Get model name without extension
            models[model_name] = pickle.load(file)

    return models


def train_models(dataframe, model_names, models):
    Y = dataframe.values[:,0]
    X = dataframe.values[:,1:]

    regressors = {}
    if 'reg_model_LR' in model_names:
        regressors['reg_model_LR'] = LinearRegression()

    if 'reg_model_DT' in model_names:
        regressors['reg_model_DT'] = DecisionTreeRegressor(min_samples_leaf=5)

    if 'reg_model_RF' in model_names:
        parameters = {'bootstrap': True,                        # Define parameters
              'min_samples_leaf': 3,
              'n_estimators': 200, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 20,
              'max_leaf_nodes': None}
        regressors['reg_model_RF'] = RandomForestRegressor(**parameters)

    if 'reg_model_GB' in model_names:
        regressors['reg_model_GB'] = GradientBoostingRegressor()

    if 'reg_model_XGB' in model_names:
        regressors['reg_model_XGB'] = XGBRegressor()

    if 'reg_model_BT' in model_names:
        regressors['reg_model_BT'] = BaggingRegressor()

    if 'reg_model_NN' in model_names:
        regressors['reg_model_NN'] = MLPRegressor(hidden_layer_sizes=(7,7,3))


    for name, regressor in regressors.items():
        regressor.fit(X, Y)
        models[name] = regressor

    return models


def predict_data(dataframe, models, predictions):
    Z = dataframe.values
    Y=Z[:,0]
    X=Z[:, 1:]

    for model_name, model in models.items():
        predictions[model_name] = model.predict(X)

    return predictions


def calculate_errors(models, metric):
    errors={}
    y_true = dataframes['test_complete']['Power (kWh)']
    model_errors = {}

    for model in models:
        model_errors = {}

        if 'NMBE' in metric:
            MBE = np.mean(y_true - predictions[model])
            NMBE = MBE / np.mean(y_true)
            model_errors['NMBE (%)'] = NMBE * 100

        if 'CV(RMSE)' in metric:
            RMSE = np.sqrt(metrics.mean_squared_error(y_true, predictions[model]))
            cvRMSE = RMSE / np.mean(y_true)
            model_errors['CV(RMSE) (%)'] = cvRMSE * 100

        errors[model] = model_errors

    return pd.DataFrame(errors).T


def generate_raw_data_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([
                html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ], className='styled-table')


def generate_error_table(dataframe):
    return html.Table([
        html.Thead(
            html.Tr([html.Th('Model')] + [html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(model)] + [html.Td(dataframe.loc[model][col]) for col in dataframe.columns]
            ) for model in dataframe.index
        ])
    ], className='styled-table')


def generate_ranking_table(features, ranking):
    ranked_features = [feature for _, feature in sorted(zip(ranking, features))]

    return html.Table([
        html.Thead(
            html.Tr([
                html.Th('Ranking')] + [html.Th('Feature')])
        ),
        html.Tbody([
            html.Tr([
                html.Td(number+1),
                html.Td(feature)
            ]) for number, feature in enumerate(ranked_features)
        ])
    ], className='styled-table')


def first_plot_forecast():
    y_pred = predictions['reg_model_LR']

    df_combined = pd.DataFrame({'Date': dataframes['test_default'].index, 'Real': dataframes['test_default']['Power (kWh)'], 'reg_model_LR': y_pred})

    fig = px.line(df_combined, x='Date', y=['Real', 'reg_model_LR'])
    fig.update_layout(yaxis_title='Power (kWh)', legend_title='Data:')

    return fig


def rank(values):
    value_index_tuples = [(abs(value), index) for index, value in enumerate(values)]

    sorted_tuples = sorted(value_index_tuples, key=lambda x: x[0], reverse=True)

    ranked_tuples = [(rank, tup[1]) for rank, tup in enumerate(sorted_tuples, start=1)]

    ranked_tuples.sort(key=lambda x: x[1])

    rankings = [rank for rank, _ in ranked_tuples]
    return rankings


def filter_method():
    Z = dataframes['train_complete'].values

    Y=Z[:,0]
    X=Z[:,1:] 

    features=SelectKBest(k = 5,score_func = f_regression)
    fit = features.fit(X,Y)

    fig = px.bar(x=dataframes['train_complete'].columns[1:], y=fit.scores_)
    ranking = rank(fit.scores_)

    return fig, ranking


def wrapper_method(reg_model):
    Z = dataframes['train_complete'].values

    Y=Z[:,0]
    X=Z[:,1:] 

    if reg_model == 'reg_model_LR':
        model = LinearRegression()

    elif reg_model == 'reg_model_DT':
        model = DecisionTreeRegressor()

    elif reg_model == 'reg_model_RF':
        model = RandomForestRegressor()

    rfe = RFE(model, n_features_to_select=1)
    fit = rfe.fit(X,Y)

    inverted_ranking = len(fit.ranking_) - fit.ranking_ + 1

    fig = px.bar(x=dataframes['train_complete'].columns[1:], y=inverted_ranking)
    ranking = fit.ranking_

    return fig, ranking


def embedded_method(reg_model):
    Z = dataframes['train_complete'].values

    Y=Z[:,0]
    X=Z[:,1:] 

    if reg_model == 'reg_model_LR':
        model = LinearRegression()
        fit = model.fit(X,Y)
        values = fit.coef_

    elif reg_model == 'reg_model_DT':
        model = DecisionTreeRegressor()
        fit = model.fit(X,Y)
        values = fit.feature_importances_

    elif reg_model == 'reg_model_RF':
        model = RandomForestRegressor()
        fit = model.fit(X,Y)
        values = fit.feature_importances_


    fig = px.bar(x=dataframes['train_complete'].columns[1:], y=values)
    ranking = rank(values)

    return fig, ranking


file_names, dataframes = load_data()
dataframes['train_complete'] = new_train_features(dataframes['clean'])
dataframes['test_clean'], dataframes['test_complete'], dataframes['test_default'] = clean_test_data(dataframes['test_raw'])

models = load_models()

predictions = {}
predictions = predict_data(dataframes['test_default'], models, predictions)

tab = 'button-home'


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1('SEne',
                    style={'margin': '0px',
                           'fontWeight': 'bold',
                           'textAlign': 'center',
                           'whiteSpace': 'normal'}),

            html.H4('Project #2',
                    style={'marginTop': '0px',
                           'textAlign': 'center',
                           'whiteSpace': 'normal'}),

            html.Hr(style={'borderColor': 'black',
                           'marginTop': '0px',
                           'marginBottom': '10px'}),

            html.Div([
                html.Button('Home',
                            id='button-home',
                            n_clicks=0,
                            className='button',
                            style={'backgroundColor': '#79AF82'}),  

                html.Button('Raw Data',
                            id='button-raw',
                            n_clicks=0,
                            className='button'),

                html.Button('Data Analysis',
                            id='button-analysis',
                            n_clicks=0,
                            className='button'),

                html.Button('Features',
                            id='button-features',
                            n_clicks=0,
                            className='button',
                            style={'textAlign': 'center'}),

                html.Button('Forecast',
                            id='button-forecast',
                            n_clicks=0,
                            className='button'),

                ],
                id='buttons-container',
                style={'display': 'flex',
                       'flexDirection': 'column'}),

        ]),

        html.Div([
            html.Hr(style={'borderColor': 'black',
                           'marginBottom': '10px'}),

            html.P('By João Barbosa',
                   style={'textAlign': 'center',
                          'marginBottom': '0px'}),

        ],
        style={'alignItems': 'flex-end'}),

    ],
    style={'padding': '20px',
           'position': 'absolute',
           'top': '8px',
           'bottom': '8px',
           'left': '8px',
           'width': '150px',
           'display': 'flex',
           'flexDirection': 'column',
           'overflowY': 'auto',
           'justifyContent': 'space-between',
           'backgroundColor': '#5c9966',
           'borderRadius': '10px'}),

    html.Div([
    ],
    id='main-content',
    style={'padding': '20px',
           'position': 'absolute',
           'top': '8px',
           'bottom': '8px',
           'left': '196px', 
           'right': '8px', 
           'display': 'inline-block',
           'overflowY': 'auto',
           'marginLeft': '10px',
           'backgroundColor': '#dae9dd',
           'borderRadius': '10px'})
])


content_mapping = {
    'button-home':
        html.Div([
            html.H1('Project #2 - Energy Forecast Dashboard'),

            html.Hr(style={'borderColor': 'black',
                           'marginTop': '0px',
                           'marginBottom': '10px'}),

            html.P('''This dashboard was made as an evaluation element of the course Energy 
                   Services, at Instituto Superior Técnico - ULisboa, and has been developed
                   on top on another project from the same course.'''),

            html.P('''The objective of the first project was to create an electricity 
                   consumption forecast model for a building in IST, specifically from data 
                   from the Civil Building.'''),

            html.P('''Now in the second project the objective is to display the data and 
                   insight gathered in a web dashboard.'''),

            html.P('''The dashboard will be able to display the following information:'''),

            html.P(''' • Raw Data from 2019;''',
                   style={'paddingLeft': '25px'}),

            html.P(''' • Forecast, the prediction from January to March 2019 compared 
                   to the real data;''',
                   style={'paddingLeft': '25px'}),

            html.P(''' • Metrics of the model.''',
                   style={'paddingLeft': '25px'}),

            html.P('''Additionally, it will also be able to:'''),

            html.P(''' • Choose how to visualize the data and what data to display, for 
                   a quick exploratory data analysis;''',
                   style={'paddingLeft': '25px'}),

            html.P(''' • Choose from a selection of diferent forecast models and what
                   error metrics to show and calculate;''',
             style={'paddingLeft': '25px'}),

            html.P('''This can all be accessed by clicking on the buttons on the left.'''),

            html.Hr(style={'borderColor': 'black',
                           'marginTop': '10px',
                           'marginBottom': '10px'}),

            html.P('''This was the process used to develop the model in project one, and 
                   roughly the sections of this dashboard.'''),

            html.Div([
                html.Img(src='assets/Model-img.png',
                         style={'width': '80%'})
            ], style={'marginTop': '10px',
                      'textAlign': 'center'}),

            html.P('''This dashboard was developed in python programming language 
                       using the dash package.''',
                   style={'position': 'absolute',
                          'bottom': '0px'}),

        ], id='home-content'),


    'button-raw':
        html.Div([
            html.Div([
                html.Div([
                    html.P('Select file:',
                           style={'fontWeight': 'bold'}),

                    dcc.Dropdown(
                        id='dropdown-files',
                        options=[
                            {'label': file_names[0], 'value': 'raw_2017'},
                            {'label': file_names[1], 'value': 'raw_2018'},
                            {'label': file_names[2], 'value': 'weather'},
                            {'label': file_names[4], 'value': 'test_raw'},
                        ],
                        value='raw_2017'
                    ),
                ], style={'width': '80%',
                          'marginRight': '20px'}),

                html.Div([
                    html.P('Number of rows:',
                           style={'fontWeight': 'bold'}),

                    dcc.Input(value=15,
                              type='number',
                              id='rows',
                              style={'width': '100%'}),

                ], style={'width': '20%'}),

            ], style={'display': 'flex',
                      'flexDirection': 'row'}),

            html.Div([
                dcc.Loading(
                    id="loading-raw-table",
                    type="circle",
                    children=generate_raw_data_table(dataframes['raw_2017'], 15))
            ], id='raw-table',
               style={'overflowX': 'auto',
                      'overflowY': 'auto'}),

        ], id='raw-content'),


    'button-analysis': 
        html.Div([
            html.Div([
                html.Div([
                    html.P('Select dataframe:',
                        style={'fontWeight': 'bold'}),

                    dcc.Dropdown(
                        id='dropdown-analysis-file',
                        options=[
                            {'label': 'Train data', 'value': 'clean'},
                            {'label': 'Test data', 'value': 'test_clean'},
                        ],
                        value='clean'
                    ),
                ], style={'width': '50%',
                            'marginRight': '20px'}),

                html.Div([
                    html.P('Select plot type:',
                        style={'fontWeight': 'bold'}),

                    dcc.Dropdown(
                        id='dropdown-plot-type',
                        options=[
                            {'label': 'Line', 'value': 'line'},
                            {'label': 'Histogram', 'value': 'histogram'},
                            {'label': 'Box', 'value': 'box'},
                            {'label': 'Violin', 'value': 'violin'},
                            {'label': 'Scatter', 'value': 'scatter'}
                        ],
                        value='line'
                    ),
                ], style={'width': '50%'}),

            ], style={'display': 'flex',
                      'flexDirection': 'row',
                      'marginBottom': '10px'}),

            html.Div([
                html.P('Select variables to plot:',
                    style={'fontWeight': 'bold'}),

                dcc.Dropdown(
                    id='dropdown-columns',
                    multi=True,
                    value=['Power (kWh)']
                ),
            ], style={'marginBottom': '20px'}),

            html.Div([
                dcc.Loading(
                    id="loading-analysis-plot",
                    type="circle",
                    children=dcc.Graph(id='analysis-plot', figure= px.line(dataframes['clean'], x=dataframes['clean'].index, y='Power (kWh)')))
            ])
        ]),


    'button-features':
        html.Div([
            html.Div([
                html.Div([
                    html.P('Selection method:',
                           style={'fontWeight': 'bold'}),

                    dcc.Dropdown(
                        id='dropdown-sel-method',
                        options=[
                            {'label': 'Filter Method', 'value': 'filter'},
                            {'label': 'Wrapper Method', 'value': 'wrapper'},
                            {'label': 'Embedded Method', 'value': 'embedded'},
                        ],
                        value='filter'
                    ),
                ], id='sel-method',
                    style={'width': '100%'}),

                html.Div([
                    html.P('Model:',
                           style={'fontWeight': 'bold'}),

                    dcc.Dropdown(
                        id='dropdown-sel-model',
                        options=[
                            {'label': 'Linear Regression',  'value': 'reg_model_LR'},
                            {'label': 'Decision Tree',      'value': 'reg_model_DT'},
                            {'label': 'Random Forest',      'value': 'reg_model_RF'},
                        ],
                        value='reg_model_LR'),

                ], id='sel-model',
                    style={'width': '20%',
                            'marginLeft': '20px'}),

            ], style={'display': 'flex',
                      'flexDirection': 'row',
                      'marginBottom': '10px'}),

            html.Div([
                html.P('Results:',
                        style={'fontWeight': 'bold'}),

                dcc.Loading(
                    id="loading-sel-results",
                    type="circle",
                    children=
                        html.Div([
                            dcc.Graph(id='feature-plot')
                        ])),

                html.Div(id='ranking-table',style={'marginTop': '20px'})

            ], id='sel-results',
             style={'marginBottom': '20px'}),

            html.Div([
                html.P('Select features:',
                    style={'fontWeight': 'bold'}),

                dcc.Dropdown(
                    id='dropdown-features',
                    multi=True,
                    options=[{'label': col, 'value': col} for col in dataframes['train_complete'].columns[1:]]
                ), 
            ], style={'marginBottom': '20px'}),

            html.Div([
                html.Div([
                    html.P('Select model to train:',
                        style={'fontWeight': 'bold'}),

                    dcc.Dropdown(
                        id='dropdown-train-models',
                        multi=True,
                        options=[
                            {'label': 'Linear Regression',         'value': 'reg_model_LR'},
                            {'label': 'Decision Tree',             'value': 'reg_model_DT'},
                            {'label': 'Random Forest',             'value': 'reg_model_RF'},
                            {'label': 'Gradient Boosting',         'value': 'reg_model_GB'},
                            {'label': 'Extreme Gradient Boosting', 'value': 'reg_model_XGB'},
                            {'label': 'Bootstrapping',             'value': 'reg_model_BT'},
                            {'label': 'Neural Network',            'value': 'reg_model_NN'}
                        ],
                        value=['reg_model_LR']
                    ), 
                ], style={'width': '91%'}),

                html.Div([
                    html.P(''),

                    dcc.Loading(
                        children=
                            html.Button('Train',
                                    id='button-train',
                                    n_clicks=0,
                                    className='button',
                                    style={'backgroundColor': '#79AF82',
                                        'marginTop': '22px',
                                        'width': '100%'}),  
                    ),

                ], style={'width': '9%',
                            'marginLeft': '20px'}),

            ], style={'display': 'flex',
                      'flexDirection': 'row',
                      'marginBottom': '10px'}),

            html.Div(id='train-label'),

        ], id='features-content'),


    'button-forecast':
        html.Div([
            html.Div([
                html.P('Select model:',
                       style={'fontWeight': 'bold'}),

                dcc.Dropdown(
                    id='dropdown-models',
                    multi=True,
                    options=[
                        {'label': 'Linear Regression',         'value': 'reg_model_LR'},
                        {'label': 'Decision Tree',             'value': 'reg_model_DT'},
                        {'label': 'Random Forest',             'value': 'reg_model_RF'},
                        {'label': 'Gradient Boosting',         'value': 'reg_model_GB'},
                        {'label': 'Extreme Gradient Boosting', 'value': 'reg_model_XGB'},
                        {'label': 'Bootstrapping',             'value': 'reg_model_BT'},
                        {'label': 'Neural Network',            'value': 'reg_model_NN'}

                    ],
                    value=['reg_model_LR']
                ),

            ], style={'marginBottom': '20px'}),

            html.Div([
                dcc.Loading(
                    id="loading-forecast-plot",
                    type="circle",
                    children=dcc.Graph(id='forecast-plot', figure=first_plot_forecast())
                )
            ]),

            html.Div([
                html.Div([
                    generate_error_table(calculate_errors(['reg_model_LR'], ['NMBE'])),
                ], id='error-table',
                   style={'overflowX': 'auto',
                          'overflowY': 'auto',
                          'width': '80%',
                          'marginRight': '20px'}),

                dcc.Checklist(
                    options=[{'label': metric, 'value': metric} for metric in ['NMBE', 'CV(RMSE)']],
                    value=['NMBE'],
                    id='errors',
                    className='custom-checkbox',
                    style={'width': '20%', 'marginTop': '28px', 'fontSize': '20px'}
                )

            ], style={'display': 'flex',
                      'flexDirection': 'row'}),

        ], id='forecast-content'),

}



@app.callback(
    Output('main-content', 'children'),
    [Input('button-home', 'n_clicks'),
     Input('button-raw', 'n_clicks'),
     Input('button-analysis', 'n_clicks'),
     Input('button-features', 'n_clicks'),
     Input('button-forecast', 'n_clicks')]
)
def update_main_content(clicks1, clicks2, clicks3, clicks4, clicks5):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    tab = button_id

    return content_mapping.get(button_id, content_mapping.get('button-home'))

@app.callback(
    [Output('button-home', 'style'),
     Output('button-raw', 'style'),
     Output('button-analysis', 'style'),
     Output('button-features', 'style'),
     Output('button-forecast', 'style')],
    [Input('button-home', 'n_clicks'),
     Input('button-raw', 'n_clicks'),
     Input('button-analysis', 'n_clicks'),
     Input('button-features', 'n_clicks'),
     Input('button-forecast', 'n_clicks')]
)
def update_button_color(clicks1, clicks2, clicks3, clicks4, clicks5):
    button_styles = [
        {'backgroundColor': '#5c9966'},
        {'backgroundColor': '#5c9966'},
        {'backgroundColor': '#5c9966'},
        {'backgroundColor': '#5c9966'},
        {'backgroundColor': '#5c9966'}
    ]

    ctx = dash.callback_context
    if ctx.triggered:
        clicked_button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if clicked_button_id == 'button-home':
            button_styles[0] = {'backgroundColor': '#79AF82'}
        elif clicked_button_id == 'button-raw':
            button_styles[1] = {'backgroundColor': '#79AF82'}
        elif clicked_button_id == 'button-analysis':
            button_styles[2] = {'backgroundColor': '#79AF82'}
        elif clicked_button_id == 'button-features':
            button_styles[3] = {'backgroundColor': '#79AF82'}
        elif clicked_button_id == 'button-forecast':
            button_styles[4] = {'backgroundColor': '#79AF82'}

        return button_styles

    if clicks1 == 0 and clicks2 == 0 and clicks3 == 0 and clicks4 == 0 and clicks5 == 0:
        button_styles[0] = {'backgroundColor': '#79AF82'}

        return button_styles



@app.callback(
    Output('raw-table', 'children'),
    [Input('dropdown-files', 'value'),
    Input('rows', 'value')]
)
def update_raw_table(selected_index, rows):
    if selected_index is None:
        return []

    return generate_raw_data_table(dataframes[selected_index], rows)



@app.callback(
    Output('dropdown-columns', 'options'),
    [Input('dropdown-analysis-file', 'value')]
)
def update_column_options(selected_file):
    if selected_file is None:
        return []
    else:
        selected_dataframe = dataframes[selected_file]
        column_options = [{'label': col, 'value': col} for col in selected_dataframe.columns]
    return column_options

@app.callback(
    Output('analysis-plot', 'figure'),
    [Input('dropdown-analysis-file', 'value'),
    Input('dropdown-plot-type', 'value'),
    Input('dropdown-columns', 'value')]
)
def update_analysis_plot(selected_file, plot_type, selected_columns):
    if selected_file is None or plot_type is None or selected_columns is None:
        return {}

    selected_dataframe = dataframes[selected_file]

    if plot_type == 'line':
        fig = px.line(selected_dataframe, x=selected_dataframe.index, y=selected_columns)
    elif plot_type == 'histogram':
        fig = px.histogram(selected_dataframe, x=selected_columns)
    elif plot_type == 'box':
        fig = px.box(selected_dataframe, y=selected_columns)
    elif plot_type == 'violin':
        fig = px.violin(selected_dataframe, y=selected_columns, box=True)
    elif plot_type == 'scatter':
        if len(selected_columns) == 1:
            fig = px.scatter(selected_dataframe, x=selected_dataframe.index, y=selected_columns)
        else:
            fig = px.scatter(selected_dataframe, x=selected_columns[0], y=selected_columns[1:])

    fig.update_layout(height=600)      
    fig.update_layout(yaxis_title='Value', legend_title='Variable:')

    return fig



@app.callback(
    [Output('sel-method', 'style'),
     Output('sel-model', 'style'),
     Output('feature-plot', 'figure'),
     Output('ranking-table', 'children')],
    [Input('dropdown-sel-method', 'value'),
     Input('dropdown-sel-model', 'value')],
)
def update_sel_model_visibility_and_plot(selected_method, selected_model):

    if selected_method == 'filter':
        sel_method_style = {'width': '100%'}
        sel_model_style = {'display': 'none'}

        fig, ranking = filter_method()
    else:
        sel_method_style = {'width': '80%'} 
        sel_model_style = {'display': 'block', 'width': '20%', 'marginLeft': '20px'}

        if selected_method == 'wrapper':
            fig, ranking = wrapper_method(selected_model)

        if selected_method == 'embedded':
            fig, ranking = embedded_method(selected_model)

    fig.update_xaxes(title_text='')
    fig.update_yaxes(title_text='')

    ranking_table = generate_ranking_table(dataframes['train_complete'].columns[1:], ranking)

    return sel_method_style, sel_model_style, fig, ranking_table

@app.callback(
    Output('train-label', 'children'),
    [Input('button-train', 'n_clicks')],
    [State('dropdown-features', 'value'),
     State('dropdown-train-models', 'value')],
)
def train_callback(clicks, features, model_names):
    if features is None or model_names is None:
        return dash.no_update

    global predictions, models

    train_dataframe = dataframes['train_complete'][['Power (kWh)'] + features]
    test_dataframe = dataframes['test_complete'][['Power (kWh)'] + features]
    print('Training...')
    models = train_models(train_dataframe, model_names, models)
    selected_models = {model_name: models[model_name] for model_name in model_names if model_name in models}
    print('Predicting...')
    predictions = predict_data(test_dataframe, selected_models, predictions)
    print('Done!')

    return html.P('Training complete!')



@app.callback(
    Output('forecast-plot', 'figure'),
    [Input('dropdown-models', 'value')],
    [State('forecast-plot', 'relayoutData')]
)
def update_forecast_plot(selected_models, prev_zoom):
    y_pred = {}
    for model_name in selected_models:
        if model_name in predictions:
            y_pred[model_name] = predictions[model_name]

    df_combined = pd.DataFrame({'Date': dataframes['test_complete'].index, 'Real': dataframes['test_complete']['Power (kWh)'], **y_pred})

    fig = px.line(df_combined, x='Date', y=['Real'] + selected_models)

    fig.update_layout(yaxis_title='Power (kWh)', legend_title='Data:')

    if prev_zoom:
        if 'xaxis.range[0]' in prev_zoom and 'xaxis.range[1]' in prev_zoom:
            fig['layout']['xaxis']['range'] = [prev_zoom['xaxis.range[0]'], prev_zoom['xaxis.range[1]']]
        if 'yaxis.range[0]' in prev_zoom and 'yaxis.range[1]' in prev_zoom:
            fig['layout']['yaxis']['range'] = [prev_zoom['yaxis.range[0]'], prev_zoom['yaxis.range[1]']]

    return fig

@app.callback(
    Output('error-table', 'children'),
    [Input('dropdown-models', 'value'),
    Input('errors', 'value')]
)
def update_error_table(selected_models, selected_metrics):
    if not selected_models or not selected_metrics:
        return []

    errors_df = calculate_errors(selected_models, selected_metrics)
    return generate_error_table(errors_df)



if __name__ == '__main__':
    app.run_server()
