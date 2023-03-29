# Uso de Dash/Plotly para creación de dashboards como aplicaciones Web.
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
from flask import Flask
import joblib

#--------------PREPARACIÓN SERVIDOR-----
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Opción 2
#server = Flask(__name__)
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server)

# Opción 1
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# -------------DATOS Y MODELOS--------------------
# Cargamos los datos en un dataframe de Pandas

# Cargar datos
url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
df_ = pd.read_csv(url)
features = ['Age', 'Sex', 'Embarked', 'Survived']
df = df_[features]

#Cargar modelos
model = joblib.load('model1.joblib')
model_columns = joblib.load("model_columns.joblib")

#--------------LAYOUT-------------------
# Aquí definimos toda la interfaz gráfica de nuestra aplicación
app.layout = html.Div(children=[#<Div>
    html.H1(children="Dashboard Titanic"), # <h1>"Dashboard Titanic</h1>"
    html.P(children="Dashboard con los datos del dataset de sobreviencia del Titanic"),
    html.Div(children=[
        html.P("Eje X"), 
        dcc.Dropdown(id='dropdown_x_axis', options=[{'label':'Edad', 'value':'Age'}, {'label':'Sexo', 'value':'Sex'}, {'label':'Punto de abordaje', 'value':'Embarked'}], placeholder='Selecciona..', value='Age')
        ],
        style={'width': '24%'}),
    html.Div(children=[html.P("Eje Y"),dcc.Dropdown(id='dropdown_y_axis', options=[{'label':'Edad', 'value':'Age'}, {'label':'Sexo', 'value':'Sex'}, {'label':'Punto de abordaje', 'value':'Embarked'}], placeholder='Selecciona..', value='Sex')], style={'width': '24%', 'padding-top': '10px'}),
    html.Div(id='contenedor_grafica', children=[dcc.Graph(id='grafica')]),
    html.Div(children=[
        html.H5("Sexo"), 
        dcc.Dropdown(id='dropdown_sex', options=[{'label': 'Hombre','value': 'male'}, {'label': 'Mujer', 'value': 'female'}]),
        html.H5("Edad"),
        dcc.Input(id='input_age', type='number'),
        html.H5("Punto de embarque"),
        dcc.Dropdown(id='dropdown_embarked', options=[{'label': 'S','value': 'S'}, {'label': 'Q', 'value': 'Q'}, {'label': 'C', 'value': 'C'}]),
        html.Button("Predecir", id='predecir', n_clicks=0),
        html.H4(id='resultado', children="Datos no completos para predicción.")
    ], style={'width': '24%', 'padding-top': '10px'})
])

#--------------BACKEND------------------
# Aquí definimos metodos callbacks que se ejecutarán cuando ocurran ciertos eventos en la interfaz.
# Un callback es una lista de entrada, salidas y estados. Cada uno lleva dos componentes que son: el id del elemento gráfico y su propiedad a modificar o leer.
@app.callback(
    Output('grafica', 'figure'), # Regresamos valores para la propiedad figure del objeto grafica
    [Input('dropdown_x_axis', 'value'), # Este metodo se activa cuando se modifica la propiedad value del objeto dropdown_axis_x
    Input('dropdown_y_axis', 'value')])
def generar_grafica(dd_x, dd_y):
    print("Eje X: ", dd_x)
    print("Eje Y: ", dd_y)
    if dd_x is not None and dd_y is not None:
        # Si tenemos definidos ambos ejes procedemos a graficar
        fig = px.scatter(df, x=dd_x, y=dd_y, color='Survived')
        return fig
    else:
        return []

@app.callback(
    Output('resultado', 'children'),
    [Input('predecir', 'n_clicks')],
    [State('dropdown_sex', 'value'),
    State('dropdown_embarked', 'value'),
    State('input_age', 'value')]
)
def predecir(n_clicks, sex, embarked, age):
    if n_clicks>0:
        if sex is not None and embarked is not None and age is not None:
            df_query = pd.DataFrame(data={'Sex':[sex], 'Age': [age], 'Embarked':[embarked]})
            print("Dataframe: \n", df_query)
            query = pd.get_dummies(df_query)
            query = query.reindex(columns=model_columns, fill_value=0)
            prediction = list(model.predict(query))
            print(prediction)
            if prediction[0] == 1:
                return "Sobrevivió"
            else:
                return "No sobrevivió"
        else:
            return "Error"
    else:
        return ""

if __name__ == '__main__':
    app.run_server(debug=False)