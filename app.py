import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import joblib
from flask import Flask

# Load the trained model and scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Create the Flask server
server = Flask(__name__)

# Create the Dash app
app = dash.Dash(__name__, server=server)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Tumor Classification Dashboard", style={'text-align': 'center'}),
    
    # Input fields for tumor features
    html.Div([
        html.Label("Feature 1:"),
        dcc.Input(id='input-feature1', type='number', placeholder='Enter Feature 1', min=0, step=1),
        html.Br(),
        
        html.Label("Feature 2:"),
        dcc.Input(id='input-feature2', type='number', placeholder='Enter Feature 2', min=0, step=1),
        html.Br(),
        
        html.Label("Feature 3:"),
        dcc.Input(id='input-feature3', type='number', placeholder='Enter Feature 3', min=0, step=1),
        html.Br(),
        
        html.Label("Feature 4:"),
        dcc.Input(id='input-feature4', type='number', placeholder='Enter Feature 4', min=0, step=1),
        html.Br(),
        
        html.Button('Classify Tumor', id='classify-btn', n_clicks=0)
    ], style={'text-align': 'center', 'padding': '20px'}),
    
    # Output for the prediction result
    html.Div(id='output-result', style={'text-align': 'center', 'padding': '20px'})
])

# Callback to handle prediction
@app.callback(
    Output('output-result', 'children'),
    [Input('classify-btn', 'n_clicks')],
    [Input('input-feature1', 'value'),
     Input('input-feature2', 'value'),
     Input('input-feature3', 'value'),
     Input('input-feature4', 'value')]
)
def classify_tumor(n_clicks, feature1, feature2, feature3, feature4):
    if n_clicks > 0:
        features = np.array([[feature1, feature2, feature3, feature4]])  # Adjust for your dataset
        features_std = scaler.transform(features)
        prediction = model.predict(features_std)
        
        result = 'Malignant' if prediction[0] == 4 else 'Benign'
        return f'The tumor is classified as: {result}'
    return ''

if __name__ == '__main__':
    app.run_server(debug=True)
