import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import joblib
from flask import Flask

# Load the trained model and scaler
model = joblib.load('svm_model.pkl')  # Ensure this path is correct
scaler = joblib.load('scaler.pkl')

# Create the Flask server
server = Flask(__name__)

# Create the Dash app
app = dash.Dash(__name__, server=server)

# Define the layout with added colors and styling
app.layout = html.Div(
    style={'backgroundColor': '#f4f4f4', 'fontFamily': 'Arial, sans-serif'},
    children=[
        html.H1(
            "Tumor Classification Dashboard",
            style={'text-align': 'center', 'color': '#333', 'padding': '20px', 'backgroundColor': '#007BFF'}
        ),
        html.Div(
            style={'text-align': 'center', 'padding': '30px'},
            children=[
                html.Div([
                    html.Label("Feature 1:", style={'font-weight': 'bold'}),
                    dcc.Input(id='input-feature1', type='number', placeholder='Enter Feature 1', min=0, step=1,
                              style={'margin': '10px', 'padding': '5px'}),
                ], style={'display': 'inline-block', 'width': '20%'}),
                html.Div([
                    html.Label("Feature 2:", style={'font-weight': 'bold'}),
                    dcc.Input(id='input-feature2', type='number', placeholder='Enter Feature 2', min=0, step=1,
                              style={'margin': '10px', 'padding': '5px'}),
                ], style={'display': 'inline-block', 'width': '20%'}),
                html.Div([
                    html.Label("Feature 3:", style={'font-weight': 'bold'}),
                    dcc.Input(id='input-feature3', type='number', placeholder='Enter Feature 3', min=0, step=1,
                              style={'margin': '10px', 'padding': '5px'}),
                ], style={'display': 'inline-block', 'width': '20%'}),
                html.Div([
                    html.Label("Feature 4:", style={'font-weight': 'bold'}),
                    dcc.Input(id='input-feature4', type='number', placeholder='Enter Feature 4', min=0, step=1,
                              style={'margin': '10px', 'padding': '5px'}),
                ], style={'display': 'inline-block', 'width': '20%'}),
                html.Div([
                    html.Label("Feature 5:", style={'font-weight': 'bold'}),
                    dcc.Input(id='input-feature5', type='number', placeholder='Enter Feature 5', min=0, step=1,
                              style={'margin': '10px', 'padding': '5px'}),
                ], style={'display': 'inline-block', 'width': '20%'}),
                html.Div([
                    html.Label("Feature 6:", style={'font-weight': 'bold'}),
                    dcc.Input(id='input-feature6', type='number', placeholder='Enter Feature 6', min=0, step=1,
                              style={'margin': '10px', 'padding': '5px'}),
                ], style={'display': 'inline-block', 'width': '20%'}),
                html.Div([
                    html.Label("Feature 7:", style={'font-weight': 'bold'}),
                    dcc.Input(id='input-feature7', type='number', placeholder='Enter Feature 7', min=0, step=1,
                              style={'margin': '10px', 'padding': '5px'}),
                ], style={'display': 'inline-block', 'width': '20%'}),
                html.Div([
                    html.Label("Feature 8:", style={'font-weight': 'bold'}),
                    dcc.Input(id='input-feature8', type='number', placeholder='Enter Feature 8', min=0, step=1,
                              style={'margin': '10px', 'padding': '5px'}),
                ], style={'display': 'inline-block', 'width': '20%'}),
                html.Div([
                    html.Label("Feature 9:", style={'font-weight': 'bold'}),
                    dcc.Input(id='input-feature9', type='number', placeholder='Enter Feature 9', min=0, step=1,
                              style={'margin': '10px', 'padding': '5px'}),
                ], style={'display': 'inline-block', 'width': '20%'}),
                html.Br(),
                html.Button(
                    'Classify Tumor', 
                    id='classify-btn', 
                    n_clicks=0, 
                    style={
                        'backgroundColor': '#28a745', 'color': 'white', 'fontSize': '16px',
                        'borderRadius': '5px', 'border': 'none', 'padding': '10px 20px'
                    }
                ),
            ]
        ),
        html.Div(id='output-result', style={'text-align': 'center', 'padding': '20px', 'color': '#007BFF', 'fontSize': '20px'})
    ]
)

@app.callback(
    Output('output-result', 'children'),
    [Input('classify-btn', 'n_clicks')],
    [State('input-feature1', 'value'),
     State('input-feature2', 'value'),
     State('input-feature3', 'value'),
     State('input-feature4', 'value'),
     State('input-feature5', 'value'),
     State('input-feature6', 'value'),
     State('input-feature7', 'value'),
     State('input-feature8', 'value'),
     State('input-feature9', 'value')]
)
def classify_tumor(n_clicks, feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9):
    if n_clicks > 0:
        # Ensure all features are provided
        if None in [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9]:
            return "Please provide all features before classifying."
        
        # Prepare the feature vector
        features = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9]])  
        features_std = scaler.transform(features)
        prediction = model.predict(features_std)
        
        # Classify based on the predicted value
        if prediction[0] == 2:
            result = 'Benign'
        elif prediction[0] == 4:
            result = 'Malignant'
        else:
            result = 'Unknown classification'

        return f'The tumor is classified as: {result}'
    return ''

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)
