import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import joblib
from flask import Flask

# Load the trained model and scaler
model = joblib.load('svm_model.pkl')  # Ensure the correct paths
scaler = joblib.load('scaler.pkl')

# Create the Flask server
server = Flask(__name__)

# Create the Dash app
app = dash.Dash(__name__, server=server)

# Define the actual feature names
feature_names = [
    'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 
    'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses'
]

# Define the layout with attractive design
app.layout = html.Div(
    style={
        'backgroundImage': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTdujqkM90d_D1_Mx5o9O27ytgKm6hnuXLHXg&s',  # Background image
        'backgroundSize': 'cover',
        'height': '100vh',
        'padding': '20px',
        'fontFamily': 'Arial, sans-serif',
        'color': 'white'
    },
    children=[
        html.H1(
            "Tumor Classification",
            style={
                'text-align': 'center',
                'color': '#ffcc00',
                'padding': '20px',
                'backgroundColor': 'rgba(0, 0, 0, 0.7)',  # Translucent background for better text visibility
                'borderRadius': '10px',
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.5)'
            }
        ),
        html.Div(
            style={'text-align': 'center', 'padding': '30px', 'backgroundColor': 'rgba(0, 0, 0, 0.5)', 'borderRadius': '10px'},
            children=[
                # Input fields for all the tumor features with updated names
                *[html.Div([
                    html.Label(f"{feature_names[i]}:", style={'font-weight': 'bold', 'fontSize': '18px', 'color': '#ffcc00'}),
                    dcc.Input(
                        id=f'input-feature{i+1}', 
                        type='number', 
                        placeholder=f'Enter {feature_names[i]}', 
                        min=0, 
                        step=1,
                        style={
                            'margin': '10px', 
                            'padding': '10px', 
                            'width': '80%', 
                            'borderRadius': '5px', 
                            'border': '2px solid #28a745',
                            'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.3)',
                        }),
                ], style={'display': 'inline-block', 'width': '20%'}) for i in range(9)],
                
                html.Br(),
                html.Button(
                    'Classify Tumor', 
                    id='classify-btn', 
                    n_clicks=0, 
                    style={
                        'backgroundColor': '#28a745',
                        'color': 'white',
                        'fontSize': '18px',
                        'borderRadius': '5px',
                        'border': 'none',
                        'padding': '15px 30px',
                        'cursor': 'pointer',
                        'transition': 'all 0.3s ease'
                    },
                    # Add hover effect using inline style
                    n_clicks_timestamp=0
                ),
                html.Br(),
                html.Div(
                    id='output-result', 
                    style={
                        'text-align': 'center', 
                        'padding': '20px', 
                        'color': '#ffcc00', 
                        'fontSize': '22px',
                        'marginTop': '20px',
                        'backgroundColor': 'rgba(0, 0, 0, 0.7)',
                        'borderRadius': '10px',
                        'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.5)'
                    }
                ),
            ]
        )
    ]
)

@app.callback(
    Output('output-result', 'children'),
    [Input('classify-btn', 'n_clicks')],
    [State(f'input-feature{i+1}', 'value') for i in range(9)]
)
def classify_tumor(n_clicks, *features):
    if n_clicks > 0:
        # Ensure all features are provided
        if None in features:
            return "Please provide all features before classifying."
        
        # Prepare the feature vector
        features_array = np.array([features])  
        features_std = scaler.transform(features_array)
        
        prediction = model.predict(features_std)
        
        # Classify based on the predicted value
        if prediction[0] == 2:
            result = 'Benign'
        elif prediction[0] == 4:
            result = 'Malignant'
        else:
            result = 'Unknown classification'

        return f'The Tumor is classified as: {result}'
    return ''

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)
