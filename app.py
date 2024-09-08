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

# Define the layout with added colors, styling, and an image
app.layout = html.Div(
    style={'backgroundColor': '#f0f8ff', 'fontFamily': 'Arial, sans-serif'},
    children=[
        html.H1(
            "Tumor Classification",
            style={'text-align': 'center', 'color': 'white', 'padding': '20px', 'backgroundColor': 'black'}
        ),
        # Adding a placeholder image related to medical data
        html.Div(
            children=[
                html.Img(
                    src='https://www.istockphoto.com/photos/tumor', 
                    style={'width': '50%', 'display': 'block', 'margin': 'auto'}
                )
            ]
        ),
        html.Div(
            style={'text-align': 'center', 'padding': '30px'},
            children=[
                # Input fields for all the tumor features
                *[html.Div([
                    html.Label(f"Feature {i+1}:", style={'font-weight': 'bold'}),
                    dcc.Input(id=f'input-feature{i+1}', type='number', placeholder=f'Enter Feature {i+1}', min=0, step=1,
                              style={'margin': '10px', 'padding': '5px'}),
                ], style={'display': 'inline-block', 'width': '20%'}) for i in range(9)],
                
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
        
        # Print for debugging
        print("Received features:", feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9)
        
        # Prepare the feature vector
        features = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9]])  
        features_std = scaler.transform(features)
        print("Transformed features:", features_std)  # Print the standardized features for debugging
        
        prediction = model.predict(features_std)
        print("Prediction:", prediction)  # Check what the model is predicting
        
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
