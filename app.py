# importing dash library, pandas for dataframe, pickle to use trained model or save model, numpy for array or matrix operations
from dash import Dash, html, dcc, Input, Output, State, dash_table
import pandas as pd
import pickle
import numpy as np

app = Dash()    #initialization of app using Dash

df = pd.read_csv("assets/data/Cars.csv")    # reading csv file using pandas

filename = 'model/carPricePrediction.model' # importing pre-trained model carPricePrediction.model
loaded_model = pickle.load(open(filename, 'rb'))    # loading trained model to loaded_model variable

app.layout = html.Div(children=[    #creating main app layout and using its children class as subclass
    html.H1(children='Welcome to Machine Learning'), # Main heading h1
    html.H2(children='Car Price Prediction'),

    html.Div("Car Price Prediction (CPP): A web application framework for predicting Car Prices."), # creating div


    html.Div([  # creating button inside div to easily work with UI providing style
        html.Button("DataFrame", id="dataFrame",
                    style={"background-color": "blue", "color": "white", "padding": "10px",
                        "border-radius": "10px", "margin-right": '10px', "cursor": "pointer","height": "40px"}),

        html.Button("Instructions", id="instruction",
                    style={"background-color": "blue", "color": "white", "padding": "10px",
                        "border-radius": "10px", "cursor": "pointer","height": "40px"}),
    ],style={"display": "flex","justify-content":"space-between","margin-top": "10px"}),

    html.Div(id="table-container"),
    dcc.Store(id="table-visible", data=False),  # using dash core component (dcc) for component to store data to browser
    
    html.Div(id="instruction-container"),
    dcc.Store(id="instruction-visible", data=False),  
    
    html.Div([
    html.Div([
    html.Label(["Car Brand Name:"],style={"fontWeight": "bold"}),   # using label for dropdown 
    dcc.Dropdown(
        id="brand-input",
        options=[   # Provided with Brand names and value as per LabelEncoder in A_Z ascending order
            {"label": "Ambassador", "value": 0},
            {"label": "Ashok", "value": 1},
            {"label": "Audi", "value": 2},
            {"label": "BMW", "value": 3},
            {"label": "Chevrolet", "value": 4},
            {"label": "Daewoo", "value": 5},
            {"label": "Datsun", "value": 6},
            {"label": "Fiat", "value": 7},
            {"label": "Force", "value": 8},
            {"label": "Ford", "value": 9},
            {"label": "Honda", "value": 10},
            {"label": "Hyundai", "value": 11},
            {"label": "Isuzu", "value": 12},
            {"label": "Jaguar", "value": 13},
            {"label": "Jeep", "value": 14},
            {"label": "Kia", "value": 15},
            {"label": "Land", "value": 16},
            {"label": "Lexus", "value": 17},
            {"label": "MG", "value": 18},
            {"label": "Mahindra", "value": 19},
            {"label": "Maruti", "value": 20},
            {"label": "Mercedes-Benz", "value": 21},
            {"label": "Mitsubishi", "value": 22},
            {"label": "Nissan", "value": 23},
            {"label": "Opel", "value": 24},
            {"label": "Peugeot", "value": 25},
            {"label": "Renault", "value": 26},
            {"label": "Skoda", "value": 27},
            {"label": "Tata", "value": 28},
            {"label": "Toyota", "value": 29},
            {"label": "Volkswagen", "value": 30},
            {"label": "Volvo", "value": 31}
        ],
        placeholder="Select Brand",
        style={"margin-bottom": "10px","width": "22rem"}
        ),
    ],style={"margin-top": "1rem"}),
    
    html.Div([
        html.Label(["Kms Driven:"],style={"fontWeight": "bold"}),
        dcc.Input(id="km", type="number", placeholder="Enter Km driven by Car", debounce=True,
                  style={"margin-bottom": "10px", "display": "block","width": "22rem","padding": "6px"}),
    ]),

    html.Div([
        html.Label(["Fuel Type:"],style={"fontWeight": "bold"}),
        dcc.Dropdown(
            id="fuel-input",
            options=[
            {"label": "Diesel", "value": 0},
            {"label": "Petrol", "value": 1}
            ],
            placeholder="Select fuel",
            style={"margin-bottom": "10px","width": "22rem"}
        ),
    ]),

    html.Div([
        html.Label(["Seller Type:"],style={"fontWeight": "bold"}),
        dcc.Dropdown(
            id="seller-type-input",  
            options=[
            {"label": "Dealer", "value": 0},
            {"label": "Individual", "value": 1},
            {"label": "Trustmark Dealer", "value": 2}
            ],
            placeholder="Select Seller Type",
            style={"margin-bottom": "10px","width": "22rem"}
        ),
    ]),

    html.Label(["Mileage (kmpl):"],style={"fontWeight": "bold"}),
    dcc.Input(id="mileage", type="number", placeholder="Enter mileage", debounce=True,  # debounce to not run after each and everytime user make slight change in input space
                style={"margin-bottom": "10px", "display": "block","width": "22rem","padding": "6px"}),     # Taking user input and passing to app callback through id

    html.Label(["Engine (CC):"],style={"fontWeight": "bold"}),
    dcc.Input(id="engine", type="number", placeholder="Enter engine capacity", debounce=True,
                style={"margin-bottom": "10px", "display": "block","width": "22rem","padding": "6px"}),

    html.Label(["Seats/Capacity:"],style={"fontWeight": "bold"}),
    dcc.Input(id="seats", type="number", placeholder="Enter number of seats", debounce=True,
                style={"margin-bottom": "10px", "display": "block","width": "22rem","padding": "6px"}),

    html.Label(["Max Power (bhp):"],style={"fontWeight": "bold"}),
    dcc.Input(id="max_power", type="number", placeholder="Enter max power", debounce=True,
                style={"margin-bottom": "10px", "display": "block","width": "22rem","padding": "6px"}),
    ],style={"margin-left": "0.2rem"}),
    html.Button("Predict", id="prediction",
                style={"background-color": "blue", "color": "white", "padding": "10px",
                       "border-radius": "10px", "cursor": "pointer" }),

    html.Div(id="prediction-container"),
    dcc.Store(id="prediction-visible", data=False)
], style={"padding": "1rem"})


@app.callback(  # calling back id to change state provider and toggle button clicked 
    Output("table-container", "children"),
    Output("table-visible", "data"),
    Input("dataFrame", "n_clicks"),
    State("table-visible", "data"),  
    prevent_initial_call=True
)

def show_dataframe(n_clicks, visible):  # function to show dataFrame on click
    if not visible:
        return dash_table.DataTable(
            columns=[{"name": "Brand", "id": "name"},
                     {"name": "Fuel", "id": "fuel"},
                     {"name": "Max Power", "id": "max_power"},
                     {"name": "Kms Driven", "id": "km_driven"},
                     {"name": "Engine", "id": "engine"},
                     {"name": "Mileage", "id": "mileage"},
                     {"name": "Seats", "id": "seats"},
                     {"name": "Selling Price", "id": "selling_price"}],
            data=df[["name","fuel","max_power","km_driven","engine","mileage","seats", "selling_price"]].to_dict('records'),
            page_size=10,
            style_table={"width": "98%"}
        ), True
    else:
        return "", False    # checking click and not clicked through boolean


@app.callback(
    Output("instruction-container", "children"),
    Output("instruction-visible", "data"),
    Input("instruction", "n_clicks"),
    State("instruction-visible", "data"),
    prevent_initial_call=True
)
def toggleInstruction(n_clicks, visible):  
    if not visible:
        return html.Div(["This is a Car Price Prediction Software to Predict as per your NEED and WANT. " ,
            html.Br(),
            "DataFrame button shows the car data prices.",
            html.Br(),
            "Whereas Form below is required to be filled to get result of Price Prediction.",
            html.Br(),
            "Just hit PREDICT and Predicted Price will appear."],
                        style={"border": "2px solid red", "border-radius": "20px","margin-left": "10px","margin-right":"10px", "padding-left":"10px","padding-right":"10px","padding":"6px"}), True
    else:
        return "", False

@app.callback(
    Output("prediction-container", "children"),
    Output("prediction-visible", "data"),
    Input("prediction", "n_clicks"),
    State("brand-input", "value"),     
    State("km", "value"),
    State("fuel-input", "value"),
    State("seller-type-input", "value"),
    State("mileage", "value"),
    State("engine", "value"),
    State("seats", "value"),
    State("max_power", "value"),
    State("prediction-visible", "data"),
    prevent_initial_call=True
)
def prediction_fun(n_clicks, brand, km, fuel, seller, mileage, engine, seats, max_power, visible):  # prediction function for predicting price
    if not visible:
        sample = np.array([[brand, km, fuel, seller, mileage, engine, seats, max_power]])

        predicted_price = loaded_model.predict(sample)[0]
        return f"Predicted Car Price: {predicted_price:,.0f}", True
    else:
        return "", False

if __name__ == '__main__':      # main app running file for running server default runs on localhost= 127.0.0.1 port 8050 .i.e. http://127.0.0.1:8050/
    app.run(debug=True)
