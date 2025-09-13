from dash import html, dcc
# import dash_table
import pandas as pd
import pickle
import numpy as np

layout = html.Div([
    html.H2("Car Price Prediction: Version 2"),
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


