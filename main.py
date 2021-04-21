import pandas as pd
from sklearn.model_selection import train_test_split
import statistics
import numpy
from sklearn.linear_model import LinearRegression


# Regression Model
def simple_linear_regression(input_feature, output,input_test,output_test):
    # Computing sums needed to calculate slope and intercept
    regr = LinearRegression()
    regr.fit(input_feature, output)
    intercept = regr.intercept_
    slope = regr.coef_
    print("Residual sum of squares Train: %.2f"
          % ((regr.predict(input_feature) - output) ** 2).sum())
    print("Residual sum of squares Test: %.2f"
          % ((regr.predict(input_test) - output_test) ** 2).sum())
    return intercept, slope


# FUNCTIONS
def divide_train_test(data):
    X_train_model, X_test_model, y_train_model, y_test_model = train_test_split(data,
                                                                                kc_house_data["price"],
                                                                                test_size=0.2,
                                                                                random_state=0,
                                                                                )
    return X_train_model, X_test_model, y_train_model, y_test_model


# DATA
dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float, 'grade': int,
              'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float,
              'sqft_lot15': float, 'sqft_living': float, 'floors': str, 'condition': int, 'lat': float, 'date': str,
              'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}
kc_house_data = pd.read_csv("kc_house_data.csv", dtype=dtype_dict)
kc_house_test_data = pd.read_csv("kc_house_test_data.csv", dtype=dtype_dict)
kc_house_train_data = pd.read_csv("kc_house_train_data.csv", dtype=dtype_dict)
# ADD 4 VARIABLES
bedrooms_squared = kc_house_data['bedrooms'] * kc_house_data['bedrooms']
bed_bath_rooms = kc_house_data['bedrooms'] * kc_house_data['bathrooms']
log_sqft_living = numpy.log(kc_house_data['sqft_living'])
lat_plus_long = kc_house_data["lat"] + kc_house_data['long']
# ADD MORE COLUMNS
kc_house_data['bedrooms_squared'] = bedrooms_squared
kc_house_data['bed_bath_rooms'] = bed_bath_rooms
kc_house_data['log_sqft_living'] = log_sqft_living
kc_house_data['lat_plus_long'] = lat_plus_long

# QUIZ QUESTION
bedrooms_squared_Mean = statistics.mean(bedrooms_squared)
bed_bath_rooms_Mean = statistics.mean(bed_bath_rooms)
log_sqft_living_Mean = statistics.mean(log_sqft_living)
lat_plus_long_Mean = statistics.mean(lat_plus_long)
print("bedrooms_squared_Mean is ", bedrooms_squared_Mean)
print("bed_bath_rooms_Mean is", bed_bath_rooms_Mean)
print("log_sqft_living_Mean is", log_sqft_living_Mean)
print("lat_plus_long_Mean is", lat_plus_long_Mean)
# Split training and test

# features
model_features_one = ["sqft_living", "bedrooms", "bathrooms", "lat", "long"]
model_features_two = ["sqft_living", "bedrooms", "bathrooms", "lat", "long", "bed_bath_rooms"]
model_features_three = ["sqft_living", "bedrooms", "bathrooms", "lat", "long", "bed_bath_rooms", "bedrooms_squared",
                        "log_sqft_living", "lat_plus_long"]
df_1 = pd.DataFrame(kc_house_data, columns=model_features_one)
df_2 = pd.DataFrame(kc_house_data, columns=model_features_two)
df_3 = pd.DataFrame(kc_house_data, columns=model_features_three)
# TRAIN TEST
x_train_1, x_test_1, y_train_1, y_test_1 = divide_train_test(df_1)
x_train_2, x_test_2, y_train_2, y_test_2 = divide_train_test(df_2)
x_train_3, x_test_3, y_train_3, y_test_3 = divide_train_test(df_3)

# linear regression model
intercept1, slope1 = simple_linear_regression(x_train_1, y_train_1,x_test_1,y_test_1)
intercept2, slope2 = simple_linear_regression(x_train_2, y_train_2,x_test_2,y_test_2)
intercept3, slope3 = simple_linear_regression(x_train_3, y_train_3,x_test_3,y_test_3)
print("MODEL1 ###########################")
print(intercept1)
print(slope1)
#
print("MODEL2 ###########################")
print(intercept2)
print(slope2)
#
print("MODEL3 ###########################")
print(intercept3)
print(slope3)
