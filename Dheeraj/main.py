import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load your datasets
data1 = pd.read_csv(r"C:\Users\DELL\Desktop\Dheeraj\collegePlace.csv")
data2 = pd.read_csv(r"C:\Users\DELL\Desktop\Dheeraj\Train_Data.csv")

# # Assuming you want to concatenate data1 and data2 horizontally
# df = pd.concat([data1, data2], axis=1)

# # List of selected features
# features = ["Gender", "Stream", "Internships", "CGPA", "HistoryOfBacklogs", "ssc_p", "hsc_p", "etest_p"]

# # Replace categorical values with numerical values
# df['Gender'].replace(['Male', 'Female'], [0, 1], inplace=True)

# # Create a DataFrame with selected features
# featured_data = df[features]

# # Handle missing values by randomly filling them
# for column in featured_data.columns:
#     mask = featured_data[column].isna()
#     non_nan_values = featured_data.loc[~mask, column]
#     random_values = np.random.choice(non_nan_values, size=sum(mask), replace=True)
    # featured_data.loc[mask, column] = random_values

# Initialize LabelEncoders for each categorical column
col = ["gender", "degree_t", "workex", "ssc_b", 'hsc_b', 'specialisation', "hsc_s"]
mat=data2.drop(["mba_p","status"],axis=1)
label_encoders = {}

for c in col:
    le = LabelEncoder()  # Initialize LabelEncoder for each column
    mat[c] = le.fit_transform(mat[c])
    label_encoders[c] = le

# Define feature (X) and target (y)
y_mat = le.fit_transform(data2["status"])

# Create and train a Random Forest Regressor model
linear_reg =LogisticRegression(solver='liblinear', max_iter=1000, multi_class='auto', verbose=1, n_jobs=-1, random_state=0, warm_start=True, l1_ratio=0.5)
linear_reg.fit(mat,y_mat)

y_pred = linear_reg.predict([[1, 70, 1, 60, 1, 2, 75, 2, 90, 1, 0]])

# Save the trained model to a pickle file
with open('your_model.pkl', 'wb') as model_file:
    pickle.dump(linear_reg, model_file)
