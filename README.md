# Lateral Flow Assay Regression Model
This program is designed to work with a handheld reader that measures the saturation values of the control and test lines in lateral flow assays. The model is trained on a dataset that includes known protein concentrations and the corresponding measurements from the reader’s test and control lines. In this dataset, the earlier columns represent the features (the measured values from the reader), while the final column contains the target outcome (the protein concentration). The data is processed using Pandas and NumPy. Ridge regression, a machine learning technique, is then applied to build a predictive model. This model is used to estimate the concentration of a specific protein based on the user's input of their test and control line values from their lateral flow assay.

# How Does Ridge Regression Work?
Ridge regression is a technique that helps build a predictive model by using data loaded from a CSV file, which is organized into a dataframe with the help of Pandas. The data is divided into two subsets: the x subset contains the features (which for this model are the values from the test and control lines), while the y subset contains the target outcome (in this case, the concentration of the protein neutrophil elastase).


To train and test the model, we use the function ‘load_and_split_data,’ which separates the data into training and testing sets for both x and y. The ratio of the split is controlled by the ‘test_size’ variable, which defaults to 0.2, meaning 20% of the data is used for testing, and 80% for training.


Ridge regression differs from linear regression in that it can handle situations where the independent variables (features) are highly correlated. It achieves this by using a regularization parameter, (that we refer to as lambda), which helps prevent overfitting by penalizing large coefficients. When the function runs, it returns two key outputs: th, which is an array of coefficients for the model, and th0, which represents the intercept or offset of the best-fit line.


This approach allows us to build a model that can accurately predict the concentration of neutrophil elastase based on new input values from the test and control lines.

# Graphical User Interface
This program uses tkinter to support a user-friendly interface. When the script lat_flow_ml.py is run, a window pops up which asks the user to put in the resulst from the lateral flow assay reader. It then prints out an approximation of protein concentration in ng/mL, which can be used to decide what next steps might be needed. 
