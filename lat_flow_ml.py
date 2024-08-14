import pandas as pd
import numpy as np
import tkinter as tk

# load data from CSV
data = pd.read_csv('lft_data.csv', header=0)

# shuffle data
data = data.sample(frac=1, random_state=0).reset_index(drop=True)

# x and y arrays
x = data.iloc[:, :-1].values
y = np.array([i**(1/2) for i in data.iloc[:, -1].values])

#functions to help perform later calculations
def make_splits(X, Y, n_splits):
    '''
    Splits the dataset into n_split chunks, creating n_split sets for
    cross-validation later on.
    The tuples are in the format (X_train, Y_train, X_test, Y_test).
    For the ith returned tuple:
    * X_train and Y_train include all data except the ith chunk, and
    * X_test and Y_test are the ith chunk.
    
    Args:
        X : d x n numpy array (d = #features, n = #data points)
        Y : 1 x n numpy array
        n_splits : integer
    '''
    
    d, n = X.shape
    split_size = n // n_splits  # Size of each chunk
    splits = []
    
    for i in range(n_splits):
        # Indices for the test set
        test_indices = np.arange(i * split_size, (i + 1) * split_size)
        
        # Handle the case where n is not perfectly divisible by n_splits
        if i == n_splits - 1:  # Last split
            test_indices = np.arange(i * split_size, n)
        
        # Indices for the training set
        train_indices = np.setdiff1d(np.arange(n), test_indices)
        
        # Create training and test sets
        X_train, Y_train = X[:, train_indices], Y[:, train_indices]
        X_test, Y_test = X[:, test_indices], Y[:, test_indices]
        
        # Append the tuple to the splits list
        splits.append((X_train, Y_train, X_test, Y_test))
    
    return splits

def ridge_analytic(X_train, Y_train, lam):
    '''
    Applies analytic ridge regression on the given training data.
    Returns th, th0.
    
    Args:
        X_train : d x n numpy array (d = # features, n = # data points)
        Y_train : 1 x n numpy array
        lam : (float) regularization strength parameter
    
    Returns:
        th : d x 1 numpy array
        th0 : 1 x 1 numpy array
    '''
    d, n = X_train.shape
    
    # Adding a row of ones to X_train for the intercept term
    X_train_augmented = np.vstack([X_train, np.ones((1, n))])
    
    # Creating the augmented version of th
    I_augmented = np.eye(d + 1)
    I_augmented[-1, -1] = 0  # No regularization for the intercept term
    
    # Augmented version of X_train and Y_train
    XtX = np.dot(X_train_augmented, X_train_augmented.T)
    XtY = np.dot(X_train_augmented, Y_train.T)
    
    # Compute the ridge regression weights
    th_augmented = np.linalg.solve(XtX + lam * I_augmented, XtY)
    
    # Separate the intercept term from the weight vector
    th = th_augmented[:-1]
    th0 = th_augmented[-1].reshape(1, 1)
    
    return th, th0

def mse(x, y, th, th0):
    '''
    Calculates the mean-squared loss of a linear regression.
    Returns a scalar.
    
    Args:
        x : d x n numpy array
        y : 1 x n numpy array
        th : d x 1 numpy array
        th0 : 1 x 1 numpy array
    
    Returns:
        error: (float) mean-squared loss
    '''
    n = x.shape[1]
    # Predicted values
    y_pred = np.dot(th.T,x) + th0
    # Mean Squared Error
    error = np.mean((y - y_pred) ** 2)
    
    return error

# def cross_validate(X, Y, n_splits, lam, learning_algorithm, loss_function):
#     '''
#     Splitting data into n_splits different groups and generating mean loss
#     across the entire dataset using our learning algorithm. 
    
#     Args:
#         x : d x n numpy array
#         y : 1 x n numpy array
#         lam : (float) regularization strength parameter
#         learning_algorithm: function to generate theta/theta_0 values for our dataset
#         loss_function: function to generate errors from our model
    
#     Returns:
#         mean error from our cross-validation sets
        
#     '''
#     test_errors = []
#     for (X_train, Y_train, X_test, Y_test) in make_splits(X, Y, n_splits):
#         th, th0 = learning_algorithm(X_train, Y_train, lam)
#         test_errors.append(loss_function(X_test, Y_test, th, th0))
#     return f'test error: {np.array(test_errors).mean()}'

# FUNCTIONS TO MAKE A POLYNOMIAL MODEL START HERE
# def create_polynomial_features(X):
#     '''
#     Transforms the input features to include polynomial terms up to degree 2.
    
#     Args:
#         X : d x n numpy array (d = # features, n = # data points)
    
#     Returns:
#         X_poly : p x n numpy array (p = transformed feature dimension)
#     '''
#     d, n = X.shape
    
#     # Start with the original features
#     features = [X]
    
#     # Add square terms
#     for i in range(d):
#         features.append(X[i:i+1, :] ** 2)
    
#     # Add interaction terms
#     for i in range(d):
#         for j in range(i + 1, d):
#             features.append(X[i:i+1, :] * X[j:j+1, :])
    
#     # Combine all features into one array
#     X_poly = np.vstack(features)
    
#     return X_poly

# def ridge_analytic_poly(X_train, Y_train, lam):
#     '''
#     Applies analytic ridge regression on the given training data for degree 2 polynomial features.
#     Returns th, th0.

#     Args:
#         X_train : d x n numpy array (d = # features, n = # data points)
#         Y_train : 1 x n numpy array
#         lam : (float) regularization strength parameter
    
#     Returns:
#         th : p x 1 numpy array (p = transformed feature dimension)
#         th0 : 1 x 1 numpy array
#     '''
#     # Create polynomial features
#     X_train_poly = create_polynomial_features(X_train)
    
#     # Apply ridge regression on the polynomial features
#     d_poly, n = X_train_poly.shape
#     X_train_augmented = np.vstack([X_train_poly, np.ones((1, n))])
    
#     I_augmented = np.eye(d_poly + 1)
#     I_augmented[-1, -1] = 0  # No regularization for the intercept term
    
#     XtX = np.dot(X_train_augmented, X_train_augmented.T)
#     XtY = np.dot(X_train_augmented, Y_train.T)
    
#     th_augmented = np.linalg.solve(XtX + lam * I_augmented, XtY)
    
#     th = th_augmented[:-1]
#     th0 = th_augmented[-1].reshape(1, 1)
    
#     return th, th0

# def mse_poly(x, y, th, th0):
#     '''
#     Calculates the mean-squared loss of a linear regression with polynomial features.
#     Returns a scalar.

#     Args:
#         x : d x n numpy array
#         y : 1 x n numpy array
#         th : p x 1 numpy array (p = transformed feature dimension)
#         th0 : 1 x 1 numpy array
#     '''
#     # Create polynomial features
#     x_poly = create_polynomial_features(x)
    
#     n = x_poly.shape[1]
#     y_pred = np.dot(th.T, x_poly) + th0
    
#     error = np.mean((y - y_pred) ** 2)
    
#     return error
# FUNCTIONS TO MAKE A POLYNOMIAL MODEL END HERE

def load_and_split_data(file_path, test_size=0.2, random_state=None):
    '''
    Loads data from a CSV file and splits it into training and testing sets.
    
    Args:
        file_path (str): Path to the CSV file
        test_size (float): Proportion of the data to be used as the test set (default is 0.2)
        random_state (int): Seed for random number generator (default is None)
    
    Returns:
        X_train, X_test, Y_train, Y_test (DataFrames/Series): Training and testing data
    '''
    # load data from CSV
    data = pd.read_csv(file_path)
    
    # shuffle data
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # split data
    n_test = int(len(data) * test_size)
    test_data = data[:n_test]
    train_data = data[n_test:]
    
    # generating training and testing sets for the test/control line values. 
    # to improve the fit of the model, this takes the square root of the neutrophil elastace concentrations in the last column.
    X_train = train_data.iloc[:, :-1].values
    X_test = test_data.iloc[:, :-1].values
    Y_train = np.array([i**(1/2) for i in train_data.iloc[:, -1].values])
    Y_test = np.array([i**(1/2) for i in test_data.iloc[:, -1].values])
    
    return X_train.T, X_test.T, Y_train.T, Y_test.T

# checking different lambda values to see which one has the lowest corresponding error
lams = [0, 0.01, 0.02, 0.1]
# errors = [cross_validate(x, y, 4, lam, ridge_analytic, mse) for lam in lams]

# EXAMPLE USAGE HERE
file_path = 'lft_data.csv'  # replace with the path to csv file containing test, control, and concentration.
test_size = 0.2  # 20% of the data will be used as the test set
random_state = 50  # set a seed for reproducibility (can be any number)

# splitting our data into training and testing sets
X_train, X_test, Y_train, Y_test = load_and_split_data(file_path, test_size, random_state)

# generate the theta and theta_naught values that we will use in our model
# lambda values changes depending on what gave the least amount of testing error from earlier trials.
th, th0 = ridge_analytic(X_train, Y_train, 0)
test_errors = mse(X_test, Y_test, th, th0)

# assigning our theta values to their different variables (i.e. for the test line value, control line value, and offset)
th_t = th[0]
th_c = th[1]
th0 = th0[0]

# printing out the test error that our model generates.
# print(cross_validate(x,y,4,0, ridge_analytic, mse))

# using tkinter to generate a GUI, where the user puts in their test/control values and gets estimated concentration:
def calc_concentration():
    try:
        test = float(entry1.get())
        ctrl = float(entry2.get())
        
        # estimating teh amount of neutrophil elastace (ng/ml)
        res = (th_t*test + th_c*ctrl + th0[0])**2
        
        if res < 0:
            res=0
            newline = None
        if res < 75:
            descript = "Minimal levels of neutrophil elastase detected."
            newline = None
        elif res < 500:
            descript = "Some concentratoin of neutrophil elastase detected."
            newline =  'Please continue tracking your symptoms and results.'
        elif res < 1400:
            descript = "Moderately high concentration of neutrophil elastase detected."
            newline = 'Please reach out to your care provider and continue to monitor symptoms and your results.'
        else:
            descript = "Very high levels of neutrophil elastase detected."
            newline = 'Please reach out to your care provider.'
        
        res_label.config(text=f"Concentration: {res}")
        descript_label.config(text=f'Feedback: {descript}')
        last.config(text=newline)
        
    except ValueError:
        res_label.config(text="Invalid Input.")

root = tk.Tk()
root.title("Neutrophil Elastace Calculator")

# place widgets
tk.Label(root, text="Enter Test Line Value").grid(row=0, column=0, padx=10, pady=10, sticky='w')
entry1=tk.Entry(root)
entry1.grid(row=0, column=1, padx=10, pady=10)

tk.Label(root, text="Enter Control Line Value").grid(row=1, column=0, padx=10, pady=10, sticky='w')
entry2=tk.Entry(root)
entry2.grid(row=1, column=1, padx=10, pady=10)

calc_button = tk.Button(root, text="Calculate", command=calc_concentration)
calc_button.grid(row=1, column=3, padx=20, pady=10, sticky='w')

res_label = tk.Label(root, text="Concentration: ")
res_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky='w')

descript_label = tk.Label(root, text="Feedback: ")
descript_label.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky='w')

last = tk.Label(root)
last.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky='w')

root.mainloop()