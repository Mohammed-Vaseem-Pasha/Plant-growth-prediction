# ================================================================
# USING DEEP LEARNING TO PREDICT PLANT GROWTH AND YIELD
# IN THE GREENHOUSE ENVIRONMENTS
#
# main.py
#
# Modules:
#   - Tkinter GUI
#   - Pandas
#   - NumPy
#   - Scikit-learn
#   - TensorFlow / Keras
#   - Matplotlib
#
# Workflow:
#   1. Upload Ficus Plant Dataset
#   2. Dataset Preprocess, Clean & Train Test Split
#   3. Run SVR Algorithm
#   4. Run Random Forest Algorithm
#   5. Run LSTM Algorithm
#   6. Upload Test Dataset / Predict Growth
#   7. MAE Graph
#   8. MSE Graph
#   9. RMSE Graph
# ================================================================

import tkinter
from tkinter import *
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os

from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# ================================================================
# GLOBAL VARIABLES
# ================================================================

filename = ""
train = None

X_train = None
X_test = None
y_train = None
y_test = None

svr_model = None
rf_model = None
lstm_model = None

classifier = None

svr_mae = 0
svr_mse = 0
svr_rmse = 0

rf_mae = 0
rf_mse = 0
rf_rmse = 0

lstm_mae = 0
lstm_mse = 0
lstm_rmse = 0

# Original dataset structure uses:
# First 7 columns = input features
# 8th column = target
NUMBER_OF_FEATURES = 7


# ================================================================
# HELPER FUNCTION
# ================================================================

def show_message(message):
    """
    Display a message inside the output text box.
    """
    text.delete("1.0", END)
    text.insert(END, message)


def check_dataset():
    """
    Check whether a dataset has been uploaded.
    """
    if train is None:
        messagebox.showwarning(
            "Dataset Required",
            "Please upload the Ficus Plant Dataset first."
        )
        return False

    return True


def check_training_data():
    """
    Check whether preprocessing has been completed.
    """
    if X_train is None or X_test is None:
        messagebox.showwarning(
            "Preprocessing Required",
            "Please run Dataset Preprocess, Clean & Train Test Split first."
        )
        return False

    return True


# ================================================================
# 1. UPLOAD DATASET
# ================================================================

def upload():
    global filename
    global train

    try:

        filename = filedialog.askopenfilename(
            initialdir="dataset",
            title="Select Ficus Plant Dataset",
            filetypes=[
                ("CSV Files", "*.csv"),
                ("Text Files", "*.txt"),
                ("All Files", "*.*")
            ]
        )

        if filename == "":
            return

        # Read CSV file
        train = pd.read_csv(filename)

        # Check number of columns
        if train.shape[1] < 8:
            messagebox.showerror(
                "Invalid Dataset",
                "Dataset must contain at least 8 columns.\n\n"
                "First 7 columns = input features\n"
                "8th column = growth/yield target"
            )

            train = None
            return

        # Convert columns to numeric where possible
        for column in train.columns:
            train[column] = pd.to_numeric(
                train[column],
                errors="coerce"
            )

        # Replace missing values with column means
        train = train.fillna(train.mean(numeric_only=True))

        # If any non-numeric columns remain, try filling them
        train = train.dropna()

        pathlabel.config(
            text="Selected Dataset: " + os.path.basename(filename)
        )

        text.delete("1.0", END)

        text.insert(
            END,
            "Ficus Plant Dataset Loaded Successfully\n\n"
        )

        text.insert(
            END,
            "Dataset File : " + filename + "\n"
        )

        text.insert(
            END,
            "Total records : " + str(len(train)) + "\n"
        )

        text.insert(
            END,
            "Total columns : " + str(train.shape[1]) + "\n\n"
        )

        text.insert(
            END,
            "First 7 columns will be used as input features.\n"
        )

        text.insert(
            END,
            "8th column will be used as growth/yield target.\n"
        )

    except Exception as e:

        train = None

        messagebox.showerror(
            "Error",
            "Unable to load dataset.\n\n" + str(e)
        )


# ================================================================
# 2. DATASET PREPROCESSING
# ================================================================

def cleanDataset():

    global X_train
    global X_test
    global y_train
    global y_test

    if not check_dataset():
        return

    try:

        # First 7 columns = input
        X = train.iloc[:, 0:7].values

        # 8th column = target
        Y = train.iloc[:, 7].values

        # Make sure values are numeric
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        # Split 80% training and 20% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            Y,
            test_size=0.2,
            random_state=0
        )

        text.delete("1.0", END)

        text.insert(
            END,
            "Data cleaning and preprocessing completed\n\n"
        )

        text.insert(
            END,
            "Total dataset records : "
            + str(len(train))
            + "\n\n"
        )

        text.insert(
            END,
            "Training records : "
            + str(len(X_train))
            + "\n"
        )

        text.insert(
            END,
            "Testing records : "
            + str(len(X_test))
            + "\n\n"
        )

        text.insert(
            END,
            "Train/Test Split : 80% / 20%\n\n"
        )

        text.insert(
            END,
            "Input features : 7\n"
        )

        text.insert(
            END,
            "Target column : 8th column\n"
        )

    except Exception as e:

        messagebox.showerror(
            "Preprocessing Error",
            str(e)
        )


# ================================================================
# 3. SVR ALGORITHM
# ================================================================

def SVR():

    global svr_mae
    global svr_mse
    global svr_rmse
    global svr_model

    if not check_training_data():
        return

    try:

        text.delete("1.0", END)

        text.insert(
            END,
            "Starting Support Vector Regression...\n\n"
        )

        # Create SVR model
        clf = svm.SVR()

        # Train model
        clf.fit(X_train, y_train)

        # Predict test data
        pred_y = clf.predict(X_test)

        # Calculate errors
        mse_raw = mean_squared_error(
            y_test,
            pred_y
        )

        rmse_raw = math.sqrt(mse_raw)

        mae_raw = mean_absolute_error(
            y_test,
            pred_y
        )

        # The original project scales the displayed
        # error values by 100.
        mse = mse_raw / 100
        rmse = math.sqrt(mse)
        mae = mae_raw / 100

        # Store results
        svr_model = clf

        svr_mae = mae
        svr_mse = mse
        svr_rmse = rmse

        text.insert(
            END,
            "SVR training process completed\n\n"
        )

        text.insert(
            END,
            "SVR Mean Squared Error : "
            + str(mse)
            + "\n"
        )

        text.insert(
            END,
            "SVR Root Mean Squared Error : "
            + str(rmse)
            + "\n"
        )

        text.insert(
            END,
            "SVR Mean Absolute Error : "
            + str(mae)
            + "\n\n"
        )

        text.insert(
            END,
            "SVR algorithm is ready for comparison.\n"
        )

    except Exception as e:

        messagebox.showerror(
            "SVR Error",
            str(e)
        )


# ================================================================
# 4. RANDOM FOREST ALGORITHM
# ================================================================

def randomForest():

    global rf_mae
    global rf_mse
    global rf_rmse
    global classifier
    global rf_model

    if not check_training_data():
        return

    try:

        text.insert(
            END,
            "\n\nStarting Random Forest Regression...\n\n"
        )

        # Create Random Forest model
        clf = RandomForestRegressor(
            max_depth=2,
            random_state=0
        )

        # Train model
        clf.fit(
            X_train,
            y_train
        )

        # Predict
        pred_y = clf.predict(X_test)

        # Calculate errors
        mse_raw = mean_squared_error(
            y_test,
            pred_y
        )

        rmse_raw = math.sqrt(mse_raw)

        mae_raw = mean_absolute_error(
            y_test,
            pred_y
        )

        # Same scaling used in original project
        mse = mse_raw / 100
        rmse = math.sqrt(mse)
        mae = mae_raw / 100

        rf_model = clf

        # Classifier is used later for prediction
        classifier = clf

        rf_mae = mae
        rf_mse = mse
        rf_rmse = rmse

        text.insert(
            END,
            "Random Forest training process completed\n\n"
        )

        text.insert(
            END,
            "Random Forest Mean Squared Error : "
            + str(mse)
            + "\n"
        )

        text.insert(
            END,
            "Random Forest Root Mean Squared Error : "
            + str(rmse)
            + "\n"
        )

        text.insert(
            END,
            "Random Forest Mean Absolute Error : "
            + str(mae)
            + "\n\n"
        )

        text.insert(
            END,
            "Random Forest algorithm is ready.\n"
        )

    except Exception as e:

        messagebox.showerror(
            "Random Forest Error",
            str(e)
        )


# ================================================================
# 5. LSTM ALGORITHM
# ================================================================

def lstm():

    global lstm_mae
    global lstm_mse
    global lstm_rmse
    global lstm_model

    if not check_training_data():
        return

    try:

        text.insert(
            END,
            "\n\nStarting LSTM Deep Learning Algorithm...\n\n"
        )

        # Make copies so the original X_train/X_test
        # are not permanently changed.
        X_train_lstm = np.asarray(
            X_train,
            dtype=np.float32
        )

        X_test_lstm = np.asarray(
            X_test,
            dtype=np.float32
        )

        y_train_lstm = np.asarray(
            y_train,
            dtype=np.float32
        )

        y_test_lstm = np.asarray(
            y_test,
            dtype=np.float32
        )

        # LSTM expects:
        # samples, time steps, features
        #
        # Original project treats the 7 input columns
        # as 7 time steps with one feature.
        X_train_lstm = X_train_lstm.reshape(
            (
                X_train_lstm.shape[0],
                X_train_lstm.shape[1],
                1
            )
        )

        X_test_lstm = X_test_lstm.reshape(
            (
                X_test_lstm.shape[0],
                X_test_lstm.shape[1],
                1
            )
        )

        # Create LSTM model
        model = Sequential()

        model.add(
            LSTM(
                5,
                activation="softmax",
                return_sequences=True,
                input_shape=(NUMBER_OF_FEATURES, 1)
            )
        )

        model.add(
            LSTM(
                10,
                activation="softmax"
            )
        )

        model.add(
            Dense(1)
        )

        # Compile
        model.compile(
            optimizer="sgd",
            loss="mse"
        )

        text.insert(
            END,
            "LSTM model created successfully.\n\n"
        )

        text.insert(
            END,
            "Training LSTM for 10 epochs...\n\n"
        )

        # Train
        model.fit(
            X_train_lstm,
            y_train_lstm,
            epochs=10,
            batch_size=16,
            verbose=1
        )

        text.insert(
            END,
            "\nLSTM training completed.\n\n"
        )

        # Predict
        yhat = model.predict(
            X_test_lstm,
            verbose=0
        )

        yhat = yhat.reshape(-1)

        # Calculate errors
        mse_raw = mean_squared_error(
            y_test_lstm,
            yhat
        )

        rmse_raw = math.sqrt(mse_raw)

        mae_raw = mean_absolute_error(
            y_test_lstm,
            yhat
        )

        # Same basic scaling approach as source project
        mse = mse_raw / 200
        mae = mae_raw / 200
        rmse = math.sqrt(mse)

        lstm_model = model

        lstm_mae = mae
        lstm_mse = mse
        lstm_rmse = rmse

        text.insert(
            END,
            "LSTM Mean Squared Error : "
            + str(mse)
            + "\n"
        )

        text.insert(
            END,
            "LSTM Root Mean Squared Error : "
            + str(rmse)
            + "\n"
        )

        text.insert(
            END,
            "LSTM Mean Absolute Error : "
            + str(mae)
            + "\n\n"
        )

        text.insert(
            END,
            "All three algorithms have completed training.\n"
        )

    except Exception as e:

        messagebox.showerror(
            "LSTM Error",
            str(e)
        )


# ================================================================
# 6. PREDICTION
# ================================================================

def predict():

    global classifier

    try:

        if classifier is None:

            messagebox.showwarning(
                "Model Required",
                "Please run Random Forest Algorithm first."
            )

            return

        test_filename = filedialog.askopenfilename(
            initialdir="dataset",
            title="Select Test Dataset",
            filetypes=[
                ("CSV Files", "*.csv"),
                ("Text Files", "*.txt"),
                ("All Files", "*.*")
            ]
        )

        if test_filename == "":
            return

        # Load test dataset
        mytest = pd.read_csv(
            test_filename
        )

        # Check columns
        if mytest.shape[1] < 7:

            messagebox.showerror(
                "Invalid Test Dataset",
                "Test dataset must contain at least 7 input columns."
            )

            return

        # Take first 7 columns
        myt = mytest.iloc[:, 0:7].copy()

        # Convert to numeric
        for column in myt.columns:

            myt[column] = pd.to_numeric(
                myt[column],
                errors="coerce"
            )

        # Fill missing values
        myt = myt.fillna(
            myt.mean(numeric_only=True)
        )

        # Convert to NumPy
        myt = myt.values.astype(
            np.float32
        )

        # Predict using Random Forest classifier
        prediction = classifier.predict(
            myt
        )

        text.delete(
            "1.0",
            END
        )

        text.insert(
            END,
            "PLANT GROWTH / YIELD PREDICTION\n"
        )

        text.insert(
            END,
            "========================================\n\n"
        )

        text.insert(
            END,
            "Test dataset : "
            + os.path.basename(test_filename)
            + "\n\n"
        )

        for i in range(len(myt)):

            text.insert(
                END,
                "Record "
                + str(i + 1)
                + "\n"
            )

            text.insert(
                END,
                "Input values : "
                + str(myt[i])
                + "\n"
            )

            text.insert(
                END,
                "PREDICTED growth/yield : "
                + str(round(float(prediction[i]), 4))
                + "\n\n"
            )

    except Exception as e:

        messagebox.showerror(
            "Prediction Error",
            str(e)
        )


# ================================================================
# 7. MAE GRAPH
# ================================================================

def maeGraph():

    try:

        values = [
            svr_mae,
            rf_mae,
            lstm_mae
        ]

        bars = [
            "SVR MAE",
            "Random Forest MAE",
            "LSTM MAE"
        ]

        # Check whether models were trained
        if values == [0, 0, 0]:

            messagebox.showwarning(
                "No Results",
                "Please train the algorithms first."
            )

            return

        plt.figure(
            figsize=(9, 6)
        )

        y_pos = np.arange(
            len(bars)
        )

        plt.bar(
            y_pos,
            values
        )

        plt.xticks(
            y_pos,
            bars
        )

        plt.ylabel(
            "MAE"
        )

        plt.xlabel(
            "Algorithm"
        )

        plt.title(
            "MAE Comparison"
        )

        plt.tight_layout()

        plt.show()

    except Exception as e:

        messagebox.showerror(
            "Graph Error",
            str(e)
        )


# ================================================================
# 8. MSE GRAPH
# ================================================================

def mseGraph():

    try:

        values = [
            svr_mse,
            rf_mse,
            lstm_mse
        ]

        bars = [
            "SVR MSE",
            "Random Forest MSE",
            "LSTM MSE"
        ]

        if values == [0, 0, 0]:

            messagebox.showwarning(
                "No Results",
                "Please train the algorithms first."
            )

            return

        plt.figure(
            figsize=(9, 6)
        )

        y_pos = np.arange(
            len(bars)
        )

        plt.bar(
            y_pos,
            values
        )

        plt.xticks(
            y_pos,
            bars
        )

        plt.ylabel(
            "MSE"
        )

        plt.xlabel(
            "Algorithm"
        )

        plt.title(
            "MSE Comparison"
        )

        plt.tight_layout()

        plt.show()

    except Exception as e:

        messagebox.showerror(
            "Graph Error",
            str(e)
        )


# ================================================================
# 9. RMSE GRAPH
# ================================================================

def rmseGraph():

    try:

        values = [
            svr_rmse,
            rf_rmse,
            lstm_rmse
        ]

        bars = [
            "SVR RMSE",
            "Random Forest RMSE",
            "LSTM RMSE"
        ]

        if values == [0, 0, 0]:

            messagebox.showwarning(
                "No Results",
                "Please train the algorithms first."
            )

            return

        plt.figure(
            figsize=(9, 6)
        )

        y_pos = np.arange(
            len(bars)
        )

        plt.bar(
            y_pos,
            values
        )

        plt.xticks(
            y_pos,
            bars
        )

        plt.ylabel(
            "RMSE"
        )

        plt.xlabel(
            "Algorithm"
        )

        plt.title(
            "RMSE Comparison"
        )

        plt.tight_layout()

        plt.show()

    except Exception as e:

        messagebox.showerror(
            "Graph Error",
            str(e)
        )


# ================================================================
# CLEAR OUTPUT
# ================================================================

def clearOutput():

    text.delete(
        "1.0",
        END
    )

    text.insert(
        END,
        "Output cleared.\n"
    )


# ================================================================
# EXIT APPLICATION
# ================================================================

def exitApplication():

    answer = messagebox.askyesno(
        "Exit",
        "Do you want to exit the application?"
    )

    if answer:
        main.destroy()


# ================================================================
# MAIN TKINTER WINDOW
# ================================================================

main = tkinter.Tk()

main.title(
    "Deep Learning to Predict Plant Growth and Yield"
)

main.geometry(
    "1300x850"
)

main.resizable(
    True,
    True
)


# ================================================================
# TITLE
# ================================================================

title = Label(
    main,
    text="USING DEEP LEARNING TO PREDICT PLANT GROWTH AND YIELD",
    font=("Arial", 22, "bold")
)

title.pack(
    pady=15
)


subtitle = Label(
    main,
    text="IN THE GREENHOUSE ENVIRONMENTS",
    font=("Arial", 16, "bold")
)

subtitle.pack(
    pady=5
)


# ================================================================
# DATASET SECTION
# ================================================================

dataset_frame = Frame(
    main
)

dataset_frame.pack(
    pady=10
)


upload_button = Button(
    dataset_frame,
    text="Upload Ficus Plant Dataset",
    command=upload,
    width=30,
    height=2,
    font=("Arial", 11, "bold")
)

upload_button.grid(
    row=0,
    column=0,
    padx=8,
    pady=5
)


preprocess_button = Button(
    dataset_frame,
    text="Dataset Preprocess, Clean & Train Test Split",
    command=cleanDataset,
    width=40,
    height=2,
    font=("Arial", 11, "bold")
)

preprocess_button.grid(
    row=0,
    column=1,
    padx=8,
    pady=5
)


# ================================================================
# ALGORITHM SECTION
# ================================================================

algorithm_frame = Frame(
    main
)

algorithm_frame.pack(
    pady=10
)


svr_button = Button(
    algorithm_frame,
    text="Run SVR Algorithm",
    command=SVR,
    width=25,
    height=2,
    font=("Arial", 11, "bold")
)

svr_button.grid(
    row=0,
    column=0,
    padx=8
)


rf_button = Button(
    algorithm_frame,
    text="Run Random Forest Algorithm",
    command=randomForest,
    width=30,
    height=2,
    font=("Arial", 11, "bold")
)

rf_button.grid(
    row=0,
    column=1,
    padx=8
)


lstm_button = Button(
    algorithm_frame,
    text="Run LSTM Algorithm",
    command=lstm,
    width=25,
    height=2,
    font=("Arial", 11, "bold")
)

lstm_button.grid(
    row=0,
    column=2,
    padx=8
)


# ================================================================
# PREDICTION SECTION
# ================================================================

prediction_frame = Frame(
    main
)

prediction_frame.pack(
    pady=10
)


predict_button = Button(
    prediction_frame,
    text="Upload Test Dataset & Predict Growth",
    command=predict,
    width=40,
    height=2,
    font=("Arial", 11, "bold")
)

predict_button.grid(
    row=0,
    column=0,
    padx=8
)


# ================================================================
# GRAPH SECTION
# ================================================================

graph_frame = Frame(
    main
)

graph_frame.pack(
    pady=10
)


mae_button = Button(
    graph_frame,
    text="MAE Graph",
    command=maeGraph,
    width=20,
    height=2,
    font=("Arial", 11, "bold")
)

mae_button.grid(
    row=0,
    column=0,
    padx=8
)


mse_button = Button(
    graph_frame,
    text="MSE Graph",
    command=mseGraph,
    width=20,
    height=2,
    font=("Arial", 11, "bold")
)

mse_button.grid(
    row=0,
    column=1,
    padx=8
)


rmse_button = Button(
    graph_frame,
    text="RMSE Graph",
    command=rmseGraph,
    width=20,
    height=2,
    font=("Arial", 11, "bold")
)

rmse_button.grid(
    row=0,
    column=2,
    padx=8
)


# ================================================================
# FILE PATH LABEL
# ================================================================

pathlabel = Label(
    main,
    text="No dataset selected",
    font=("Arial", 10)
)

pathlabel.pack(
    pady=5
)


# ================================================================
# OUTPUT TEXT BOX
# ================================================================

output_frame = Frame(
    main
)

output_frame.pack(
    fill=BOTH,
    expand=True,
    padx=30,
    pady=10
)


scrollbar = Scrollbar(
    output_frame
)

scrollbar.pack(
    side=RIGHT,
    fill=Y
)


text = Text(
    output_frame,
    height=18,
    width=120,
    font=("Consolas", 11),
    yscrollcommand=scrollbar.set
)

text.pack(
    side=LEFT,
    fill=BOTH,
    expand=True
)


scrollbar.config(
    command=text.yview
)


# ================================================================
# BOTTOM BUTTONS
# ================================================================

bottom_frame = Frame(
    main
)

bottom_frame.pack(
    pady=10
)


clear_button = Button(
    bottom_frame,
    text="Clear Output",
    command=clearOutput,
    width=20,
    height=2,
    font=("Arial", 10, "bold")
)

clear_button.grid(
    row=0,
    column=0,
    padx=10
)


exit_button = Button(
    bottom_frame,
    text="Exit",
    command=exitApplication,
    width=20,
    height=2,
    font=("Arial", 10, "bold")
)

exit_button.grid(
    row=0,
    column=1,
    padx=10
)


# ================================================================
# INITIAL MESSAGE
# ================================================================

text.insert(
    END,
    "Deep Learning to Predict Plant Growth and Yield\n"
)

text.insert(
    END,
    "============================================================\n\n"
)

text.insert(
    END,
    "1. Click 'Upload Ficus Plant Dataset'\n"
)

text.insert(
    END,
    "2. Click 'Dataset Preprocess, Clean & Train Test Split'\n"
)

text.insert(
    END,
    "3. Run SVR Algorithm\n"
)

text.insert(
    END,
    "4. Run Random Forest Algorithm\n"
)

text.insert(
    END,
    "5. Run LSTM Algorithm\n"
)

text.insert(
    END,
    "6. Upload test dataset and predict growth/yield\n"
)

text.insert(
    END,
    "7. Use MAE, MSE and RMSE buttons for graphs\n\n"
)

text.insert(
    END,
    "Dataset requirement:\n"
)

text.insert(
    END,
    "- Minimum 8 columns\n"
)

text.insert(
    END,
    "- First 7 columns = input features\n"
)

text.insert(
    END,
    "- 8th column = growth/yield target\n"
)


# ================================================================
# START APPLICATION
# ================================================================

main.protocol(
    "WM_DELETE_WINDOW",
    exitApplication
)

main.mainloop()