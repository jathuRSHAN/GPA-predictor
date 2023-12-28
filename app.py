from flask import Flask
from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import pickle
import os
from matplotlib import pyplot as plt

app = Flask(__name__)

# Importing the machine Learning model
model = pickle.load(open('regressor_CGPA.pkl', 'rb'))

# Home End Point
@app.route('/')
def home():
    return render_template('home.html')

# Define a Flask route that listens for HTTP POST requests with the path '/predict'
@app.route('/predict', methods=['POST'])
def predict():
    # Initialize the prediction variable to None
    prediction = None
    # Check if the incoming HTTP request method is POST
    if request.method == 'POST':
        # Extract the values from the HTTP POST request form data and convert them to the appropriate data type
        gender = int(request.form.get('gender', 0))
        tm = float(request.form.get('Tm', 0))
        motivation = float(request.form.get('Motivation', 0))
        efficacy = float(request.form.get('Efficacy', 0))
        social_behavior = float(request.form.get('SocialBehavior', 0))
        rlb = float(request.form.get('RLB', 0))
        ape = float(request.form.get('APE', 0))
        curiosity = float(request.form.get('Curiosity', 0))
        # Store the extracted values as a list of inputs to the model
        input_data = [[gender, tm, motivation, efficacy, social_behavior, rlb, ape, curiosity]]
        # Use the trained model to make a prediction based on the input data
        prediction = model.predict(input_data)
    # Render the 'result.html' template with the prediction variable as a parameter
    return render_template('result.html', prediction=prediction)




@app.route('/upload', methods=['GET', 'POST'])
def upload():
    try:
        if request.method == 'POST':
            # Get the uploaded file
            file = request.files.get('file')
            # Error Handlings
            if file is None:
                return render_template('upload.html', message='No file selected')
            # Remove the file if the file Name Already exists
            if os.path.exists(file.filename):
                os.remove(file.filename)
            # Save the file locally
            file.save(file.filename)
            # Preprocess the data
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file.filename)
            elif file.filename.endswith('.xlsx'):
                df = pd.read_excel(file.filename)
            else:
                return render_template('upload.html', message='File format not supported')
            # Data cleaning
            df = df.dropna() # remove rows with NaN values
            pre_processing(df, file.filename)
            fileName =None
            fileName = "Predicted_"+file.filename
            return render_template('upload.html', filename = fileName)
        else:
            return render_template('upload.html')
    except:
        return render_template('upload.html', message='Error occurred while uploading the file')

@app.route('/view-file/<filename>')
def view_file(filename):
    # Load the file as a DataFrame
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
    elif filename.endswith('.xlsx'):
        df = pd.read_excel(filename)
    else:
        return render_template('error.html', message='File format not supported')
    # Render the table view
    return render_template('fileview.html', table=df.to_html())


@app.route('/display_dashboard', methods=['GET'])
def display_dashboard():
    return render_template("dashboard.html")
    


def pre_processing(df, fileName):
    try:
        # Read the preprocessed dataframe
        df1 = df.copy()
        df = df.drop(["Efficacy/ Condfidance (groups)", "Efficacy/ Condfidance (groups) 2",
                      "Efficacy/ Condfidance (groups) 2 (groups)", 'STUDENT ID NUMBER', 'IDNO'], axis='columns')
        df_dummy = pd.get_dummies(df, prefix=['GENDER'], columns=['GENDER'], drop_first=False)
        # Extract the features from the new data
                # Extract the features from the new data
        features = ['Time Management', 'Motivation', 'Efficacy/ Condfidance', 'Social Behavior',
                          'Resources Learning Behavior', 'Academic Performance Evaluation ',
                          'Curiosity', 'Curiosity (bins)', 'Motivation (bins)',
                          'Efficacy/ Condfidance (bins)', 'GENDER_F', 'GENDER_M']
        for feature in features:
            if feature in df_dummy.columns:
                X_new = df_dummy[features]
        y_pred = model.predict(X_new)
        df['CGPA_prediction'] = y_pred
        df =  pd.concat([df1['STUDENT ID NUMBER'], df], axis=1) 
        fileName = "Predicted_"+fileName
        if fileName.endswith('.csv'):
            # Check if file already exists and remove it
            if os.path.exists(fileName):
                os.remove(fileName)
            df.to_csv(fileName, index=False)
            return  render_template('upload.html', fileName=fileName)
        elif fileName.endswith('.xlsx'):
            # Check if file already exists and remove it
            if os.path.exists(fileName):
                os.remove(fileName)
            df.to_excel(fileName, index=False)
            return render_template('upload.html', fileName=fileName)
        else:
            return render_template('upload.html', message='File format not supported')
    except:
        return render_template('upload.html', message='Wrong file!')


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)


if __name__ == '__main__':
    app.run()
