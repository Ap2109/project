from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from keras.preprocessing import image
import numpy as np
import os
from keras.models import load_model
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import csv
from classify import classify_image

app = Flask(__name__)

# Adjust the database URI based on your MySQL configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost:3306/user'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Registration(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    First_Name = db.Column(db.String(20), nullable=False)
    Last_Name = db.Column(db.String(20), nullable=False)
    Email_Address = db.Column(db.String(40), unique=True, nullable=False)
    Username = db.Column(db.String(10), nullable=False)
    Password = db.Column(db.String(128), nullable=False)  # Increased length for hashed passwords


with app.app_context():
    db.create_all()

'''
# Path to the original dataset
original_dataset_path = "C:\\Users\\manis\\Desktop\\plantvillage dataset"

# Path to the output directory where the split dataset will be stored
output_dataset_path = "Generated_dataset"

# Split the dataset into training, validation, and test sets
splitfolders.ratio(original_dataset_path, output=output_dataset_path, seed=1337, ratio=(0.6, 0.2, 0.2))
'''

# Load the model
model = load_model('trmodel.h5')

# Create the uploads folder if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the directory to store preprocessed images
PREPROCESSED_FOLDER = 'preprocessed_images'
if not os.path.exists(PREPROCESSED_FOLDER):
    os.makedirs(PREPROCESSED_FOLDER)

# Initialize the ImageDataGenerator for data augmentation and normalization
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1. / 255  # normalize pixel values to [0, 1]
)


def get_recommendation(predicted_class):
    recommendations_file = 'C:\\Users\\manis\\PycharmProjects\\plant1\\recommendation.csv'
    with open(recommendations_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(row)  # Print each row from the CSV file
            if row['Disease'] == predicted_class:
                return row['Recommendation']
    return None


# Route to serve uploaded images
@app.route('/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# Route to render home page
@app.route('/')
def home():
    return render_template('Home.html')


# Route for the registration page
@app.route("/Register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']

        # Hash the password before storing it in the database
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        new_registration = Registration(
            First_Name=first_name,
            Last_Name=last_name,
            Email_Address=email,
            Username=username,
            Password=hashed_password
        )

        db.session.add(new_registration)
        db.session.commit()

        return redirect(url_for('login'))

    return render_template('Register.html')


# Route for the login page
@app.route("/Login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        entered_password = request.form['password']

        # Retrieve the user from the database based on the entered username
        user = Registration.query.filter_by(Username=username).first()

        if user and check_password_hash(user.Password, entered_password):
            # Passwords match, proceed with login
            return redirect(url_for('upload_file'))
        else:
            # Incorrect username or password, handle accordingly
            error_message = "Invalid username or password. Please try again."
            return render_template('Login.html', error_message=error_message)

    return render_template('Login.html')


# Route to render file upload form
@app.route('/Upload', methods=['GET', 'POST'])
def upload_file():
    return render_template('Upload.html')


# Route to handle file upload and classification
@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Preprocess the uploaded image using ImageDataGenerator
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch
        datagen.fit(img_array)
        img_iterator = datagen.flow(img_array, batch_size=1)

        # Save the preprocessed image
        preprocessed_image = img_iterator.next()[0]
        preprocessed_image_path = os.path.join(PREPROCESSED_FOLDER, filename)
        image_pil = Image.fromarray((preprocessed_image * 255).astype(np.uint8))
        image_pil.save(preprocessed_image_path)

        # Classify the preprocessed image using the function from classify.py
        predicted_class = classify_image(preprocessed_image_path)  # Placeholder for actual classification
        # Get recommendation based on the predicted disease
        recommendation = get_recommendation(predicted_class)

        # Render template with classification result and recommendation
        return render_template('result.html', filename=filename, predicted_class=predicted_class,
                               recommendation=recommendation)


if __name__ == '__main__':
    app.run(debug=True)
