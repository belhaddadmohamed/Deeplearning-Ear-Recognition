import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Combobox, Label
import numpy as np
import cv2
from PIL import Image, ImageTk
import joblib
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the CNN model
cnn_model = load_model('CNN_Training_img\my_cnn_model_img150x100.h5')

# Load the pre-trained classification models
knn_albp_model = joblib.load('Saved models\knn_ALBP.pkl')
knn_llbp_model = joblib.load('Saved models\knn_ALBP.pkl')
svm_albp_model = joblib.load('Saved models\svm_ALBP.pkl')
svm_llbp_model = joblib.load('Saved models\svm_ALBP.pkl')

# =========================================================================================================
# Global dictionary to keep references to the images
image_refs = {}



# Functions for feature extraction
def compute_llbp(image):
    llbp_image = np.zeros_like(image, dtype=np.float32)
    weights = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            value = 0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    value += image[i + m, j + n] * weights[m + 1, n + 1]
            llbp_image[i, j] = np.clip(value, 0, 255)
    return llbp_image

def compute_albp(image):
    albp_image = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            center = image[i, j]
            value = 0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    if m != 0 or n != 0:
                        value += int(image[i + m, j + n] >= center) * 2**((3 * (m + 1)) + (n + 1))
            albp_image[i, j] = np.clip(value, 0, 255)
    return albp_image
# =========================================================================================================



def redimensionner_image(image, width, height):
    return cv2.resize(image, (width, height))


# Function to preprocess the image
def preprocess_image(image):
    filtered_image = cv2.GaussianBlur(image, (5, 5), 0)
    gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    width, height = 100, 150
    resized_image = redimensionner_image(gray_image, width, height)
    return resized_image


from tensorflow.keras.preprocessing import image
# Load and preprocess the image
def preprocess_cnn_image():
    img = image.load_img(file_path, color_mode='grayscale', target_size=(150, 100))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale if required by the model
    return img_array


# Function to predict with a preprocessed image
def predict_cnn(model, preprocessed_img_array):
    class_labels = ['1', '10', '100', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']
    predicted_probabilities = model.predict(preprocessed_img_array)
    predicted_class_index = np.argmax(predicted_probabilities, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label



# =========================================================================================================

# Function to handle image loading
def load_image():
    global file_path
    image_refs.clear()
    file_path = filedialog.askopenfilename()
    if file_path:
        original_image = cv2.imread(file_path)
        resized_image = redimensionner_image(original_image, 100, 150)
        # display_image(resized_image, canvas.winfo_width() // 4, canvas.winfo_height() // 2, "Original Image")
        display_image(resized_image, canvas.winfo_width() // 6, canvas.winfo_height() // 4, "Original Image")



# Function to display image in the GUI
def display_image(image, initial_x , initial_y, title):
    global image_refs
    global image_label
    # image_label.config(text=title)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # method = method_combobox.get()  # Added New...
    # if method == 'LLBP':    # Added New...
    #     image = (image * 255).astype(np.uint8)   # Added New...
    image = Image.fromarray((image * 1).astype(np.uint8)).convert('RGB')
    # image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)

    image_id = canvas.create_image(initial_x, initial_y, anchor=tk.CENTER, image=image)
    canvas.create_text(initial_x, initial_y + 100, text=title, anchor=tk.N, fill="white")
    image_refs[image_id] = image  # Keep a reference to avoid garbage collection
    

def preprocessing(display_status=True):
    original_image = cv2.imread(file_path)
    global preprocessed_image
    preprocessed_image = preprocess_image(original_image)
    if display_status:
        display_image(preprocessed_image, canvas.winfo_width() // 6, canvas.winfo_height() * 0.7, "Preprocessed Image")
        

def extract_features(display_status=True):
    # Preprocessing
    preprocessing(display_status=False)

    # Extract features based on the selected method (ALBP or LLBP)
    global features
    global method
    method = method_combobox.get()
    if method == 'ALBP':
        features = compute_albp(preprocessed_image)
        if display_status:
            display_image(features, canvas.winfo_width() // 2, canvas.winfo_height() // 4, "ALBP Features")
            display_histogram(features)
    elif method == 'LLBP':
        features = compute_llbp(preprocessed_image)
        if display_status:
            display_image(features, canvas.winfo_width() // 2, canvas.winfo_height() // 4, "LLBP Features")
            display_histogram(features)
    else:
        messagebox.showerror('Error', 'Invalid feature extraction method selected')
        return


# Function to display the histogram
def display_histogram(features):
    global canvas, image_refs
    fig, ax = plt.subplots(figsize=(4, 2))  # Set the desired size here
    ax.hist(features.ravel(), bins=256, color='blue', alpha=0.7)
    ax.set_title('Features Histogram')
    
    # Convert the plot to an image
    fig.canvas.draw()
    plot_image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

    # Display the plot on the canvas
    plot_tk = ImageTk.PhotoImage(plot_image)
    plot_id = canvas.create_image(canvas.winfo_width() * 0.68, canvas.winfo_height() * 0.75, image=plot_tk)
    # canvas.create_text(canvas.winfo_width() * 0.8, canvas.winfo_height() * 0.9 - 100, text="Feature Histogram", anchor=tk.N, fill="white")
    image_refs[plot_id] = plot_tk




# Function to perform feature extraction and classification
def classify_image():

    preprocessing(display_status=False)
    extract_features(display_status=False)

    # Flatten the features
    features_flat = features.flatten()

    # Perform classification based on the selected method (KNN, SVM, or CNN)
    classification_method = classification_combobox.get()
    if classification_method == 'KNN':
        if method == 'ALBP':
            result = knn_albp_model.predict([features_flat])[0]
        elif method == 'LLBP':
            result = knn_llbp_model.predict([features_flat])[0]
    elif classification_method == 'SVM':
        if method == 'ALBP':
            result = svm_albp_model.predict([features_flat])[0]
        elif method == 'LLBP':
            result = svm_llbp_model.predict([features_flat])[0]
    elif classification_method == 'CNN':
        # Preprocess features for CNN model
        features_cnn = preprocess_cnn_image()
        result = predict_cnn(cnn_model, features_cnn)
    else:
        messagebox.showerror('Error', 'Invalid classification method selected')
        return

    # Display the classification result
    messagebox.showinfo('Classification Result', f'This ear image is blong to person number : {result}')



# ========================================================================================

def resize_bg(event):
    # Get the new size of the window
    new_width = event.width
    new_height = event.height
    # Resize the background image to the new size
    resized_bg_image = bg_image.resize((new_width, new_height), Image.LANCZOS)
    # Update the image on the canvas
    bg_photo = ImageTk.PhotoImage(resized_bg_image)
    canvas.itemconfig(bg_image_item, image=bg_photo)
    canvas.bg_photo = bg_photo  # Keep a reference to avoid garbage collection


# =========================================================================================================

def create_main_window():
    global canvas, bg_image, bg_image_item, root, method_combobox, classification_combobox

    # GUI setup
    root = tk.Tk()
    root.title('Image Classification')

    # Create canvas to display images
    canvas = tk.Canvas(root, width=800, height=500)
    canvas.grid(row=3, column=0, columnspan=3)

    # Load the background image
    bg_image = Image.open("Interface/Netzwerk_04.jpg")

    # Configure the root grid layout
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    # Bind the resize event to update the background image
    root.bind('<Configure>', resize_bg)

    # Add the background image to the canvas
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_image_item = canvas.create_image(0, 0, image=bg_photo, anchor="nw")

    # Create methods comboboxes
    method_label = tk.Label(root, text="Select Method:")
    method_label.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

    method_combobox = Combobox(root, values=['ALBP', 'LLBP'])
    method_combobox.current(0)  # Set default value
    method_combobox.grid(row=0, column=1, padx=100, pady=5, sticky=tk.W)

    # Create classification comboboxes
    classification_label = tk.Label(root, text="Select Classifier:")
    classification_label.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)

    classification_combobox = Combobox(root, values=['KNN', 'SVM', 'CNN'])
    classification_combobox.current(0)  # Set default value
    classification_combobox.grid(row=0, column=2, padx=100, pady=5, sticky=tk.W)

    # Create buttons
    load_button = tk.Button(root, text='Load Image', command=load_image)
    load_button.grid(row=0, column=0, pady=10, padx=5, sticky=tk.W)

    load_button = tk.Button(root, text='Preprocess Image', command=preprocessing)
    load_button.grid(row=1, column=0, pady=10, padx=5, sticky=tk.W)

    process_button = tk.Button(root, text='Extract Features', command=extract_features)
    process_button.grid(row=1, column=1, pady=10, padx=5, sticky=tk.W)

    process_button = tk.Button(root, text='Classify Image', command=classify_image)
    process_button.grid(row=1, column=2, pady=10, padx=5, sticky=tk.W)

    # Create label for image
    image_label = tk.Label(root, text="")
    image_label.grid(row=4, column=0, columnspan=2, padx=5, pady=5)


    root.mainloop()



# ============================================================================================
# SPLASH
# Create the splash screen
# splash = tk.Tk()
# splash.title("Loading...")

# # Set the size of the splash screen and make it non-resizable
# splash.geometry("400x200")
# splash.resizable(False, False)
# splash.overrideredirect(True)  # Remove window borders and title bar

# # Center the splash screen on the screen
# window_width = 595
# window_height = 301
# screen_width = splash.winfo_screenwidth()
# screen_height = splash.winfo_screenheight()
# position_top = int(screen_height / 2 - window_height / 2)
# position_right = int(screen_width / 2 - window_width / 2)
# splash.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

# # Create a canvas to hold the background image
# splash_canvas = tk.Canvas(splash, width=400, height=200)
# splash_canvas.pack(fill="both", expand=True)

# # Load the background image for the splash screen
# splash_bg_image_path = "Interface/Ear-splash (595x301).jpg"  # Replace with your splash screen image path
# splash_bg_image = Image.open(splash_bg_image_path)
# splash_bg_photo = ImageTk.PhotoImage(splash_bg_image)

# # Add the background image to the canvas
# splash_canvas.create_image(0, 0, image=splash_bg_photo, anchor="nw")

# # Add a label to the splash screen
# splash_label = tk.Label(splash_canvas, text="""Deep Learning-Based Ear Recognition: Enhancing
# Accuracy and Performance in Real-World Scenarios""", font=("Helvetica", 13), bg="#d3d3d3", anchor=tk.CENTER)
# splash_label_window = splash_canvas.create_window(220, 80, window=splash_label)

# # Close the splash screen after a delay and open the main application
# splash.after(3000, lambda: [splash.destroy(), create_main_window()])

# # Run the splash screen main loop
# splash.mainloop()


# ================================================================================================
def launch_main_window():
    intro_root.destroy()  # Close the introductory window
    create_main_window()  # Launch the main window

# Introductory window setup
intro_root = tk.Tk()
intro_root.title('Welcome')
intro_root.resizable(False, False)
intro_root.overrideredirect(True)  # Remove window borders and title bar

# Set the size of the introductory window
window_width = 595
window_height = 301

# Get the screen dimension
screen_width = intro_root.winfo_screenwidth()
screen_height = intro_root.winfo_screenheight()

# Find the center point
center_x = int(screen_width / 2 - window_width / 2)
center_y = int(screen_height / 2 - window_height / 2)

# Set the position of the window to the center of the screen
intro_root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

# Load and set the background image
intro_bg_image = Image.open("Interface\Ear-splash (595x301).jpg")
intro_bg_photo = ImageTk.PhotoImage(intro_bg_image)

intro_canvas = tk.Canvas(intro_root, width=window_width, height=window_height)
intro_canvas.pack(fill="both", expand=True)
intro_canvas.create_image(0, 0, image=intro_bg_photo, anchor="nw")

# Create title label
title_label = tk.Label(intro_canvas, text="""Deep Learning-Based Ear Recognition: Enhancing
 Accuracy and Performance in Real-World Scenarios""", font=("Helvetica", 13), bg="#d3d3d3", anchor=tk.CENTER)
title_label.place(relx=0.4, rely=0.2, anchor="center")

# Create "Launch" button
launch_button = tk.Button(intro_root, text="Launch", command=launch_main_window)
launch_button.place(relx=0.2, rely=0.8, anchor="center")

# Create "Exit" button
exit_button = tk.Button(intro_root, text="Exit", command=intro_root.quit)
exit_button.place(relx=0.4, rely=0.8, anchor="center")

intro_root.mainloop()