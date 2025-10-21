import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import sklearn.preprocessing
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import os
import random

# --- Set Random Seeds for Reproducibility ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# For older TensorFlow versions or specific operations, you might also need:
# tf.compat.v1.set_random_seed(SEED)


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
EPOCHS = 15
BATCH_SIZE = 64
file_path = '/kaggle/input/dataforwebsite/FCR Comments Includes multiple issue comments - Copy 3(Sheet3) (1).csv'
# --- Data Loading and Preprocessing ---
#df = pd.read_csv('/kaggle/input/helloooooo/FCR Comments Includes multiple issue comments - Copy(Sheet3) (10).csv')
try:
    # Attempt to read with 'latin1' encoding
    df = pd.read_csv(file_path, encoding='latin1')
    print("Successfully read the CSV with 'latin1' encoding.")

except UnicodeDecodeError:
    print("Failed to read with 'latin1' encoding. Trying 'cp1252'...")
    try:
        # Attempt to read with 'cp1252' encoding
        df = pd.read_csv(file_path, encoding='cp1252')
        print("Successfully read the CSV with 'cp1252' encoding.")
    except UnicodeDecodeError:
        print("Failed to read with 'cp1252' encoding. The file might be corrupted or encoded differently.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure the file name and path are correct.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure the file name and path are correct.")








df.dropna(subset=['text', 'subcategory', 'label'], inplace=True)
df['label'] = df['label'].astype(str).apply(lambda x: x.title())
df['subcategory'] = df['subcategory'].astype(str).apply(lambda x: x.title())
for col_name in ['label', 'subcategory']:
    df[col_name] = df[col_name].astype(str).str.strip().str.replace(r'[\n\r\s]+', ' ', regex=True).str.strip()


# Initialize separate LabelBinarizers for main categories and subcategories
main_label_binarizer = sklearn.preprocessing.LabelBinarizer()
sub_label_binarizer = sklearn.preprocessing.LabelBinarizer()

# Apply binarization for 'label' (main categories)
main_label_binarizer.fit(df['label'])
df['main_category_int_label'] = main_label_binarizer.transform(df['label']).argmax(axis=1)
df.drop('label', axis='columns', inplace=True)

# Apply binarization for 'subcategory'
sub_label_binarizer.fit(df['subcategory'])
df['subcategory_int_label'] = sub_label_binarizer.transform(df['subcategory']).argmax(axis=1)
df.drop('subcategory', axis='columns', inplace=True)

print("DataFrame head after preprocessing (showing integer labels):")
print(df.head())
print("\nMain Categories (main_label_binarizer.classes_):")
print(main_label_binarizer.classes_)
print("\nSubcategories (sub_label_binarizer.classes_):")
print(sub_label_binarizer.classes_)

# Split the dataset into training and validation sets
val_df = df.sample(frac=0.2, random_state=SEED)
train_df = df.drop(val_df.index)

print(f'\nTraining Dataset Size: {len(train_df)}', f'Validation Dataset Size: {len(val_df)}', sep='\n')

# --- Calculate Class Weights ---
print("\nCalculating Class Weights...")

train_main_labels_int = train_df['main_category_int_label'].values
train_sub_labels_int = train_df['subcategory_int_label'].values

NUM_MAIN_CLASSES = len(main_label_binarizer.classes_)
NUM_SUB_CLASSES = len(sub_label_binarizer.classes_)

main_class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(NUM_MAIN_CLASSES),
    y=train_main_labels_int
)
main_class_weight_dict = dict(zip(np.arange(NUM_MAIN_CLASSES), main_class_weights_array))

print("Main Category Class Weights (by index):", main_class_weight_dict)

sub_class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(NUM_SUB_CLASSES),
    y=train_sub_labels_int
)
sub_class_weight_dict = dict(zip(np.arange(NUM_SUB_CLASSES), sub_class_weights_array))

print("Subcategory Class Weights (by index):", sub_class_weight_dict)

# --- Sanity Check on Integer Labels ---
print("\n--- Sanity Check on Integer Labels ---")
print("Unique main_category_int_label in train_df:", np.unique(train_df['main_category_int_label']))
print("Min main_category_int_label:", np.min(train_df['main_category_int_label']))
print("Max main_category_int_label:", np.max(train_df['main_category_int_label']))
print("Number of main classes defined by binarizer:", NUM_MAIN_CLASSES)

print("\nUnique subcategory_int_label in train_df:", np.unique(train_df['subcategory_int_label']))
print("Min subcategory_int_label:", np.min(train_df['subcategory_int_label']))
print("Max subcategory_int_label:", np.max(train_df['subcategory_int_label']))
print("Number of sub classes defined by binarizer:", NUM_SUB_CLASSES)
print("--------------------------------------")


# --- Add Sample Weights to DataFrame ---
# Map integer labels to their computed weights
train_df['main_sample_weight'] = train_df['main_category_int_label'].map(main_class_weight_dict)
train_df['sub_sample_weight'] = train_df['subcategory_int_label'].map(sub_class_weight_dict)

# --- Helper Function to Convert DataFrame to TensorFlow Dataset (MODIFIED for Sample Weights) ---
def dataframe_to_tf_dataset(dataframe, include_sample_weights=False):
    dataframe = dataframe.copy()
    feature = dataframe.pop('text')

    main_labels = tf.cast(dataframe.pop('main_category_int_label'), tf.int32)
    sub_labels = tf.cast(dataframe.pop('subcategory_int_label'), tf.int32)

    if include_sample_weights:
        main_weights = tf.cast(dataframe.pop('main_sample_weight'), tf.float32)
        sub_weights = tf.cast(dataframe.pop('sub_sample_weight'), tf.float32)
        # Yield (features, {labels_dict}, {sample_weights_dict})
        ds = tf.data.Dataset.from_tensor_slices(
            (feature,
             {'main_category_output': main_labels, 'subcategory_output': sub_labels},
             {'main_category_output': main_weights, 'subcategory_output': sub_weights})
        )
    else:
        # Yield (features, {labels_dict}) for validation set
        ds = tf.data.Dataset.from_tensor_slices(
            (feature, {'main_category_output': main_labels, 'subcategory_output': sub_labels})
        )

    # Use .map to apply tf.squeeze, ensuring labels are scalar before batching
    def _map_fn(text, labels_dict, weights_dict=None):
        processed_labels = {
            'main_category_output': tf.squeeze(labels_dict['main_category_output']),
            'subcategory_output': tf.squeeze(labels_dict['subcategory_output'])
        }
        if weights_dict is not None:
            processed_weights = {
                'main_category_output': tf.squeeze(weights_dict['main_category_output']),
                'subcategory_output': tf.squeeze(weights_dict['subcategory_output'])
            }
            return text, processed_labels, processed_weights
        return text, processed_labels

    # Apply map based on whether sample weights are included
    if include_sample_weights:
        ds = ds.map(lambda text, labels_dict, weights_dict: _map_fn(text, labels_dict, weights_dict), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda text, labels_dict: _map_fn(text, labels_dict), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

# Create TensorFlow Datasets
# Pass include_sample_weights=True for training dataset
train_ds = dataframe_to_tf_dataset(train_df, include_sample_weights=True).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
# Validation dataset does not need sample weights
val_ds = dataframe_to_tf_dataset(val_df, include_sample_weights=False).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)


# --- Model Architecture ---
text_input = tf.keras.Input(shape=(), name='text', dtype='string')
preprocessor = hub.KerasLayer(
    'https://kaggle.com/models/tensorflow/bert/frameworks/TensorFlow2/variations/en-uncased-preprocess/versions/3',
    name='bert_preprocessor'
)
encoder_inputs = preprocessor(text_input)
encoder = hub.KerasLayer(
    'https://www.kaggle.com/models/tensorflow/bert/frameworks/TensorFlow2/variations/bert-en-uncased-l-6-h-128-a-2/versions/2',
    trainable=True,
    name='bert_encoder'
)
outputs = encoder(encoder_inputs)
pooled_output = outputs['pooled_output']
x = tf.keras.layers.Dropout(0.20, name='dropout')(pooled_output)

main_outputs = tf.keras.layers.Dense(len(main_label_binarizer.classes_), activation='softmax', name='main_category_output')(x)
sub_outputs = tf.keras.layers.Dense(len(sub_label_binarizer.classes_), activation='softmax', name='subcategory_output')(x)

model = tf.keras.models.load_model(
    "two_layer_categorization_model.keras",
    custom_objects={"KerasLayer": hub.KerasLayer},
    compile=False
)

# Re-export using the universal SavedModel format
model.save("two_layer_categorization_model_fixed", save_format="tf")
model.compile(
    optimizer='adamax',
    loss={'main_category_output': 'sparse_categorical_crossentropy', 'subcategory_output': 'sparse_categorical_crossentropy'},
    metrics={'main_category_output': ['accuracy'], 'subcategory_output': ['accuracy']}
)
print(model.optimizer)
print("\nModel Summary:")
model.summary()

# --- Custom Callback for Overfitting Detection ---
class OverfittingEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=5, min_delta=0.001, monitor='val_loss'):
        super(OverfittingEarlyStopping, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_loss = np.inf
        self.wait = 0
        self.history = {'loss': [], 'val_loss': []}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get('loss')
        current_val_loss = logs.get(self.monitor)

        if current_loss is None or current_val_loss is None:
            warnings.warn(f"Monitor '{self.monitor}' or 'loss' not found in logs. "
                          "Overfitting detection may not work as expected.", RuntimeWarning)
            return

        self.history['loss'].append(current_loss)
        self.history['val_loss'].append(current_val_loss)

        print(f"Epoch {epoch+1}: Train Loss: {current_loss:.4f}, Validation Loss: {current_val_loss:.4f}")

        if current_val_loss < self.best_loss - self.min_delta:
            self.best_loss = current_val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\nOverfitting detected! Validation loss did not improve for {self.patience} epochs.")
                self.model.stop_training = True


# --- Model Training (Using Sample Weights from Dataset) ---
print("\nStarting Model Training...")

overfitting_callback = OverfittingEarlyStopping(patience=5, min_delta=0.001, monitor='val_loss')

# history = model.fit( # TEMPORARILY COMMENTED OUT FOR THIS ATTEMPT
#     train_ds,
#     epochs=EPOCHS,
#     validation_data=val_ds,
#     callbacks=[overfitting_callback],
#     class_weight={
#         'main_category_output': main_class_weight_dict,
#         'subcategory_output': sub_class_weight_dict
#     }
# )

# Use the train_ds which now yields sample weights
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[overfitting_callback]
    # No class_weight argument needed here, as weights are part of the dataset
)
print("Model Training Complete.")

# --- Save Model and Binarizer Classes ---
model.save('two_layer_categorization_model.keras')

with open('main_category_classes.json', 'w') as file:
    json.dump(main_label_binarizer.classes_.tolist(), file)

with open('subcategory_classes.json', 'w') as file:
    json.dump(sub_label_binarizer.classes_.tolist(), file)

print("\nModel saved as 'two_layer_categorization_model.keras'.")
print("Main category classes saved as 'main_category_classes.json'.")
print("Subcategory classes saved as 'subcategory_classes.json'.")

# --- Prediction Function ---
def predict(text_input_str):
    feature = tf.constant([text_input_str])
    main_predictions_raw, sub_predictions_raw = model(feature)

    main_predictions_dict = {}
    for i, cls in enumerate(main_label_binarizer.classes_):
        main_predictions_dict[cls] = main_predictions_raw[0][i].numpy() * 100
    main_predictions_dict = {k: float(v) for k, v in sorted(main_predictions_dict.items(), key=lambda item: item[1], reverse=True)}

    sub_predictions_dict = {}
    for i, cls in enumerate(sub_label_binarizer.classes_):
        sub_predictions_dict[cls] = sub_predictions_raw[0][i].numpy() * 100
    sub_predictions_dict = {k: float(v) for k, v in sorted(sub_predictions_dict.items(), key=lambda item: item[1], reverse=True)}

    return {"main_category_predictions": main_predictions_dict, "subcategory_predictions": sub_predictions_dict}

# --- Evaluation ---
print("\n--- Model Evaluation ---")

# --- For Main Category Evaluation ---
true_main_labels_indices = val_df['main_category_int_label'].values

predicted_main_labels_indices = []
for text_val in val_df['text'].values:
    preds = predict(text_val)
    predicted_class_name = max(preds["main_category_predictions"], key=preds["main_category_predictions"].get)
    predicted_main_labels_indices.append(main_label_binarizer.classes_.tolist().index(predicted_class_name))

predicted_main_labels_indices = np.array(predicted_main_labels_indices)

accuracy_main = accuracy_score(true_main_labels_indices, predicted_main_labels_indices)
precision_main = precision_score(true_main_labels_indices, predicted_main_labels_indices, average='weighted', zero_division=0)
recall_main = recall_score(true_main_labels_indices, predicted_main_labels_indices, average='weighted', zero_division=0)
f1_main = f1_score(true_main_labels_indices, predicted_main_labels_indices, average='weighted', zero_division=0)

print("\n--- Main Category Evaluation ---")
print(f"Accuracy (Main): {accuracy_main:.4f}")
print(f"Precision (Main, weighted): {precision_main:.4f}")
print(f"Recall (Main, weighted): {recall_main:.4f}")
print(f"F1-score (Main, weighted): {f1_main:.4f}")
print("\nClassification Report (Main):")
print(classification_report(true_main_labels_indices, predicted_main_labels_indices, target_names=main_label_binarizer.classes_, zero_division=0))

cm_main = confusion_matrix(true_main_labels_indices, predicted_main_labels_indices, labels=np.arange(len(main_label_binarizer.classes_)))

plt.figure(figsize=(12, 10))
sns.heatmap(cm_main, annot=True, fmt='d', cmap='Blues',
            xticklabels=main_label_binarizer.classes_,
            yticklabels=main_label_binarizer.classes_)
plt.xlabel('Predicted Main Labels')
plt.ylabel('True Main Labels')
plt.title('Confusion Matrix - Main Categories')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# --- For Subcategory Evaluation ---
true_sub_labels_indices = val_df['subcategory_int_label'].values

predicted_sub_labels_indices = []
for text_val in val_df['text'].values:
    preds = predict(text_val)
    predicted_class_name = max(preds["subcategory_predictions"], key=preds["subcategory_predictions"].get)
    predicted_sub_labels_indices.append(sub_label_binarizer.classes_.tolist().index(predicted_class_name))

predicted_sub_labels_indices = np.array(predicted_sub_labels_indices)

all_present_sub_label_indices = np.unique(np.concatenate((true_sub_labels_indices, predicted_sub_labels_indices)))
target_names_for_sub_report = [sub_label_binarizer.classes_[idx] for idx in all_present_sub_label_indices]

cm_sub = confusion_matrix(true_sub_labels_indices, predicted_sub_labels_indices, labels=all_present_sub_label_indices)

accuracy_sub = accuracy_score(true_sub_labels_indices, predicted_sub_labels_indices)
precision_sub = precision_score(true_sub_labels_indices, predicted_sub_labels_indices, average='weighted', zero_division=0, labels=all_present_sub_label_indices)
recall_sub = recall_score(true_sub_labels_indices, predicted_sub_labels_indices, average='weighted', zero_division=0, labels=all_present_sub_label_indices)
f1_sub = f1_score(true_sub_labels_indices, predicted_sub_labels_indices, average='weighted', zero_division=0, labels=all_present_sub_label_indices)

print("\n--- Subcategory Evaluation ---")
print(f"Accuracy (Subcategory): {accuracy_sub:.4f}")
print(f"Precision (Subcategory, weighted): {precision_sub:.4f}")
print(f"Recall (Subcategory, weighted): {recall_sub:.4f}")
print(f"F1-score (Subcategory, weighted): {f1_sub:.4f}")
print("\nClassification Report (Subcategory):")
print(classification_report(true_sub_labels_indices, predicted_sub_labels_indices,
                             labels=all_present_sub_label_indices,
                             target_names=target_names_for_sub_report,
                             zero_division=0))

plt.figure(figsize=(16, 14))
sns.heatmap(cm_sub, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names_for_sub_report,
            yticklabels=target_names_for_sub_report)
plt.xlabel('Predicted Sub Labels')
plt.ylabel('True Sub Labels')
plt.title('Confusion Matrix - Subcategories')
plt.xticks(rotation=90, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
sample_text = "I fight bears"
# --- Example Prediction and Visualization ---

prediction_results = predict(sample_text)
print(f"Text: '{sample_text}'")

print("\nMain Category Predictions:")
for category, percentage in prediction_results["main_category_predictions"].items():
    print(f"- {category}: {percentage:.2f}%")

print("\nSubcategory Predictions:")
for category, percentage in prediction_results["subcategory_predictions"].items():
    print(f"- {category}: {percentage:.2f}%")

# Plotting the main category prediction results for the sample text.
main_categories_for_plot = list(prediction_results["main_category_predictions"].keys())
main_percentages_for_plot = list(prediction_results["main_category_predictions"].values())

plt.figure(figsize=(10, 6)) # Adjusted figure size
plt.bar(main_categories_for_plot, main_percentages_for_plot, color='skyblue')
plt.xlabel("Main Categories")
plt.ylabel("Probability (%)")
#plt.title(f"Predicted Main Category Probabilities for:\n'{sample_text}'")
plt.xticks(rotation=45, ha="right") # Rotate x-axis labels for better visibility
plt.tight_layout() # Adjust layout to prevent labels from overlapping
for i, percentage in enumerate(main_percentages_for_plot):
    plt.text(i, percentage + 0.5, f"{percentage:.2f}%", ha="center", va="bottom", fontsize=8)
plt.show()

# Plotting the subcategory prediction results for the sample text.
sub_categories_for_plot = list(prediction_results["subcategory_predictions"].keys())
sub_percentages_for_plot = list(prediction_results["subcategory_predictions"].values())

plt.figure(figsize=(12, 8)) # Adjusted figure size
plt.bar(sub_categories_for_plot, sub_percentages_for_plot, color='lightcoral')
plt.xlabel("Subcategories")
plt.ylabel("Probability (%)")
#plt.title(f"Predicted Subcategory Probabilities for:\n'{sample_text}'")
plt.xticks(rotation=90, ha="right") # Rotate x-axis labels for better visibility
plt.tight_layout() # Adjust layout to prevent labels from overlapping
for i, percentage in enumerate(sub_percentages_for_plot):
    plt.text(i, percentage + 0.5, f"{percentage:.2f}%", ha="center", va="bottom", fontsize=8)
plt.show()
