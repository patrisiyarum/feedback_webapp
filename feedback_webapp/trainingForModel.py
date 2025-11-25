# Full Model for Subcategory Classification (8 labels)
# Excel loading, text cleaning, deduplication, TRAIN-ONLY augmentation, keyword features,
# correct evaluation, top-2/top-3 accuracy, and per-class counts.

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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import os
import random
import re  # for regex string cleaning

# --- Set Random Seeds for Reproducibility ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

warnings.filterwarnings('ignore')

# --- Configuration ---
EPOCHS = 20
BATCH_SIZE = 64
file_path = '/kaggle/input/thisisthelastonetoday/balanced_100_each_plus50_per_class.csv'

# ==========================================
# --- DATA LOADING & BASIC CLEANING ---
# ==========================================
print("--- Loading Data ---")

try:
    df = pd.read_csv(file_path)
    print("Successfully read Excel file.")
except Exception as e:
    raise ValueError(f"Could not read Excel file. Error: {e}")

# Clean column headers
print(f"Original Columns: {df.columns.tolist()}")
df.columns = (
    df.columns.astype(str)
      .str.replace(r'[^\w\s]', '', regex=True)
      .str.strip()
      .str.lower()
)
print(f"Cleaned Columns:  {df.columns.tolist()}")

# Standardize names: text / subcategory
if 'text' not in df.columns:
    if len(df.columns) >= 1:
        print(f"Mapping column 0 ('{df.columns[0]}') to 'text'")
        df.rename(columns={df.columns[0]: 'text'}, inplace=True)

if 'subcategory' not in df.columns:
    found = False
    for col in df.columns:
        if 'cat' in col or 'label' in col:
            print(f"Mapping column '{col}' to 'subcategory'")
            df.rename(columns={col: 'subcategory'}, inplace=True)
            found = True
            break
    if not found and len(df.columns) >= 2:
        print(f"Mapping column 1 ('{df.columns[1]}') to 'subcategory'")
        df.rename(columns={df.columns[1]: 'subcategory'}, inplace=True)

print(f"Final Columns:    {df.columns.tolist()}")
if 'text' not in df.columns or 'subcategory' not in df.columns:
    raise KeyError(
        "Could not identify 'text' and 'subcategory' columns even after cleaning."
    )

df.dropna(subset=['text', 'subcategory'], inplace=True)

# Text cleaning
def clean_text(t: str) -> str:
    t = str(t)
    t = t.replace('\r', ' ').replace('\n', ' ')
    t = re.sub(r'other/comments:\s*', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'\bY\b', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    t = t.strip().lower()
    return t

df['subcategory'] = df['subcategory'].astype(str).apply(lambda x: x.strip().title())
df['text'] = df['text'].astype(str).apply(clean_text)

# ==========================================
# --- DEDUPLICATION & CONFLICT REMOVAL ---
# ==========================================
print("\n--- Deduplication & Conflict Handling ---")

before = len(df)
df.drop_duplicates(subset=['text', 'subcategory'], inplace=True)
after = len(df)
print(f"Dropped {before - after} exact duplicate rows.")

conflict_counts = df.groupby('text')['subcategory'].nunique()
conflict_texts = conflict_counts[conflict_counts > 1].index
num_conflicts = len(conflict_texts)
print(f"Found {num_conflicts} texts with conflicting labels.")
if num_conflicts > 0:
    df = df[~df['text'].isin(conflict_texts)]
    print(f"Dropped {num_conflicts} conflicting text groups. New size: {len(df)}")

df = df.reset_index(drop=True)

# ==========================================
# --- LABEL ENCODING ---
# ==========================================
print("\nEncoding Labels...")
label_encoder = sklearn.preprocessing.LabelEncoder()
df['int_label'] = label_encoder.fit_transform(df['subcategory'])
NUM_CLASSES = len(label_encoder.classes_)

print(f"Total Unique Subcategories: {NUM_CLASSES}")
print(label_encoder.classes_)

# ==========================================
# --- STRATIFIED TRAIN/VAL SPLIT ---
# ==========================================
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df['int_label']
)
train_df = train_df.copy()
val_df = val_df.copy()

print(f"\nTraining Size (before aug): {len(train_df)}")
print(f"Validation Size:            {len(val_df)}")

# Per-class counts BEFORE augmentation
orig_train_counts = train_df['int_label'].value_counts().sort_index()

# Save original labels for class weights
base_train_labels = train_df['int_label'].values

# ==========================================
# --- SYNONYM-BASED AUGMENTATION (TRAIN ONLY) ---
# ==========================================
print("\n--- Augmenting Training Data ---")

SYNONYM_RULES = [
    (r'\bpassengers\b', ['customers', 'pax', 'guests']),
    (r'\bpassenger\b', ['customer', 'pax', 'guest']),
    (r'\bflight attendants\b', ['fas', 'cabin crew', 'flight crew']),
    (r'\bflight attendant\b', ['fa', 'cabin crew member']),
    (r'\bcrew meals\b', ['staff meals', 'pilot meals']),
    (r'\bmeal\b', ['dish', 'entree']),
    (r'\bmeals\b', ['dishes', 'entrees']),
    (r'\bbeverages\b', ['drinks', 'refreshments']),
    (r'\bbeverage\b', ['drink', 'refreshment']),
    (r'\bwater bottles\b', ['bottled water', 'small waters']),
    (r'\bwater bottle\b', ['bottled water', 'small water']),
    (r'\bamenity kits\b', ['amenity bags', 'travel kits']),
    (r'\bamenity kit\b', ['amenity bag', 'travel kit']),
    (r'\bslippers\b', ['house shoes', 'cabin slippers']),
    (r'\bcatering\b', ['kitchen', 'food service']),
    (r'\bnot boarded\b', ['never loaded', 'missing from load']),
    (r'\bmissing\b', ['not available', 'unavailable', 'short']),
    (r'\binsufficient\b', ['not enough', 'shortage of']),
]

def augment_text_once(text: str) -> str:
    """Apply random synonym substitutions to create one variant."""
    new_text = text
    for pattern, replacements in SYNONYM_RULES:
        if random.random() < 0.4 and re.search(pattern, new_text):
            new_text = re.sub(pattern, random.choice(replacements), new_text)
    return new_text

def augment_row(row, n_aug=2):
    """Generate up to n_aug augmented versions of a row's text."""
    original = row['text']
    variants = set()
    for _ in range(n_aug):
        t = augment_text_once(original)
        if t != original:
            variants.add(t)
    return list(variants)

aug_rows = []
for _, row in train_df.iterrows():
    new_texts = augment_row(row, n_aug=3)
    for t in new_texts:
        new_row = row.copy()
        new_row['text'] = t
        aug_rows.append(new_row)

aug_df = pd.DataFrame(aug_rows)
print(f"Generated {len(aug_df)} augmented rows.")

# Per-class AUGMENTED counts
aug_counts = aug_df['int_label'].value_counts().sort_index()

# Combine original + augmented training data
train_df = pd.concat([train_df, aug_df], ignore_index=True)
train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

print(f"Training Size (after aug):  {len(train_df)}")

# Per-class TOTAL counts after augmentation
final_train_counts = train_df['int_label'].value_counts().sort_index()

print("\nPer-class training counts (orig / aug / total):")
for i, label in enumerate(label_encoder.classes_):
    orig = int(orig_train_counts.get(i, 0))
    aug = int(aug_counts.get(i, 0))
    total = int(final_train_counts.get(i, 0))
    print(f"{i} - {label}: orig={orig}, aug={aug}, total={total}")

# ==========================================
# --- KEYWORD FEATURES (AFTER AUGMENTATION) ---
# ==========================================
print("\n--- Building Keyword Features ---")

BEVERAGE_WORDS = ["water", "beverage", "drink", "juice", "soda", "coffee", "tea", "bottle", "bottles"]
SERVICE_ITEM_WORDS = [
    "utensil", "fork", "knife", "spoon", "napkin",
    "slipper", "amenity kit", "amenity bag", "pillow",
    "blanket", "glass", "tray setup", "tray"
]
CATERING_WORDS = [
    "catering", "kitchen", "boarded", "not boarded",
    "loaded", "never loaded", "improperly loaded",
    "missing meals", "no second meal", "not catered"
]
CREW_WORDS = [
    "flight attendant", "attendant", "fa", "cabin crew",
    "steward", "staff", "crew did not provide", "not offered"
]

def keyword_features_from_text(t: str) -> dict:
    t = t.lower()
    has_beverage = int(any(w in t for w in BEVERAGE_WORDS))
    has_service_item = int(any(w in t for w in SERVICE_ITEM_WORDS))
    has_catering = int(any(w in t for w in CATERING_WORDS))
    has_crew = int(any(w in t for w in CREW_WORDS))
    return {
        "kw_beverage": has_beverage,
        "kw_service_item": has_service_item,
        "kw_catering": has_catering,
        "kw_crew": has_crew,
    }

for subset in (train_df, val_df):
    kw_df = subset['text'].apply(keyword_features_from_text).apply(pd.Series)
    for col in kw_df.columns:
        subset[col] = kw_df[col].astype('float32')

KW_COLS = ["kw_beverage", "kw_service_item", "kw_catering", "kw_crew"]
print("Keyword feature sample (train):\n", train_df[KW_COLS].head())

# ==========================================
# --- CLASS WEIGHTS (from ORIGINAL train only) ---
# ==========================================
print("\n--- Class Weights ---")
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(NUM_CLASSES),
    y=base_train_labels  # ORIGINAL labels before augmentation
)
class_weight_dict = dict(zip(np.arange(NUM_CLASSES), class_weights_array))
print(class_weight_dict)

train_df['sample_weight'] = train_df['int_label'].map(class_weight_dict)

# ==========================================
# --- TF DATASETS ---
# ==========================================
def dataframe_to_tf_dataset(dataframe, include_sample_weights=False, shuffle=True):
    dataframe = dataframe.copy()
    text_feat = dataframe.pop('text')
    labels = tf.cast(dataframe.pop('int_label'), tf.int32)
    kw_feats = dataframe[KW_COLS].astype('float32').values

    if include_sample_weights:
        weights = tf.cast(dataframe.pop('sample_weight'), tf.float32)
        ds = tf.data.Dataset.from_tensor_slices(((text_feat, kw_feats), labels, weights))
    else:
        ds = tf.data.Dataset.from_tensor_slices(((text_feat, kw_feats), labels))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(text_feat),
            seed=SEED,
            reshuffle_each_iteration=True
        )
    return ds

train_ds = (
    dataframe_to_tf_dataset(train_df, include_sample_weights=True, shuffle=True)
      .batch(BATCH_SIZE)
      .cache()
      .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    dataframe_to_tf_dataset(val_df, include_sample_weights=False, shuffle=False)
      .batch(BATCH_SIZE)
      .cache()
      .prefetch(tf.data.AUTOTUNE)
)

# ==========================================
# --- MODEL: BERT + KEYWORD FEATURES ---
# ==========================================
print("\nBuilding Model...")

text_input = tf.keras.Input(shape=(), name='text', dtype='string')
kw_input = tf.keras.Input(shape=(len(KW_COLS),), name='kw_features', dtype='float32')

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

encoder_outputs = encoder(encoder_inputs)
pooled_output = encoder_outputs['pooled_output']

concat = tf.keras.layers.Concatenate(name='concat_features')([pooled_output, kw_input])
x = tf.keras.layers.Dropout(0.20, name='dropout')(concat)

model_output = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation='softmax',
    name='subcategory_output'
)(x)

model = tf.keras.Model(inputs=[text_input, kw_input], outputs=model_output)

optimizer = tf.keras.optimizers.Adamax(learning_rate=3e-4)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

class OverfittingEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=5, min_delta=0.001, monitor='val_loss'):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_loss = np.inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get(self.monitor)
        if current_val_loss is None:
            return

        print(f"Epoch {epoch+1} Val Loss: {current_val_loss:.4f}")

        if current_val_loss < self.best_loss - self.min_delta:
            self.best_loss = current_val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Stopping early. No improvement for {self.patience} epochs.")
                self.model.stop_training = True

# ==========================================
# --- TRAINING ---
# ==========================================
print("\nStarting Training...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[OverfittingEarlyStopping(patience=4)]
)

model.save('subcategory_model_augmented.keras')
with open('subcategory_classes.json', 'w') as f:
    json.dump(label_encoder.classes_.tolist(), f)
print("\nModel and classes saved.")

# ==========================================
# --- EVALUATION (with Top-2 / Top-3 accuracy) ---
# ==========================================
print("\n--- Evaluation ---")
true_labels = []
pred_labels = []
top2_correct = 0
top3_correct = 0
total = 0

for (text_batch, kw_batch), label_batch in val_ds:
    probs = model.predict([text_batch, kw_batch], verbose=0)
    top1 = np.argmax(probs, axis=1)
    # indices of 2 and 3 highest probs
    top2 = np.argsort(probs, axis=1)[:, -2:]
    top3 = np.argsort(probs, axis=1)[:, -3:]

    labels_np = label_batch.numpy()
    for i, true_label in enumerate(labels_np):
        total += 1
        true_labels.append(true_label)
        pred_labels.append(top1[i])

        if true_label in top2[i]:
            top2_correct += 1
        if true_label in top3[i]:
            top3_correct += 1

true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

top1_accuracy = (pred_labels == true_labels).mean()
top2_accuracy = top2_correct / total
top3_accuracy = top3_correct / total

print(f"\nTop-1 Accuracy: {top1_accuracy:.3f}")
print(f"Top-2 Accuracy: {top2_accuracy:.3f}")
print(f"Top-3 Accuracy: {top3_accuracy:.3f}")

unique_labels = np.unique(np.concatenate((true_labels, pred_labels)))
target_names = [label_encoder.classes_[i] for i in unique_labels]

print(
    "\nClassification Report (Top-1 predictions):\n",
    classification_report(
        true_labels,
        pred_labels,
        labels=unique_labels,
        target_names=target_names,
        zero_division=0
    )
)

cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=target_names,
    yticklabels=target_names
)
plt.title('Confusion Matrix')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ==========================================
# --- SINGLE PREDICTION ---
# ==========================================
def keyword_features_for_single(text_str: str) -> np.ndarray:
    feats = keyword_features_from_text(clean_text(text_str))
    return np.array([[feats[c] for c in KW_COLS]], dtype='float32')

def predict_single(text_str: str):
    clean_str = clean_text(text_str)
    kw_vec = keyword_features_for_single(clean_str)
    probs = model.predict([np.array([clean_str]), kw_vec], verbose=0)[0] * 100
    results = {label_encoder.classes_[i]: float(probs[i]) for i in range(len(probs))}
    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

sample = "Passenger complained about missing vegan meal and no water was loaded"
print(f"\nPrediction for: '{sample}'")
res = predict_single(sample)
for k, v in list(res.items())[:5]:
    print(f"- {k}: {v:.2f}%")
