import argparse
from pathlib import Path
import time, re

parser = argparse.ArgumentParser(description="Train an LSTM for IMDb sentiment classification")
parser.add_argument("--data", type=Path, default=Path("data/imdb_full.csv"))
parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
args = parser.parse_args()
if not args.data.is_file():
    parser.error(f"Dataset not found: {args.data}")
args.output_dir.mkdir(parents=True, exist_ok=True)
t_start = time.time()
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

# 1. LOAD full 50k combined dataset, re-split 40k/10k (matches original project's split)
df = pd.read_csv(args.data)
if not {"Reviews", "Sentiment"}.issubset(df.columns):
    raise ValueError("CSV must contain Reviews and Sentiment columns")
if len(df) != 50000 or df[["Reviews", "Sentiment"]].isna().any().any():
    raise ValueError("Expected 50,000 reviews with no missing reviews or labels")
if set(df["Sentiment"].unique()) != {"neg", "pos"}:
    raise ValueError("Sentiment labels must be neg and pos")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean'] = df['Reviews'].apply(clean_text)
df['label'] = (df['Sentiment'] == 'pos').astype(int)

train_df, test_df = train_test_split(df, test_size=10000, train_size=40000,
                                      stratify=df['label'], random_state=SEED)
print(f"Train: {len(train_df)}, Test: {len(test_df)}  (t={time.time()-t_start:.1f}s)")

# 2. TOKENIZE
VOCAB_SIZE = 10000
MAX_LEN = 120

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(train_df['clean'])

X_train = pad_sequences(tokenizer.texts_to_sequences(train_df['clean']), maxlen=MAX_LEN, padding='post', truncating='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(test_df['clean']), maxlen=MAX_LEN, padding='post', truncating='post')
y_train = train_df['label'].values
y_test = test_df['label'].values
print(f"Tokenized. X_train {X_train.shape}  (t={time.time()-t_start:.1f}s)")

# 3. MODEL - sized to train fast on CPU while still being a real LSTM
model = Sequential([
    Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),
    LSTM(32, dropout=0.2, recurrent_dropout=0.2),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. TRAIN
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=4,
    batch_size=256,
    verbose=2
)
print(f"Training done (t={time.time()-t_start:.1f}s)")

# 5. EVALUATE
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTEST ACCURACY: {test_acc*100:.2f}%")
print(f"TEST LOSS: {test_loss:.4f}")

y_pred = (model.predict(X_test, verbose=0) >= 0.5).astype(int).flatten()
print(classification_report(y_test, y_pred, target_names=['neg','pos'], digits=4))
print(confusion_matrix(y_test, y_pred))

# 6. PLOTS
plt.figure(figsize=(11,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], marker='o', label='Train Acc')
plt.plot(history.history['val_accuracy'], marker='o', label='Val Acc')
plt.axhline(test_acc, color='red', linestyle='--', label=f'Test Acc={test_acc*100:.2f}%')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('LSTM Accuracy'); plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], marker='o', label='Train Loss')
plt.plot(history.history['val_loss'], marker='o', label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('LSTM Loss'); plt.legend()
plt.tight_layout()
plt.savefig(args.output_dir / 'lstm_training_curves.png', dpi=120)
plt.close()

with open(args.output_dir / 'sentiment_results_summary.txt','w') as f:
    f.write("SENTIMENT ANALYSIS - IMDB REVIEWS - LSTM RESULTS\n" + "="*55 + "\n\n")
    f.write(f"Train: {len(train_df)}, Test: {len(test_df)} (re-split from 50K combined reviews)\n")
    f.write(f"Vocab size: {VOCAB_SIZE}, Max sequence length: {MAX_LEN}\n\n")
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\nTest Loss: {test_loss:.4f}\n\n")
    f.write(classification_report(y_test, y_pred, target_names=['neg','pos'], digits=4))

print(f"\nALL DONE (t={time.time()-t_start:.1f}s)")
