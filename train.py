import pandas as pd
import jieba.posseg as pseg
import keras
from keras import Input
from keras.layers import Embedding, LSTM, concatenate, Dense
from keras.models import Model
from keras.utils import plot_model
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

TRAIN_CSV_PATH = 'fake-news-pair-classification-challenge/train.csv'
TEST_CSV_PATH = 'fake-news-pair-classification-challenge/test.csv'
MAX_NUM_WORDS = 10000
tokenizer = keras.preprocessing.text.Tokenizer(
    num_words=MAX_NUM_WORDS)
MAX_SEQUENCE_LENGTH = 20
VALIDATION_RATIO = 0.1
# 小彩蛋
RANDOM_STATE = 9527
NUM_EMBEDDING_DIM = 256
# LSTM 輸出的向量維度
NUM_LSTM_UNITS = 128
# 定義每一個分類對應到的索引數字
label_to_index = {
    'unrelated': 0,
    'agreed': 1,
    'disagreed': 2
}
NUM_CLASSES = 3


def main():
    train = pd.read_csv(TRAIN_CSV_PATH, index_col=0, nrows=1000)
    cols = ['title1_zh', 'title2_zh', 'label']
    train = train.loc[:, cols]
    print(train.head(3))

    train['title1_tokenized'] = train.loc[:, 'title1_zh'].astype(str).apply(jieba_tokenizer)
    train['title2_tokenized'] = train.loc[:, 'title2_zh'].astype(str).apply(jieba_tokenizer)

    # print(train.iloc[:, [0, 3]].head())
    # print(train.iloc[:, [1, 4]].head())
    corpus_x1 = train.title1_tokenized
    corpus_x2 = train.title2_tokenized
    corpus = pd.concat([corpus_x1, corpus_x2])
    print(corpus.shape)
    # print(pd.DataFrame(corpus.iloc[:5], columns=['title']))
    tokenizer.fit_on_texts(corpus)
    x1_train = tokenizer.texts_to_sequences(corpus_x1)
    x2_train = tokenizer.texts_to_sequences(corpus_x2)
    # print(x1_train[:1])
    # for seq in x1_train[:1]:
    #     print([tokenizer.index_word[idx] for idx in seq])

    # for seq in x1_train[:10]:
    #     print(len(seq), seq[:5], ' ...')

    x1_train = keras.preprocessing.sequence.pad_sequences(x1_train, maxlen=MAX_SEQUENCE_LENGTH)
    x2_train = keras.preprocessing.sequence.pad_sequences(x2_train, maxlen=MAX_SEQUENCE_LENGTH)
    # print(x1_train[:5])

    # 將分類標籤對應到剛定義的數字
    y_train = train.label.apply(lambda x: label_to_index[x])
    y_train = np.asarray(y_train).astype('float32')
    y_train = keras.utils.to_categorical(y_train)
    # print(y_train[:5])
    x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(x1_train, x2_train, y_train,
                                                                          test_size=VALIDATION_RATIO,
                                                                          random_state=RANDOM_STATE)
    # print("Training Set")
    # print("-" * 10)
    # print(f"x1_train: {x1_train.shape}")
    # print(f"x2_train: {x2_train.shape}")
    # print(f"y_train : {y_train.shape}")

    # print("-" * 10)
    # print(f"x1_val:   {x1_val.shape}")
    # print(f"x2_val:   {x2_val.shape}")
    # print(f"y_val :   {y_val.shape}")
    # print("-" * 10)
    # print("Test Set")

    # 分別定義 2 個新聞標題 A & B 為模型輸入
    # 兩個標題都是一個長度為 20 的數字序列
    top_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    bm_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    shared_lstm = LSTM(NUM_LSTM_UNITS)
    # for i, seq in enumerate(x1_train[:5]):
    #     print(f"新聞標題 {i + 1}: ")
    #     print([tokenizer.index_word.get(idx, '') for idx in seq])
    #     print()
    embedding_layer = Embedding(MAX_NUM_WORDS, NUM_EMBEDDING_DIM)
    top_embedded = embedding_layer(top_input)
    bm_embedded = embedding_layer(bm_input)
    top_output = shared_lstm(top_embedded)
    bm_output = shared_lstm(bm_embedded)
    merged = concatenate([top_output, bm_output], axis=-1)
    dense = Dense(units=NUM_CLASSES, activation='softmax')
    predictions = dense(merged)
    model = Model(inputs=[top_input, bm_input], outputs=predictions)
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False, rankdir='LR')
    # print(model.summary())
    img = plt.imread('model.png')
    plt.imshow(img)
    plt.show()
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    history = model.fit(
        # 輸入是兩個長度為 20 的數字序列
        x=[x1_train, x2_train],
        y=y_train,
        batch_size=512,
        epochs=10,
        # 每個 epoch 完後計算驗證資料集
        # 上的 Loss 以及準確度
        validation_data=(
            [x1_val, x2_val],
            y_val
        ),
        # 每個 epoch 隨機調整訓練資料集
        # 裡頭的數據以讓訓練過程更穩定
        shuffle=True
    )

    test = pd.read_csv(TEST_CSV_PATH, index_col=0, nrows=100)
    # print(test.head(3))
    # 文本斷詞 / Word Segmentation
    test['title1_tokenized'] = \
        test.loc[:, 'title1_zh'] \
            .apply(jieba_tokenizer)
    test['title2_tokenized'] = \
        test.loc[:, 'title2_zh'] \
            .apply(jieba_tokenizer)

    # 將詞彙序列轉為索引數字的序列
    x1_test = tokenizer \
        .texts_to_sequences(
        test.title1_tokenized)
    x2_test = tokenizer \
        .texts_to_sequences(
        test.title2_tokenized)

    # 為數字序列加入 zero padding
    x1_test = keras \
        .preprocessing \
        .sequence \
        .pad_sequences(
        x1_test,
        maxlen=MAX_SEQUENCE_LENGTH)
    x2_test = keras \
        .preprocessing \
        .sequence \
        .pad_sequences(
        x2_test,
        maxlen=MAX_SEQUENCE_LENGTH)

    # 利用已訓練的模型做預測
    predictions = model.predict([x1_test, x2_test])
    print(predictions[:50])

    index_to_label = {v: k for k, v in label_to_index.items()}

    test['Category'] = [index_to_label[idx] for idx in np.argmax(predictions, axis=1)]

    submission = test.loc[:, ['Category']].reset_index()

    submission.columns = ['Id', 'Category']
    print(submission.head(10))


def jieba_tokenizer(text):
    if text == '':
        return text
    words = pseg.cut(text)
    r = ' '.join([word for word, flag in words if flag != 'x'])
    # print(r)
    return r


if __name__ == "__main__":
    # execute only if run as a script
    main()
