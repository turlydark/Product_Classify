from data import data_loader
from data import data_processing
from model import built_attention_model
from keras.losses import categorical_crossentropy
from keras.utils import multi_gpu_model


def get_model():
    data_processing.data_diastract()
    (train_processed, train_label), (test_processed, test_label) = data_loader(test_rate=0.01)
    model = built_attention_model()

    # model = multi_gpu_model(model, gpus=2)

    model.compile(loss=categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    print(train_processed.shape)
    print(train_label.shape)
    print(test_processed.shape)
    print(test_label.shape)
    model.fit(train_processed,
              train_label,
              batch_size=256,
              epochs=18,
              validation_data=(test_processed, test_label))
    return model

    # model.save('attention.h5')
