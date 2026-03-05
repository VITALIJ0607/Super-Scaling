import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt


FILES_PATH = "flowers/"
SCALE = 4                  
CROP_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 100


def map_function(y):
    x= tf.image.resize(y, (64,64))
    return x / 255, y / 255


def load_images(path):
    images = tf.keras.utils.image_dataset_from_directory(
    FILES_PATH, # Pfad zu den Bildern
    labels='inferred', # Labels aus den Dateiname ableiten
    # 'int' oder 'categorical' je nach Aufgabe Regression oder Klassifikation
    label_mode=None,
    class_names=None, # Man kann optional eine Liste der Klassennamen übergeben
    color_mode='rgb', # Bilder farbig ausgeben. Alternativ: 'grayscale', 'rgba'
    # Wie viele Bilder auf einmal von der Festplatte geladen werden
    batch_size=BATCH_SIZE,
    image_size=(CROP_SIZE, CROP_SIZE), # Biler auf die angegebene Größe sklaieren
    shuffle=True, # Vor jeder Epoche die Bilder durchmischen
    # if not None Durchmischen in bestimmter Reihenfolge durchführen
    seed=0,
    validation_split=0.2, # Gibt den Anteil der Validationdaten an
    # Gibt an, ob man die Trainigs- oder Validationdaten bekommen möchte
    subset='training',
    interpolation='bilinear', # Wie die Skalierung durchfgeführt wird
    follow_links=False, # Ordnerstruktur nachverfolgen
    # Auf die Bildmitte zuschneiden um das Seitenverhältnis nicht zu verändern
    crop_to_aspect_ratio=False)
    return images.map(map_function)


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


def build_unet(input_shape=(int(CROP_SIZE / SCALE), int(CROP_SIZE / SCALE), 3)):
    inputs = layers.Input(shape=input_shape)

    # ---- Encoder ----
    # 64x64x64
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D()(c1)
    # 32x32x128
    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D()(c2)
    # 16x16x256
    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D()(c3)
    # 8x8x512
    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D()(c4)

    # ---- Bottleneck ----
    # 4x4x1024
    bn = conv_block(p4, 1024)

    # ---- Decoder ----
    # 8x8x512
    u1 = layers.Conv2DTranspose(512, 2, strides=2, padding="same")(bn)
    u1 = layers.Concatenate()([u1, c4])
    c5 = conv_block(u1, 512)
    # 16x16x256
    u2 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(c5)
    u2 = layers.Concatenate()([u2, c3])
    c6 = conv_block(u2, 256)
    # 32x32x128
    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(c6)
    u3 = layers.Concatenate()([u3, c2])
    c7 = conv_block(u3, 128)
    # 64x64x64
    u4 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c7)
    u4 = layers.Concatenate()([u4, c1])
    c8 = conv_block(u4, 64)
    # 128x128x64
    u5 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c8)
    c9 = conv_block(u5, 64)
    # 256x256x64
    u6 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c9)
    c10 = conv_block(u6, 64)
    
    outputs = layers.Conv2D(3, 1, activation="sigmoid")(c10)

    return Model(inputs, outputs)


def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


def train():
    images = load_images(FILES_PATH)
    model = build_unet()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mae",
        metrics=[psnr_metric, ssim_metric]
    )

    print(model.summary())

    early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='loss', # Metrik, die überwacht wird
    patience=10,             # Anzahl der Epochen ohne Verbesserung
    restore_best_weights=True # bestes Modell wiederherstellen
    )
    
    history = model.fit(images, epochs=EPOCHS, callbacks=[early_stop])
    
    # Training History plotten
    plt.figure(figsize=(15, 5))
    
    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'])
    plt.title('Model Loss (MAE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # PSNR
    plt.subplot(1, 3, 2)
    plt.plot(history.history['psnr_metric'])
    plt.title('PSNR Metric')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    
    # SSIM
    plt.subplot(1, 3, 3)
    plt.plot(history.history['ssim_metric'])
    plt.title('SSIM Metric')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("Training History Plot gespeichert als training_history.png")
    plt.show()
    
    loss = model.evaluate(images)
    print(f"Evaluations Loss: {loss}")
    model.save("unet_sr.h5")
    print("Modell gespeichert als unet_sr.h5")

    return model


if __name__ == "__main__":
    model = train()
