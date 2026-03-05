import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from super_resolution import SCALE, CROP_SIZE
import glob
import os


IMG_DIR = "test_images/"
MODEL_PATH = "unet_sr.h5"


def process_image(img_path, model):
    """Lädt ein Bild, erzeugt LR-Version und wendet Super-Resolution an"""
    # HR Bild laden
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    hr = tf.image.convert_image_dtype(img, tf.float32)
    
    # Auf CROP_SIZE skalieren (wie im Training)
    hr = tf.image.resize(hr, (CROP_SIZE, CROP_SIZE), method="bicubic")
    
    # LR erzeugen
    lr_small = tf.image.resize(hr, (CROP_SIZE // SCALE, CROP_SIZE // SCALE), method="bicubic")
    lr_up = tf.image.resize(lr_small, (CROP_SIZE, CROP_SIZE), method="bicubic")
    
    # Batch-Dimension hinzufügen für Modell
    lr_batch = tf.expand_dims(lr_small, axis=0)  # lr_small verwenden, nicht lr_up!
    
    # Modell anwenden
    pred = model.predict(lr_batch, verbose=0)[0]
    pred = np.clip(pred, 0, 1)
    
    return lr_small, lr_up, pred, hr


class ImageBrowser:
    def __init__(self, image_paths, model):
        self.image_paths = image_paths
        self.model = model
        self.current_idx = 0
        self.num_images = len(image_paths)
        
        # Figure und Axes erstellen
        self.fig, self.axes = plt.subplots(1, 4, figsize=(20, 5))
        self.fig.subplots_adjust(bottom=0.15)
        
        # Navigation Buttons
        ax_prev = plt.axes([0.3, 0.02, 0.1, 0.05])
        ax_next = plt.axes([0.6, 0.02, 0.1, 0.05])
        
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        
        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)
        
        # Erstes Bild anzeigen
        self.update_plot()
    
    def update_plot(self):
        """Aktualisiert den Plot mit dem aktuellen Bild"""
        img_path = self.image_paths[self.current_idx]
        lr_small, lr_up, pred, hr = process_image(img_path, self.model)
        
        # Alle Axes leeren
        for ax in self.axes:
            ax.clear()
            ax.axis("off")
        
        # Low-Res (small)
        self.axes[0].imshow(lr_small.numpy())
        self.axes[0].set_title(f"Low-Res small ({self.current_idx + 1}/{self.num_images})\n{os.path.basename(img_path)}")
        
        # Original High-Res
        self.axes[1].imshow(hr.numpy())
        self.axes[1].set_title("Original High-Res")
        
        # Bicubic upscaled
        self.axes[2].imshow(lr_up.numpy())
        self.axes[2].set_title("Bicubic upscaled")
        
        # U-Net Super Resolution
        self.axes[3].imshow(pred)
        self.axes[3].set_title("U-Net Super Resolution")
        
        self.fig.canvas.draw()
    
    def next_image(self, event):
        """Zeigt das nächste Bild"""
        if self.current_idx < self.num_images - 1:
            self.current_idx += 1
            self.update_plot()
    
    def prev_image(self, event):
        """Zeigt das vorherige Bild"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_plot()


if __name__ == "__main__":
    
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Alle Bilder im Verzeichnis laden
    image_paths = glob.glob(os.path.join(IMG_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(IMG_DIR, "*.jpeg"))
    
    if len(image_paths) == 0:
        print("Keine Bilder gefunden!")
    else:
        # Image Browser erstellen
        browser = ImageBrowser(image_paths, model)
        plt.show()
