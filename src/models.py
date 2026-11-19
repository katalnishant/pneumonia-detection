"""Model zoo: custom CNN, the original VGG16 v1 recipe, and four fine-tuned ImageNet backbones.

All models
  * take float images in [0, 255] with shape (S, S, 3): preprocessing lives inside the model;
  * are keras.Sequential: [preprocess] -> [backbone or conv blocks] -> [head].
    Grad-CAM and MC-dropout simply walk model.layers (see gradcam.py / predict.py);
  * output sigmoid(1) for the binary task or softmax(3) for the 3-class task.
"""
from __future__ import annotations

import keras
from keras import layers

from src import config

IMAGENET_MEAN_BGR = [103.939, 116.779, 123.68]
TORCH_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
TORCH_VAR = [(0.229 * 255) ** 2, (0.224 * 255) ** 2, (0.225 * 255) ** 2]

# name -> (constructor, preprocessing mode, layer-name prefixes unfrozen in phase 2)
BACKBONES = {
    "vgg16": (keras.applications.VGG16, "caffe", ("block5",)),
    "vgg16_v1": (keras.applications.VGG16, "caffe", ()),
    "resnet50": (keras.applications.ResNet50, "caffe", ("conv5",)),
    "densenet121": (keras.applications.DenseNet121, "torch", ("conv5",)),
    "efficientnetb0": (keras.applications.EfficientNetB0, "none", ("block6", "block7", "top")),
}


def preprocess_layer(mode: str):
    """Reproduces keras.applications.<net>.preprocess_input as a serialisable layer.
    'caffe' = RGB->BGR + ImageNet mean subtraction. Our X-rays have 3 identical channels,
    so the channel swap is a no-op and a fixed Normalization layer is exact.
    'torch' = scale to [0,1] then ImageNet mean/std. 'none' = model normalises internally."""
    if mode == "caffe":
        return layers.Normalization(mean=IMAGENET_MEAN_BGR, variance=[1.0] * 3, name="preprocess")
    if mode == "torch":
        return layers.Normalization(mean=TORCH_MEAN, variance=TORCH_VAR, name="preprocess")
    if mode == "rescale":
        return layers.Rescaling(1.0 / 255, name="preprocess")
    return None


def output_layer(task: str):
    if task == "binary":
        return layers.Dense(1, activation="sigmoid", name="output", dtype="float32")
    return layers.Dense(len(config.CLASS_NAMES[task]), activation="softmax", name="output",
                        dtype="float32")


def build_cnn(task: str, img_size: int) -> keras.Model:
    blocks = []
    for i, f in enumerate([32, 64, 128, 256], start=1):
        blocks += [
            layers.Conv2D(f, 3, padding="same", activation="relu",
                          name="last_conv" if i == 4 else f"conv{i}a"),
            layers.BatchNormalization(name=f"bn{i}"),
            layers.MaxPooling2D(name=f"pool{i}"),
        ]
    return keras.Sequential(
        [keras.Input((img_size, img_size, 3)), preprocess_layer("rescale")] + blocks + [
            layers.GlobalAveragePooling2D(name="gap"),
            layers.Dense(128, activation="relu", name="fc"),
            layers.Dropout(0.5, name="dropout"),
            output_layer(task)],
        name="cnn")


def build_backbone_model(name: str, task: str, img_size: int, weights="imagenet") -> keras.Model:
    ctor, mode, _ = BACKBONES[name]
    backbone = ctor(include_top=False, weights=weights, input_shape=(img_size, img_size, 3))
    pre = preprocess_layer(mode)
    if name == "vgg16_v1":
        # The original notebook: frozen VGG16 -> Flatten -> Dense(256) -> Dropout(0.5) -> out
        head = [layers.Flatten(name="flatten"), layers.Dense(256, activation="relu", name="fc"),
                layers.Dropout(0.5, name="dropout"), output_layer(task)]
    else:
        head = [layers.GlobalAveragePooling2D(name="gap"),
                layers.Dense(256, activation="relu", name="fc"),
                layers.Dropout(0.5, name="dropout"), output_layer(task)]
    stack = [keras.Input((img_size, img_size, 3))] + ([pre] if pre else []) + [backbone] + head
    return keras.Sequential(stack, name=name)


def build_model(name: str, task: str, img_size=config.IMG_SIZE, weights="imagenet"):
    if name == "cnn":
        return build_cnn(task, img_size)
    if name in BACKBONES:
        return build_backbone_model(name, task, img_size, weights)
    raise ValueError(f"unknown model {name}; choose from {config.MODEL_NAMES}")


def get_backbone(model: keras.Model):
    """The nested ImageNet backbone (a keras.Model inside the Sequential), or None."""
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    return None


def set_phase(model: keras.Model, name: str, phase: str) -> None:
    """phase 'head': whole backbone frozen. phase 'finetune': unfreeze layers whose name
    starts with the backbone's prefixes, EXCEPT BatchNorm layers, which stay frozen
    (in Keras 3 a frozen BN runs in inference mode, so its running stats are preserved)."""
    backbone = get_backbone(model)
    if backbone is None:
        return
    if phase == "head":
        backbone.trainable = False
        return
    prefixes = BACKBONES[name][2]
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = (layer.name.startswith(prefixes)
                           and not isinstance(layer, layers.BatchNormalization))


def compile_model(model: keras.Model, task: str, lr: float) -> None:
    n = len(config.CLASS_NAMES[task])
    if task == "binary":
        loss, acc = "binary_crossentropy", keras.metrics.BinaryAccuracy(name="accuracy")
        auc = keras.metrics.AUC(name="auc")
    else:
        loss, acc = "categorical_crossentropy", keras.metrics.CategoricalAccuracy(name="accuracy")
        auc = keras.metrics.AUC(name="auc", multi_label=True, num_labels=n)  # macro one-vs-rest
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss, metrics=[acc, auc])


def gradcam_layer_name(model: keras.Model) -> str:
    """Grad-CAM target: the backbone's final feature map, or the CNN's last conv layer."""
    bb = get_backbone(model)
    return bb.name if bb is not None else "last_conv"
