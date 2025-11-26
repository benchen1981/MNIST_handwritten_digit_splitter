import numpy as np
import cv2
import tensorflow as tf

def compute_gradcam_overlay(model, img, last_conv_layer_name=None):
    """
    對單一 28x28 灰階影像產生 Grad-CAM 熱度圖並疊加回原影像。
    """
    img_norm = img.astype("float32") / 255.0
    img_input = img_norm.reshape(1, 28, 28, 1)

    if last_conv_layer_name is None:
        # 預設取模型中最後一層 Conv2D
        conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        if not conv_layers:
            raise ValueError("模型中沒有 Conv2D 層，無法計算 Grad-CAM。")
        last_conv_layer = conv_layers[-1]
    else:
        last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.layers[-1].output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_input)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(conv_outputs.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    heatmap = cv2.resize(heatmap, (28, 28))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img_3ch = cv2.cvtColor((img_norm * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(img_3ch, 0.5, heatmap_color, 0.5, 0)

    return overlay
