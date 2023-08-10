
import os
import argparse
import math
import pandas as pd
import tensorflow as tf

from pathlib import Path

def arg_parser():
    """Argument Parser
    """
    parser = argparse.ArgumentParser(description='This program can get the number of parameters and the FLOPs of DeepLearning models.',
                formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('--output_dir', dest='output_dir', type=str, default="outputs", required=False, \
            help='Output directory path to save results')
    
    return parser.parse_args()

def get_model_info(model):
    """Get Model Information
    """

    print(f"model name: {model.name}")
    layer_input_shape = []
    layer_output_shape = []
    layer_ops = []
    layer_flops = []
    layer_weights = []
    layer_weights_shape = []
    layer_feature = []
    layer_feature_shape = []
    for layer in model.layers:
        if (isinstance(layer, tf.keras.layers.Conv2D)):
            input_shape = layer.input_shape[1:]
            output_shape = layer.output_shape[1:]
            layer_input_shape.append(tuple(input_shape))
            layer_output_shape.append(tuple(output_shape))

            layer_ops.append(layer.__class__.__name__)

            layer_config = layer.get_config()
            kernel_size = layer_config["kernel_size"]

            # output_shape = input_shape * strides
            layer_flops.append((kernel_size[0] * kernel_size[1] * input_shape[2]) \
                * (output_shape[0] * output_shape[1]) \
                * output_shape[2])
            
            weights = [math.prod(_w.shape) for _w in layer.weights]
            layer_weights.append(sum(weights))

            weights_shape = [tuple(_w.shape) for _w in layer.weights]
            layer_weights_shape.append(weights_shape)

            layer_feature.append(math.prod(output_shape))
            layer_feature_shape.append(output_shape)
        elif (isinstance(layer, tf.keras.layers.SeparableConv2D)):
            input_shape = layer.input_shape[1:]
            output_shape = layer.output_shape[1:]
            layer_input_shape.append(tuple(input_shape))
            layer_output_shape.append(tuple(output_shape))

            layer_ops.append(layer.__class__.__name__)

            layer_config = layer.get_config()
            kernel_size = layer_config["kernel_size"]

            layer_flops.append((kernel_size[0] * kernel_size[1] * input_shape[2]) \
                * (output_shape[0] * output_shape[1]) \
                + input_shape[2] \
                * (output_shape[0] * output_shape[1]) \
                * output_shape[2]
            )

            weights = [math.prod(_w.shape) for _w in layer.weights]
            layer_weights.append(sum(weights))

            weights_shape = [tuple(_w.shape) for _w in layer.weights]
            layer_weights_shape.append(weights_shape)

            layer_feature.append(math.prod(output_shape))
            layer_feature_shape.append(output_shape)
        elif (isinstance(layer, tf.keras.layers.Dense)):
            input_shape = layer.input_shape[1:]
            output_shape = layer.output_shape[1:]
            layer_input_shape.append(tuple(input_shape))
            layer_output_shape.append(tuple(output_shape))

            layer_ops.append(layer.__class__.__name__)

            layer_flops.append(input_shape[0] * output_shape[0])

            weights = [math.prod(_w.shape) for _w in layer.weights]
            layer_weights.append(sum(weights))

            weights_shape = [tuple(_w.shape) for _w in layer.weights]
            layer_weights_shape.append(weights_shape)

            layer_feature.append(math.prod(output_shape))
            layer_feature_shape.append(output_shape)
        elif (isinstance(layer, tf.keras.layers.DepthwiseConv2D)):
            input_shape = layer.input_shape[1:]
            output_shape = layer.output_shape[1:]
            layer_input_shape.append(tuple(input_shape))
            layer_output_shape.append(tuple(output_shape))

            layer_ops.append(layer.__class__.__name__)

            layer_config = layer.get_config()
            kernel_size = layer_config["kernel_size"]

            layer_flops.append((kernel_size[0] * kernel_size[1]) \
                * (output_shape[0] * output_shape[1]) \
                * output_shape[2])
            
            weights = [math.prod(_w.shape) for _w in layer.weights]
            layer_weights.append(sum(weights))

            weights_shape = [tuple(_w.shape) for _w in layer.weights]
            layer_weights_shape.append(weights_shape)

            layer_feature.append(math.prod(output_shape))
            layer_feature_shape.append(output_shape)
        elif (isinstance(layer, tf.keras.layers.Multiply)):
            input_shape = [_x[1:] for _x in layer.input_shape]
            output_shape = layer.output_shape[1:]
            layer_input_shape.append(input_shape)
            layer_output_shape.append(output_shape)

            layer_ops.append(layer.__class__.__name__)

            layer_config = layer.get_config()

            layer_flops.append(math.prod(output_shape))
            
            weights = [math.prod(_w.shape) for _w in layer.weights]
            layer_weights.append(sum(weights))

            weights_shape = [tuple(_w.shape) for _w in layer.weights]
            layer_weights_shape.append(weights_shape)

            layer_feature.append(math.prod(output_shape))
            layer_feature_shape.append(output_shape)
        else:
            print(f"unsupported operator: {layer.__class__.__name__}")

    df_model_info = pd.DataFrame({
        "Operators": layer_ops,
        "Input shape": layer_input_shape,
        "Output shape": layer_output_shape,
        "FLOPs": layer_flops,
        "Weights": layer_weights,
        "Shape of Weights": layer_weights_shape,
        "Features": layer_feature,
        "Shape of Features": layer_feature_shape
    })
    return df_model_info

def main():
    """Main
    """

    # --- Parse arguments ---
    args = arg_parser()
    print(f"output_dir: {args.output_dir}")

    # --- Create output directory ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Get Xception model information ---
    df_model_info = get_model_info(tf.keras.applications.Xception())
    csv_name = Path(args.output_dir, "xception.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get VGG16 model information ---
    df_model_info = get_model_info(tf.keras.applications.VGG16())
    csv_name = Path(args.output_dir, "vgg16.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get VGG19 model information ---
    df_model_info = get_model_info(tf.keras.applications.VGG19())
    csv_name = Path(args.output_dir, "vgg19.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get ResNet50 model information ---
    df_model_info = get_model_info(tf.keras.applications.ResNet50())
    csv_name = Path(args.output_dir, "resnet50.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get ResNet101 model information ---
    df_model_info = get_model_info(tf.keras.applications.ResNet101())
    csv_name = Path(args.output_dir, "resnet101.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get ResNet152 model information ---
    df_model_info = get_model_info(tf.keras.applications.ResNet152())
    csv_name = Path(args.output_dir, "resnet152.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get InceptionV3 model information ---
    df_model_info = get_model_info(tf.keras.applications.InceptionV3())
    csv_name = Path(args.output_dir, "inception_v3.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get InceptionResNetV2 model information ---
    df_model_info = get_model_info(tf.keras.applications.InceptionResNetV2())
    csv_name = Path(args.output_dir, "inception_resnet_v2.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get MobileNet model information ---
    df_model_info = get_model_info(tf.keras.applications.MobileNet())
    csv_name = Path(args.output_dir, "mobilenet.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get MobileNetV2 model information ---
    df_model_info = get_model_info(tf.keras.applications.MobileNetV2())
    csv_name = Path(args.output_dir, "mobilenet_v2.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get DenseNet121 model information ---
    df_model_info = get_model_info(tf.keras.applications.DenseNet121())
    csv_name = Path(args.output_dir, "densenet121.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get DenseNet169 model information ---
    df_model_info = get_model_info(tf.keras.applications.DenseNet169())
    csv_name = Path(args.output_dir, "densenet169.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get DenseNet201 model information ---
    df_model_info = get_model_info(tf.keras.applications.DenseNet201())
    csv_name = Path(args.output_dir, "densenet201.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get NASNetMobile model information ---
    df_model_info = get_model_info(tf.keras.applications.NASNetMobile())
    csv_name = Path(args.output_dir, "nasnet_mobile.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get NASNetLarge model information ---
    df_model_info = get_model_info(tf.keras.applications.NASNetLarge())
    csv_name = Path(args.output_dir, "nasnet_large.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get EfficientNetB0 model information ---
    df_model_info = get_model_info(tf.keras.applications.EfficientNetB0())
    csv_name = Path(args.output_dir, "efficientnet_b0.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get EfficientNetB1 model information ---
    df_model_info = get_model_info(tf.keras.applications.EfficientNetB1())
    csv_name = Path(args.output_dir, "efficientnet_b1.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get EfficientNetB2 model information ---
    df_model_info = get_model_info(tf.keras.applications.EfficientNetB2())
    csv_name = Path(args.output_dir, "efficientnet_b2.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get EfficientNetB3 model information ---
    df_model_info = get_model_info(tf.keras.applications.EfficientNetB3())
    csv_name = Path(args.output_dir, "efficientnet_b3.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get EfficientNetB4 model information ---
    df_model_info = get_model_info(tf.keras.applications.EfficientNetB4())
    csv_name = Path(args.output_dir, "efficientnet_b4.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get EfficientNetB5 model information ---
    df_model_info = get_model_info(tf.keras.applications.EfficientNetB5())
    csv_name = Path(args.output_dir, "efficientnet_b5.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get EfficientNetB6 model information ---
    df_model_info = get_model_info(tf.keras.applications.EfficientNetB6())
    csv_name = Path(args.output_dir, "efficientnet_b6.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get EfficientNetB7 model information ---
    df_model_info = get_model_info(tf.keras.applications.EfficientNetB7())
    csv_name = Path(args.output_dir, "efficientnet_b7.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get EfficientNetV2B0 model information ---
    df_model_info = get_model_info(tf.keras.applications.EfficientNetV2B0())
    csv_name = Path(args.output_dir, "efficientnet_v2_b0.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get EfficientNetV2B1 model information ---
    df_model_info = get_model_info(tf.keras.applications.EfficientNetV2B1())
    csv_name = Path(args.output_dir, "efficientnet_v2_b1.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get EfficientNetV2B2 model information ---
    df_model_info = get_model_info(tf.keras.applications.EfficientNetV2B2())
    csv_name = Path(args.output_dir, "efficientnet_v2_b2.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get EfficientNetV2B3 model information ---
    df_model_info = get_model_info(tf.keras.applications.EfficientNetV2B3())
    csv_name = Path(args.output_dir, "efficientnet_v2_b3.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get EfficientNetV2S model information ---
    df_model_info = get_model_info(tf.keras.applications.EfficientNetV2S())
    csv_name = Path(args.output_dir, "efficientnet_v2_s.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get EfficientNetV2M model information ---
    df_model_info = get_model_info(tf.keras.applications.EfficientNetV2M())
    csv_name = Path(args.output_dir, "efficientnet_v2_m.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get EfficientNetV2L model information ---
    df_model_info = get_model_info(tf.keras.applications.EfficientNetV2L())
    csv_name = Path(args.output_dir, "efficientnet_v2_l.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get ConvNeXtTiny model information ---
    df_model_info = get_model_info(tf.keras.applications.ConvNeXtTiny())
    csv_name = Path(args.output_dir, "convnext_tiny.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get ConvNeXtSmall model information ---
    df_model_info = get_model_info(tf.keras.applications.ConvNeXtSmall())
    csv_name = Path(args.output_dir, "convnext_small.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get ConvNeXtBase model information ---
    df_model_info = get_model_info(tf.keras.applications.ConvNeXtBase())
    csv_name = Path(args.output_dir, "convnext_base.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get ConvNeXtLarge model information ---
    df_model_info = get_model_info(tf.keras.applications.ConvNeXtLarge())
    csv_name = Path(args.output_dir, "convnext_large.csv")
    df_model_info.to_csv(csv_name, index=False)

    # --- Get ConvNeXtXLarge model information ---
    df_model_info = get_model_info(tf.keras.applications.ConvNeXtXLarge())
    csv_name = Path(args.output_dir, "convnext_xlarge.csv")
    df_model_info.to_csv(csv_name, index=False)

# --- main routine ---
if __name__=="__main__":
    main()
