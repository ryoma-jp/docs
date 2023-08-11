
import os
import argparse
import math
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

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

    # --- Initialize models information ---
    models_info_columns = ["Model Name", "FLOPs", "Weights", "Features"]
    df_models_info = pd.DataFrame()
    supported_models = [
        ["Xception", tf.keras.applications.Xception, "xception.csv"],
        ["VGG16", tf.keras.applications.VGG16, "vgg16.csv"],
        ["VGG19", tf.keras.applications.VGG19, "vgg19.csv"],
        ["ResNet50", tf.keras.applications.ResNet50, "resnet50.csv"],
        ["ResNet101", tf.keras.applications.ResNet101, "resnet101.csv"],
        ["ResNet152", tf.keras.applications.ResNet152, "resnet152.csv"],
        ["InceptionV3", tf.keras.applications.InceptionV3, "inception_v3.csv"],
        ["InceptionResNetV2", tf.keras.applications.InceptionResNetV2, "inception_resnet_v2.csv"],
        ["MobileNet", tf.keras.applications.MobileNet, "mobilenet.csv"],
        ["MobileNetV2", tf.keras.applications.MobileNetV2, "mobilenet_v2.csv"],
        ["DenseNet121", tf.keras.applications.DenseNet121, "densenet121.csv"],
        ["DenseNet169", tf.keras.applications.DenseNet169, "densenet169.csv"],
        ["DenseNet201", tf.keras.applications.DenseNet201, "densenet201.csv"],
        ["NASNetMobile", tf.keras.applications.NASNetMobile, "nasnet_mobile.csv"],
        ["NASNetLarge", tf.keras.applications.NASNetLarge, "nasnet_large.csv"],
        ["EfficientNetB0", tf.keras.applications.EfficientNetB0, "efficientnet_b0.csv"],
        ["EfficientNetB1", tf.keras.applications.EfficientNetB1, "efficientnet_b1.csv"],
        ["EfficientNetB2", tf.keras.applications.EfficientNetB2, "efficientnet_b2.csv"],
        ["EfficientNetB3", tf.keras.applications.EfficientNetB3, "efficientnet_b3.csv"],
        ["EfficientNetB4", tf.keras.applications.EfficientNetB4, "efficientnet_b4.csv"],
        ["EfficientNetB5", tf.keras.applications.EfficientNetB5, "efficientnet_b5.csv"],
        ["EfficientNetB6", tf.keras.applications.EfficientNetB6, "efficientnet_b6.csv"],
        ["EfficientNetB7", tf.keras.applications.EfficientNetB7, "efficientnet_b7.csv"],
        ["EfficientNetV2B0", tf.keras.applications.EfficientNetV2B0, "efficientnet_v2_b0.csv"],
        ["EfficientNetV2B1", tf.keras.applications.EfficientNetV2B1, "efficientnet_v2_b1.csv"],
        ["EfficientNetV2B2", tf.keras.applications.EfficientNetV2B2, "efficientnet_v2_b2.csv"],
        ["EfficientNetV2B3", tf.keras.applications.EfficientNetV2B3, "efficientnet_v2_b3.csv"],
        ["EfficientNetV2S", tf.keras.applications.EfficientNetV2S, "efficientnet_v2_s.csv"],
        ["EfficientNetV2M", tf.keras.applications.EfficientNetV2M, "efficientnet_v2_m.csv"],
        ["EfficientNetV2L", tf.keras.applications.EfficientNetV2L, "efficientnet_v2_l.csv"],
        ["ConvNeXtTiny", tf.keras.applications.ConvNeXtTiny, "convnext_tiny.csv"],
        ["ConvNeXtSmall", tf.keras.applications.ConvNeXtSmall, "convnext_small.csv"],
        ["ConvNeXtBase", tf.keras.applications.ConvNeXtBase, "convnext_base.csv"],
        ["ConvNeXtLarge", tf.keras.applications.ConvNeXtLarge, "convnext_large.csv"],
        ["ConvNeXtXLarge", tf.keras.applications.ConvNeXtXLarge, "convnext_xlarge.csv"],
    ]

    # --- Get models information ---
    for model in supported_models:
        model_name, model_obj, model_csv = model
        df_model_info = get_model_info(model_obj())
        csv_name = Path(args.output_dir, model_csv)
        df_model_info.to_csv(csv_name, index=False)

        df_add = pd.DataFrame({"Model Name": [model_name]})
        df_add = pd.concat([df_add, df_model_info[models_info_columns[1:]].sum().to_frame().T], axis=1)
        df_models_info = pd.concat([df_models_info, df_add], ignore_index=True)
    print(df_models_info)

    # --- Save models information(graph) ---
    df_models_info.to_csv(Path(args.output_dir, "models_info.csv"), index=False)
    ax = sns.scatterplot(data=df_models_info, x="FLOPs", y="Weights")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("FLOPs vs Weights")

    for i, model_info in df_models_info.iterrows():
        ax.text(model_info["FLOPs"]+.02, model_info["Weights"], model_info["Model Name"])

    plt.savefig(Path(args.output_dir, "models_info.png"))
    plt.close()

    # --- Save models information(LaTeX) ---
    with open(Path(args.output_dir, "models_info.tex"), mode="w", encoding="utf-8") as f:
        lines = [
            "\\begin{table}[H]\n",
            "\t\\caption{FLOPs vs Weights}\n",
            "\t\\label{table:flops_vs_weights}\n",
            "\t\\centering\n",
            "\t\\begin{tabular}{lll}\n",
            "\t\t\\hline\n",
            "\t\tModel Name & FLOPs & Weights \\\\ \n",
            "\t\t\\hline \\hline \n",
        ]

        lines += [f"\t\t{model_info['Model Name']} & {model_info['FLOPs']} & {model_info['Weights']} \\\\ \n" for i, model_info in df_models_info.iterrows()]
        
        lines += [
            "\t\t\\hline\n",
            "\t\\end{tabular}\n",
            "\\end{table}\n",
        ]

        f.writelines(lines)

# --- main routine ---
if __name__=="__main__":
    main()
