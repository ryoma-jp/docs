
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
        else:
            print(f"unsupported operator: {layer.__class__.__name__}")

    df_model_info = pd.DataFrame({
        "Operators": layer_ops,
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

    # --- Get model information ---
    df_model_info = get_model_info(tf.keras.applications.Xception())

    # --- Save model information ---
    print(df_model_info)

    csv_name = Path(args.output_dir, "model_info.csv")
    df_model_info.to_csv(csv_name, index=False)

# --- main routine ---
if __name__=="__main__":
    main()
