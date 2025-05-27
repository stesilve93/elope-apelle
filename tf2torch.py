import torch
import argparse
import os
import re
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import numpy as np
import torch
from tensorflow.contrib import image as contrib_image  # ✅ Needed for ImageProjectiveTransform

def load_tf_checkpoint(meta_path):
    tf.reset_default_graph()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, meta_path[:-5])  # Strip ".meta"
        var_list = tf.global_variables()
        var_values = sess.run(var_list)
        var_dict = {v.name: val for v, val in zip(var_list, var_values)}
    return var_dict

def map_tf_to_torch_name(tf_name):
    """Map typical TensorFlow variable names to PyTorch-style names"""
    name = tf_name

    # Remove :0 suffix
    name = re.sub(r':\d+$', '', name)

    # Common replacements
    name = name.replace('vs/vs/', '')
    name = name.replace('kernel', 'weight')
    name = name.replace('biases', 'bias')
    name = name.replace('moving_mean', 'running_mean')
    name = name.replace('moving_variance', 'running_var')
    name = name.replace('/', '.')

    return name

def convert_to_torch(var_dict, output_path):
    torch_dict = {}
    for k, v in var_dict.items():
        torch_name = map_tf_to_torch_name(k)
        torch_dict[torch_name] = torch.tensor(v)
        print(f"[→] {k} → {torch_name}")
    
    torch.save(torch_dict, output_path)
    print(f"\n✅ PyTorch weights saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta', type=str, required=True, help='Path to .meta file')
    parser.add_argument('--output', type=str, default='converted.pth', help='Output .pth filename')
    args = parser.parse_args()

    if not os.path.exists(args.meta):
        print(f"❌ .meta file not found: {args.meta}")
        return
    
    var_dict = load_tf_checkpoint(args.meta)
    convert_to_torch(var_dict, args.output)

if __name__ == '__main__':
    main()
