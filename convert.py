from .utils import modify_tf_block
from .poolformer.model import MetaFormer
from .poolformer.layers import *
from .poolformer.blocks import *
import numpy as np 
import os, sys, shutil
import tqdm 
import glob 
import pandas as pd 
import tensorflow as tf 
import tensorflow.keras as keras 
import argparse
import timm, transformers 
from typing import Dict, List
import yaml 
from imutils import paths
import torch


def port_weights(model_type="poolformer_s12", 
                model_savepath =".", 
                include_top=True,
                model_main_res_fname="main_result.csv"
              ):

    print("Intializing the Tensorflow Model")
    
    # read the data from yaml file
    config_file_path = f"configs/{model_type}.yaml"
    with open(config_file_path, "r") as f:
        data = yaml.safe_load(f)
    
    layer_scale_init_values = data.get("layer_scale_init_values")
    res_scale_init_values = data.get("res_scale_init_values")

    layer_scale_init_values = (float(layer_scale_init_values) 
                            if isinstance(layer_scale_init_values, str) else layer_scale_init_values)

    res_scale_init_values = (float(res_scale_init_values) 
                            if isinstance(res_scale_init_values, str) else res_scale_init_values)

    tf_model = MetaFormer(
        depths = data.get('depths'),
        dims = data.get("dims"),
        norm_layers = data.get("block_norm"),
        output_norm = data.get('main_norm'),
        downsample_norm = data.get("downsample_norm"),
        layer_scale_init_values = layer_scale_init_values,
        res_scale_init_values = res_scale_init_values,
        include_top = include_top,
        mlp_bias = data.get("mlp_bias"),
        mlp_act = data.get("mlp_act"),
    )

    dummy_input = np.zeros((1, 224, 224, 3))
    _ = tf_model(dummy_input)

    # path not exists 
    if not os.path.exists(model_main_res_fname):
      make_model_res_file(model_main_res_fname)

    # calculating the flops and nb_params
    #nb_flops = tf_model.flops()
    #nb_flops = int(convert_kb_to_gb(nb_flops))

    nb_params = tf_model.count_params()
    nb_params = int(nb_params / 1000000)

    add_model_res(model_main_res_fname, model_type, nb_params)

    print('Loading the Pytorch model!!!')
    #pt_model = SwinForImageClassification.from_pretrained(f"microsoft/{model_type.replace('_', '-')}")
    #pt_model.eval()

    # pt_model_dict
    pt_model = timm.create_model(
            model_name = model_type,
            pretrained = True,
            num_classes = 1000
        )

    pt_model.eval()
    pt_model_dict = pt_model.state_dict()
    pt_model_dict = {k: np.array(pt_model_dict[k]) for k in pt_model_dict.keys()}

    # main norm
    tf_model.layers[-3] = modify_tf_block(
          tf_component = tf_model.layers[-3],
          pt_weight = pt_model_dict["head.norm.weight"],
          pt_bias = pt_model_dict["head.norm.bias"]
      )

    # patch embed layer's projection
   
    tf_model.layers[0].conv = modify_tf_block(
          tf_component = tf_model.layers[0].conv,
          pt_weight = pt_model_dict["stem.conv.weight"],
          pt_bias = pt_model_dict["stem.conv.bias"]
      )

    if include_top:
      # classification layer
      tf_model.layers[-1] = modify_tf_block(
          tf_component = tf_model.layers[-1],
          pt_weight = pt_model_dict["head.fc.weight"],
          pt_bias = pt_model_dict["head.fc.bias"]
      )

    # for poolformer layers
    for idx, stage in enumerate(tf_model.layers[1: 1+len(data.get("depths"))]):
        modify_metaformer_stage(stage, idx, pt_model_dict)
    
    save_path = os.path.join(model_savepath, model_type)
    save_path = f"{save_path}_fe" if not include_top else save_path
    tf_model.save(save_path)
    
    print(f"TensorFlow model serialized at: {save_path}...")


def modify_metaformer_stage(stage, stage_indx, pt_model_dict):

  block_indx = 0
  for block in stage.layers:
    pt_block_name = f"stages.{stage_indx}.blocks.{block_indx}"

    if isinstance(block, MetaFormerBlock):
      # normalization
      block.norm1 = modify_tf_block(
          tf_component = block.norm1,
          pt_weight = pt_model_dict[f"{pt_block_name}.norm1.weight"],
          pt_bias = pt_model_dict[f"{pt_block_name}.norm1.bias"]
      )
      #block.norm1.beta = None

      #print(block.norm1.beta)

      block.norm2 = modify_tf_block(
          tf_component = block.norm2,
          pt_weight = pt_model_dict[f"{pt_block_name}.norm2.weight"],
          pt_bias = pt_model_dict[f"{pt_block_name}.norm2.bias"]
      )
      #block.norm2.beta = None

      # mlp layer
      block.mlp.fc1 = modify_tf_block(
          tf_component = block.mlp.fc1,
          pt_weight = pt_model_dict[f"{pt_block_name}.mlp.fc1.weight"],
          pt_bias = pt_model_dict[f"{pt_block_name}.mlp.fc1.bias"]
      )
      #block.mlp.fc1.bias = None

      block.mlp.fc2 = modify_tf_block(
          tf_component = block.mlp.fc2,
          pt_weight = pt_model_dict[f"{pt_block_name}.mlp.fc2.weight"],
          pt_bias = pt_model_dict[f"{pt_block_name}.mlp.fc2.bias"]
      )
      #block.mlp.fc2.bias = None

      # mlp act scale and bias
      #block.mlp.act.scale.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.mlp.act.scale"]))
      #block.mlp.act.bias.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.mlp.act.bias"]))

      # res_scale and layer_scale
      if (block.res_scale_1) is not None:
        block.res_scale_1.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.res_scale1.scale"]))

      if (block.res_scale_2) is not None:
        block.res_scale_2.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.res_scale2.scale"]))

      if (block.layer_scale_1) is not None:
        block.layer_scale_1.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.layer_scale1.scale"]))

      if (block.layer_scale_2) is not None:
        block.layer_scale_2.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.layer_scale2.scale"]))

      block_indx += 1

    if isinstance(block, Downsampling):
      block.conv = modify_tf_block(
          tf_component = block.conv,
          pt_weight = pt_model_dict[f"stages.{stage_indx}.downsample.conv.weight"],
          pt_bias = pt_model_dict[f"stages.{stage_indx}.downsample.conv.bias"]
      )

     # block.norm = modify_tf_block(
      #    tf_component = block.norm,
       #   pt_weight = pt_model_dict[f"stages.{stage_indx}.downsample.norm.weight"],
          #pt_bias = pt_model_dict[f"stages.{stage_indx}.downsample.norm.bias"]
      #)
     # block.norm.beta = None


def make_model_res_file(fpath):
  with open(fpath, "w") as file:
    file.write("model_variant, #params\n")


def add_model_res(fpath, model_variant, params):
  with open(fpath, "a") as file:
    file.write(f"{model_variant}, {params}M\n")


def convert_kb_to_gb(val):
  gb_val = val / 1000 / 1000 / 1000
  return gb_val     