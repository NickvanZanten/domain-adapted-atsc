""" Loading the models for sequence classification on a given dataset."""

from __future__ import absolute_import, division, print_function

import argparse
import os
import logging
from src.finetuning_and_classification.run_glue import MODEL_CLASSES, ALL_MODELS, processors
import pandas as pd

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    print("Start main")
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_file", default=None, type=str, required=True,
                        help="The input data file. Should contain a .csv (or other data types) for the predictions.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    args = parser.parse_args()

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load model
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    # Set data for predictions
    logger.info("Loading examples from dataset file at %s", args.data_file)

    # Predict for the given data file
    pred = model("From the build quality to the performance, everything about it has been sub-par from what I would have expected from Apple.")
    print(pred)