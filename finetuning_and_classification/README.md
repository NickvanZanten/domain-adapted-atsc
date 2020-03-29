## LM Finetuning

### Usage

There are two ways to fine-tune a language model using these scripts. The first quick approach is to use simple_lm_finetuning.py. This script does everything in a single script, but generates training instances that consist of just two sentences. This is quite different from the BERT paper, where (confusingly) the NextSentence task concatenated sentences together from each document to form two long multi-sentences, which the paper just referred to as sentences. The difference between this simple approach and the original paper approach can have a significant effect for long sequences since two sentences will be much shorter than the max sequence length. In this case, most of each training example will just consist of blank padding characters, which wastes a lot of computation and results in a model that isn't really training on long sequences.

As such, the preferred approach (assuming you have documents containing multiple contiguous sentences from your target domain is to use pregenerate_training_data.py to pre-process your data into training examples following the methodology used for LM training in the original BERT paper and repository. Since there is a significant random component to training data generation for BERT, this script includes an option to generate multiple epochs of pre-processed data, to avoid training on the same random splits each epoch. Generating an epoch of data for each training epoch should result a better final model, and so we recommend doing so.

You can then train on the pregenerated data using finetune_on_pregenerated.py, and pointing it to the folder created by pregenerate_training_data.py. Note that you should use the same bert_model and case options for both! Also note that max_seq_len does not need to be specified for the finetune_on_pregenerated.py script, as it is inferred from the training examples.

The LM finetuning code is an adaption to a script from the huggingface/pytorch-transformers repository:
* https://github.com/huggingface/pytorch-transformers/blob/v1.0.0/examples/lm_finetuning/finetune_on_pregenerated.py

Prepare the finetuning corpus, here shown for a test corpus "dev_corpus.txt":

    python pregenerate_training_data.py \
    --train_corpus dev_corpus.txt \
    --bert_model bert-base-uncased --do_lower_case \
    --output_dir dev_corpus_prepared/ \
    --epochs_to_generate 2 --max_seq_len 256


Run actual finetuning with:

    python finetune_on_pregenerated.py \
    --pregenerated_data dev_corpus_prepared/ \
    --bert_model bert-base-uncased --do_lower_case \
    --output_dir dev_corpus_finetuned/ \
    --epochs 2 --train_batch_size 16
    
## Downstream Classification

Down-stream task-specific finetuning code is an adaption to this script:
* https://github.com/huggingface/pytorch-transformers/blob/v1.0.0/examples/run_glue.py

There are various options that can be tweaked, but they are mostly set to the values from the BERT paper/repository and default values should make sense. The most relevant ones are:

--max_seq_len: Controls the length of training examples (in wordpiece tokens) seen by the model. Defaults to 128 but can be set as high as 512. Higher values may yield stronger language models at the cost of slower and more memory-intensive training.
--fp16: Enables fast half-precision training on recent GPUs.

In order to train the BERT-ADA Restaurants model on the SemEval 2014 Task 4 Subtask 2
Restaurants dataset, use the command below:
    
    python run_glue.py \ 
    --model_type bert \
    --model_name_or_path ../data/models/restaurants_10mio_ep3 \
    --do_train --evaluate_during_training --do_eval \
    --logging_steps 100 --save_steps 1200 --task_name=semeval2014-atsc \
    --seed 42 --do_lower_case \
    --data_dir=../data/transformed/restaurants_noconfl \
    --output_dir=../data/models/semeval2014-atsc-bert-ada-restaurants-restaurants \
    --max_seq_length=128 --learning_rate 3e-5 --per_gpu_eval_batch_size=32 --per_gpu_train_batch_size=32 \
    --gradient_accumulation_steps=1 --max_steps=800 --overwrite_output_dir --overwrite_cache --warmup_steps=120 --fp16

In addition, if memory usage is an issue, especially when training on a single GPU, reducing --train_batch_size from the default 32 to a lower number (4-16) can be helpful, or leaving --train_batch_size at the default and increasing --gradient_accumulation_steps to 2-8. Changing --gradient_accumulation_steps may be preferable as alterations to the batch size may require corresponding changes in the learning rate to compensate. There is also a --reduce_memory option for both the pregenerate_training_data.py and finetune_on_pregenerated.py scripts that spills data to disc in shelf objects or numpy memmaps rather than retaining it in memory, which significantly reduces memory usage with little performance impact.