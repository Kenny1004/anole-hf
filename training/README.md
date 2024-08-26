## Train Anole on your custom data

You can train Anole on your custom data. Note that the current training code has not been fully verified, but we will continuously update it soon!

## Steps



1. Prepare your raw finetuning data like [this](https://github.com/GAIR-NLP/PrivateChameleon/blob/main/facilitating_image_generation/dataset_raw.jsonl)

Note: Current code only supports finetuning on one-text segment and one image, we will support multiple interleaved text segments and images finetuning soon.
```
# Example samples
{"text": "Give me an image of Orange juice in a mason glass with an orange cut in half and wooden orange squeezer.", "image": "/path/to/image/1.png"}
{"text": "Give me an image of Chibi_Yukata_Disney_Princesses_by_vulpixfairy-picture", "image": "/path/to/image/2.png"}
```

2. Set the constants in `constants_training.py`

3. Convert raw finetuning data to tokenized data
```
bash prepare_data.sh
```

4. train the model using huggingface trainer
```
bash train.sh
```


