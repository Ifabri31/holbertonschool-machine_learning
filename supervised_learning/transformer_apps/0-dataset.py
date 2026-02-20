#!/usr/bin/env python3
"""
0-dataset.py
"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    loads and preps a dataset for machine translation
    """

    def __init__(self):
        """constructor"""
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en', split='train',
            as_supervised=True)
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en', split='validation',
            as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        creates sub-word tokenizers for our dataset
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased',
            clean_up_tokenization_spaces=True)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased', clean_up_tokenization_spaces=True)

        pt_iter = (pt.decode('utf-8') for pt, _ in data.as_numpy_iterator())
        en_iter = (en.decode('utf-8') for _, en in data.as_numpy_iterator())

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_iter, 2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_iter, 2**13)

        return tokenizer_pt, tokenizer_en
