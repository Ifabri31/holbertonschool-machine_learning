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
        def pt_iterator():
            batch = []
            for pt, _ in data.as_numpy_iterator():
                batch.append(pt.decode('utf-8'))
                if len(batch) >= 1000:
                    yield batch
                    batch = []
            if batch:
                yield batch

        def en_iterator():
            batch = []
            for _, en in data.as_numpy_iterator():
                batch.append(en.decode('utf-8'))
                if len(batch) >= 1000:
                    yield batch
                    batch = []
            if batch:
                yield batch

        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased',
            clean_up_tokenization_spaces=True)
            
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased',
            clean_up_tokenization_spaces=True)

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_iterator(), 2**13)
            
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_iterator(), 2**13)

        return tokenizer_pt, tokenizer_en
