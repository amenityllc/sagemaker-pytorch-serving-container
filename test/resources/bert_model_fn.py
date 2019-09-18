# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

BERT_WWM = "bert-large-uncased-whole-word-masking"


def model_fn(model_dir):
    """Loads a model. For PyTorch, a default function to load a model cannot be provided.
    Users should provide customized model_fn() in script.

    Args:
        model_dir: a directory where model is saved.

    Returns: A PyTorch model.
    """
    if model_dir and model_dir != BERT_WWM:
        raise NotImplementedError(f'No support for "{model_dir}". Currently supports only "{BERT_WWM}"')
    from pytorch_transformers import BertForMaskedLM
    return BertForMaskedLM.from_pretrained(BERT_WWM)
