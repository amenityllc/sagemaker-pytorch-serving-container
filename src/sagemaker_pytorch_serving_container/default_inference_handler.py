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

import json
import textwrap

import torch
from pytorch_transformers import BertTokenizer

from sagemaker_inference import content_types, decoder, default_inference_handler, encoder, errors
from sagemaker_inference.errors import UnsupportedFormatError

BERT_WWM = "bert-large-uncased-whole-word-masking"


class DefaultPytorchInferenceHandler(default_inference_handler.DefaultInferenceHandler):
    VALID_CONTENT_TYPES = (content_types.JSON)
    tokenizer = BertTokenizer.from_pretrained(BERT_WWM, do_lower_case=True)

    def default_model_fn(self, model_dir):
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

    def default_input_fn(self, input_data, content_type):
        """A default input_fn that can handle JSON, CSV and NPZ formats.
        Args:
            input_data: the request payload serialized in the content_type format
            content_type: the request content_type
        Returns: input_data deserialized into torch.FloatTensor or torch.cuda.FloatTensor depending if cuda is available.
        """
        self.validate_content_type(content_type)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        task = json.loads(input_data)
        sentence = task["sentence"]
        masked_word = task["mask"]
        seq_len = 64

        tokens = self.tokenizer.tokenize(sentence)
        mask_index = tokens.index(masked_word)  # TODO support range / phrases
        len_tokens = len(tokens)
        mask_position = len(tokens) + mask_index + 2  # [CLS] tokens.. [SEP] tok..[mask]..ens

        tokens_for_prediction = ['[CLS]'] + tokens + ['[SEP]'] + tokens + ['[SEP]']
        tokens_for_prediction[mask_position] = '[MASK]'
        padding = [0] * (seq_len - len(tokens_for_prediction))

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens_for_prediction)
        input_ids = input_ids + padding

        cls_tokens_sep = [0] * (2 + len(tokens))
        tokens_sep = [1] * (1 + len(tokens))
        input_type_ids = cls_tokens_sep + tokens_sep + padding

        input_mask = [1] * len(tokens_for_prediction) + padding


        tokens_tensor = torch.tensor([input_ids])
        token_type_ids = torch.tensor([input_type_ids])
        attention_mask = torch.tensor([input_mask])

        mast_position_tensor = torch.tensor([mask_position]).expand_as(token_type_ids)  # TODO find a better way #87912
        tensor = torch.cat([tokens_tensor, token_type_ids, attention_mask, mast_position_tensor])
        # tensor = torch.cat([tokens_tensor, token_type_ids, attention_mask, mask_position])
        return tensor.to(device)

    def validate_content_type(self, content_type):
        if content_type not in self.VALID_CONTENT_TYPES:
            raise UnsupportedFormatError(f'{content_type} is not in the valid formats: [{self.VALID_CONTENT_TYPES}]')

    def default_predict_fn(self, data, model):
        """A default predict_fn for PyTorch. Calls a model on data deserialized in input_fn.
        Runs prediction on GPU if cuda is available.
        Args:
            data: input data (torch.Tensor) for prediction deserialized by input_fn
            model: PyTorch model loaded in memory by model_fn
        Returns: a prediction
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        tokens_tensor = data[0].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tokens_tensor)

        prediction_scores = outputs[0]
        mask_position_stegan = data[3][0]  # TODO there must be a better way #87912
        return prediction_scores[0, mask_position_stegan]

    def default_output_fn(self, prediction, accept):
        """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPY format.
        Args:
            prediction: a prediction result from predict_fn
            accept: type which the output data needs to be serialized
        Returns: output data serialized
        """
        self.validate_content_type(accept)
        prediction = prediction.topk(20)
        # TODO filter out predictions that are far in meaning to masked_word, even if they make a sound replacement
        prediction = self.tokenizer.convert_ids_to_tokens(prediction[1].detach().cpu().numpy())
        encoded_prediction = encoder.encode(prediction, accept)
        if accept == content_types.CSV:
            encoded_prediction = encoded_prediction.encode("utf-8")

        return encoded_prediction
