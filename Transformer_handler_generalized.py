import ast
import json
import logging
import os
from abc import ABC

import torch
import torch.nn.functional as F
import transformers
from captum.attr import LayerIntegratedGradients
from transformers import (
    AutoTokenizer,
    GPT2TokenizerFast,
    AutoModel
)

from ts.torch_handler.base_handler import BaseHandler
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class TransformersSeqClassifierHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the BERT model is loaded and
        the Layer Integrated Gradients Algorithm for Captum Explanations
        is initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        # read configs for the mode, model_name, etc. from setup_config.json
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning("Missing the setup_config.json file.")

        # Loading the shared object of compiled Faster Transformer Library if Faster Transformer is set
        if self.setup_config["FasterTransformer"]:
            faster_transformer_complied_path = os.path.join(
                model_dir, "libpyt_fastertransformer.so"
            )
            torch.classes.load_library(faster_transformer_complied_path)
        # Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        # further setup config can be added.
        if self.setup_config["save_mode"] == "torchscript":
            self.model = torch.jit.load(model_pt_path, map_location=self.device)
        elif self.setup_config["save_mode"] == "pretrained":
            if self.setup_config["mode"] == "embeddings":
                logger.info("Yay finally its initialized")
                self.model = AutoModel.from_pretrained(model_dir)
            else:
                logger.warning("Missing the operation mode.")
            # Using the Better Transformer integration to speedup the inference
            if self.setup_config["BetterTransformer"]:
                from optimum.bettertransformer import BetterTransformer

                try:
                    self.model = BetterTransformer.transform(self.model)
                except RuntimeError as error:
                    logger.warning(
                        "HuggingFace Optimum is not supporting this model,for the list of supported models, please refer to this doc,https://huggingface.co/docs/optimum/bettertransformer/overview"
                    )
            # HF GPT2 models options can be gpt2, gpt2-medium, gpt2-large, gpt2-xl
            # this basically place different model blocks on different devices,
            # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/gpt2/modeling_gpt2.py#L962
            if (
                self.setup_config["model_parallel"]
                and "gpt2" in self.setup_config["model_name"]
            ):
                self.model.parallelize()
            else:
                self.model.to(self.device)

        else:
            logger.warning("Missing the checkpoint or state_dict.")

        if "gpt2" in self.setup_config["model_name"]:
            self.tokenizer = GPT2TokenizerFast.from_pretrained(
                "gpt2", pad_token="<|endoftext|>"
            )

        elif any(
            fname
            for fname in os.listdir(model_dir)
            if fname.startswith("vocab.") and os.path.isfile(fname)
        ):
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir, do_lower_case=self.setup_config["do_lower_case"]
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.setup_config["model_name"],
                do_lower_case=self.setup_config["do_lower_case"],
            )

        self.model.eval()
        logger.info("Transformer model from path %s loaded successfully", model_dir)

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        # Question answering does not need the index_to_name.json file.
        if not (
            self.setup_config["mode"] == "embeddings"
        ):
            if os.path.isfile(mapping_file_path):
                with open(mapping_file_path) as f:
                    self.mapping = json.load(f)
            else:
                logger.warning("Missing the index_to_name.json file.")
        self.initialized = True

    def preprocess(self, requests):
        """Basic text preprocessing, based on the user's chocie of application mode.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of Tensor for the size of the word tokens.
        """
        for idx, data in enumerate(requests):
            input_text = data
            # input_text = ast.literal_eval(input_text)
            logger.info("Received text: '%s'", input_text)
            # preprocessing text for sequence_classification, token_classification or text_generation
            if self.setup_config["mode"] == "embeddings":
                inputs = self.tokenizer(
                    input_text,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                logger.info("The model is preprocessed")

        return inputs

    def inference(self, input_batch):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_batch (list): List of Text Tensors from the pre-process function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """
        inputs = input_batch
        inferences = []
        # Handling inference for embeddings.
        if self.setup_config["mode"] == "embeddings":
            with torch.no_grad():
                model_output = self.model(**inputs)
            sentence_embeddings = mean_pooling(model_output, inputs["attention_mask"])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            inferences.append(sentence_embeddings.tolist())

        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output
