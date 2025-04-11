""" Module containing helper routines for using Keras, Tensorflow and Onnx models
"""
from __future__ import annotations

import functools
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime
import psutil

try:
    import tensorflow as tf
    from google.protobuf.json_format import MessageToDict

    # pylint: disable=all
    from tensorflow.keras.metrics import top_k_categorical_accuracy
    from tensorflow.keras.models import load_model as load_keras_model
    from tensorflow_serving.apis import (
        get_model_metadata_pb2,
        predict_pb2,
        prediction_service_pb2_grpc,
    )
except ImportError:
    pass


# pylint: enable=all
from shallowtree.utils.logging import logger

if TYPE_CHECKING:
    from shallowtree.utils.type_utils import Any, Callable, List, Union

    _ModelInput = Union[np.ndarray, List[np.ndarray]]

_logger = logger()

# Suppress tensforflow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def load_model(
    source: str, key: str, use_remote_models: bool
) -> Union[
    "LocalKerasModel", "LocalOnnxModel"
]:
    """
    Load model from a configuration specification.

    If `use_remote_models` is True, tries to load:
      1. A Tensorflow server through gRPC
      2. A Tensorflow server through REST API
      3. A local Keras model
    otherwise it just loads the local model.

    :param source: if fallbacks to a local model, this is the filename
    :param key: when connecting to Tensorflow server this is the model name
    :param use_remote_models: if True will try to connect to remote model server
    :return: a model object with a predict object
    """
    if source.split(".")[-1] == "onnx":
        return LocalOnnxModel(source)
    return LocalKerasModel(source)


class LocalKerasModel:
    """
    A keras policy model that is executed locally.

    The size of the input vector can be determined with the len() method.

    :ivar model: the compiled model
    :ivar output_size: the length of the output vector

    :param filename: the path to a Keras checkpoint file
    """

    def __init__(self, filename: str) -> None:
        top10_acc = functools.partial(top_k_categorical_accuracy, k=10)
        top10_acc.__name__ = "top10_acc"  # type: ignore

        top50_acc = functools.partial(top_k_categorical_accuracy, k=50)
        top50_acc.__name__ = "top50_acc"  # type: ignore

        self.model = load_keras_model(
            filename,
            custom_objects={"top10_acc": top10_acc, "top50_acc": top50_acc, "tf": tf},
        )
        try:
            self._model_dimensions = int(self.model.input.shape[1])
        except AttributeError:
            self._model_dimensions = int(self.model.input[0].shape[1])
        self.output_size = int(self.model.output.shape[1])

    def __len__(self) -> int:
        return self._model_dimensions

    def predict(self, *args: np.ndarray, **_: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass of the neural network.

        :param args: the input vectors
        :return: the vector of the output layer
        """
        return self.model.predict(args, verbose=0)


class LocalOnnxModel:
    """
    An Onnx model that is executed locally.

    The size of the input vector can be determined with the len() method.

    :ivar model: the compiled Onnx model
    :ivar output_size: the length of the output vector

    :param filename: the path to a Onnx model checkpoint file
    """

    def __init__(self, filename: str) -> None:
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = _get_thread_count_per_core()

        self.model = onnxruntime.InferenceSession(
            filename, sess_options=session_options
        )
        self._model_inputs = self.model.get_inputs()
        self._model_output = self.model.get_outputs()[0]
        self._model_dimensions = int(self._model_inputs[0].shape[1])
        self.output_size = int(self._model_output.shape[1])

    def __len__(self) -> int:
        return self._model_dimensions

    def predict(self, *args: np.ndarray, **_: np.ndarray) -> np.ndarray:
        """
        Perform a prediction run on the onnx model.

        :param args: the input vectors
        :return: the vector of the output layer
        """
        return self.model.run(
            [self._model_output.name],
            {
                model_input.name: input.astype(np.float32)
                for model_input, input in zip(self._model_inputs, list(args))
            },
        )[0]


def _get_thread_count_per_core() -> int:
    return psutil.cpu_count() // psutil.cpu_count(logical=False)
