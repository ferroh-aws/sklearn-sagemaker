from __future__ import absolute_import
import numpy as np
import textwrap

from sagemaker_inference import content_types, decoder, default_inference_handler, encoder
from sagemaker_inference.default_handler_service import DefaultHandlerService
from sagemaker_inference.transformer import Transformer


class HandleService(DefaultHandlerService):
    class SKLearnUserModuleInferenceHandler(default_inference_handler.DefaultInferenceHandler):
        @staticmethod
        def default_model_fn(self, model_dir, context=None):
            raise NotImplementedError(textwrap.dedent("""
            Please provide a model_fn implementation.
            """))

        @staticmethod
        def default_input_fn(self, input_data, content_type, context=None):
            np_array = decoder.decode(input_data, content_type)
            if len(np_array.shape) == 1:
                np_array = np_array.reshape(1, -1)
            return np_array.astype(np.float32) if content_type in content_types.UTF8_TYPES else np_array

        @staticmethod
        def default_output_fn(self, prediction, accept, context=None):
            return encoder.encode(prediction, accept), accept

        @staticmethod
        def default_predict_fn(self, data, model, context=None):
            output = model.predict(data)
            return output

    def __int__(self):
        transformer = Transformer(default_inference_handler=self.SKLearnUserModuleInferenceHandler)
        super(HandleService, self).__init__(transformer=transformer)
