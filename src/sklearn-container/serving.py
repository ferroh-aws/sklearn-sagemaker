from __future__ import absolute_import
import os
import importlib
import logging
import numpy as np
from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker, server)

logging.basicConfig(format='%(asctime)s %(levelname)s - %(name)s - %(message)s', level=logging.INFO)

logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('s3transfer').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.WARN)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def default_model_fn(model_dir):
    return transformer.default_model_fn(model_dir)


def default_input_fn(input_data, content_type):
    np_array = encoders.decode(input_data, content_type)
    return np_array.astype(np.float32) if content_type in content_types.UTF8_TYPES else np_array


def default_predict_fn(input_data, model):
    output = model.predict(input_data)
    return output


def default_output_fn(prediction, accept):
    return worker.Response(encoders.encode(prediction, accept), accept, mimetype=accept)


def _user_module_transformer(user_module):
    model_fn = getattr(user_module, 'model_fn', default_model_fn)
    input_fn = getattr(user_module, 'input_fn', default_input_fn)
    predict_fn = getattr(user_module, 'predict_fn', default_predict_fn)
    output_fn = getattr(user_module, 'output_fn', default_output_fn)
    return transformer.Transformer(model_fn=model_fn, input_fn=input_fn, predict_fn=predict_fn,
                                   output_fn=output_fn)


def _user_module_execution_parameters_fn(user_module):
    return getattr(user_module, 'execution_parameters_fn', None)


def import_module(module_name, module_dir):
    try:
        user_module = importlib.import_module(module_name)
    except ImportError:
        user_module = modules.import_module(module_dir, module_name)
    except Exception:
        logger.info('Encountered an unexpected error')
        raise

    user_module_transformer = _user_module_transformer(user_module)
    user_module_transformer.initialize()

    return user_module_transformer, _user_module_execution_parameters_fn(user_module)


app = None


def main(environ, start_response):
    global app

    if app is None:
        serving_env = env.ServingEnv()

        user_module_transformer, execution_parameters_fn = import_module(serving_env.module_name,
                                                                         serving_env.module_dir)
        app = worker.Worker(transform_fn=user_module_transformer.transform,
                            module_name=serving_env.module_name,
                            execution_parameters_fn=execution_parameters_fn)
    return app(environ, start_response)


def serving_entrypoint():
    server.start(env.ServingEnv().framework_module)
