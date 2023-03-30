from __future__ import absolute_import
import logging

from sagemaker_training import entry_point, environment, runner

logger = logging.getLogger(__name__)


def train(training_environment):
    logger.info('Invoking user training script')
    entry_point.run(uri=training_environment.module_dir,
                    user_entry_point=training_environment.user_entry_point,
                    args=training_environment.to_env_vars(),
                    runner_type=runner.ProcessRunnerType)


def main():
    train(environment.Environment())
