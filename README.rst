SageMaker Scikit-learn Container
================================

This is an example of a scikit-learn container using sagemaker inference, container and training kits.

This container is based on AWS https://github.com/aws/sagemaker-scikit-learn-container and is only and example on how
to build custom containers with custom libraries.

**Entry points**

The files in ``sklearn-container`` directory contain the entry points for each type of process:

- ``handle_service.py`` - Default behaviours for serving models.
- ``serving.py`` Entry point for serving models.
- ``training.py`` Entry point for training jobs.