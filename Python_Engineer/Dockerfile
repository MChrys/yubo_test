FROM emacski/tensorflow-serving:2.6.0

COPY batching.cfg /batching.cfg
COPY resnet18_places365 /model


ENTRYPOINT tensorflow_model_server \
    --port=8500 \
    --rest_api_port=8501 \
    --model_name=model \
    --model_base_path=/model \
    --batching_parameters_file=/batching.cfg \
    --enable_batching=true
