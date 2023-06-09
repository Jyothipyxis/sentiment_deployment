# SENTENCE TRANSFORMER DEPLOYMENT ON KSERVE

Currently this code supports only EMBEDDING type of models.
The sentence transformer models are based on PyTorch so we used this [doc](https://kserve.github.io/website/0.8/modelserving/v1beta1/torchserve/) to develop the models.
So this expects the models to be stored as MAR files. So this repository helps us in creating one for our models.

Consider this as example for which you want to create a MAR file - [model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 

### STEPS -
Change the model_name detail here based on sentence transformer model you choose - [setup_config](https://github.com/Jyothipyxis/sentiment_deployment/blob/main/setup_config.json#L2)

Run the following commands 
```
1. python Download_Transformer_models.py
2. torch-model-archiver --model-name sentenceTransformerMiniLM --version 1.0 --serialized-file Transformer_model_mini/pytorch_model.bin --handler ./Transformer_handler_generalized.py --extra-files "Transformer_model_mini/config.json,./setup_config.json"
```

Once done, you will have MAR file of the model you need.
[url](https://kserve.github.io/website/0.8/modelserving/v1beta1/torchserve/)

Later create a folder on s3 with the structure mentioned [here](https://kserve.github.io/website/0.8/modelserving/v1beta1/torchserve/#creating-model-storage-with-model-archive-and-config-file)

Once its deployed.
Create a model on Kubeflow dashboard using below yaml for first deployment. 

```
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "sentence-transformers"
  namespace: kubeflow-tai-example-com
  labels:
    istio: kservegateway
  annotations:
    sidecar.istio.io/inject: "false"
spec:
  predictor:
    serviceAccountName: kubeflow-user
    resources:
        limits:
          cpu: "1"
          memory: 1500Mi
        requests:
          cpu: "1"
          memory: 1000Mi
    pytorch:
      storageUri: "s3://s3-kubeflow-develop/sentenceTransformer"
```
