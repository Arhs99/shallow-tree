## Parallel execution
I have tried ```tensorflow serving``` running the models in a docker container. More details on installing ```docker```, then ```nvidia-docker``` for GPU support and finally ```tensorflow serving``` can be found in: https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/docker.md

When set-up is complete, you need to:
- modify the configuration yml file, see ``shallowtree/parallel/config_remote.yml`` as an example
- a config file to tell tensorflow serving where to find the expansion policy and the filter policy models, see ``shallowtree/parallel/config_tf_server.txt`` as an example

Then start the docker container by executing:
```
sudo docker run --gpus all -p 8500:8500 -p 8501:8501 \
--mount type=bind,source=<path/to/>/uspto_expand,target=/models/uspto_expand \
--mount type=bind,source=<path/to/>/filter_policy_all,target=/models/filter_policy_all \
--mount type=bind,source=<path/to/>/config_tf_server.txt,target=/models/config_tf_server.txt \
--mount type=bind,source=<path/to/>/config_batch.txt,target=/models/config_batch.txt \
-t tensorflow/serving:latest-gpu --model_config_file=/models/config_tf_server.txt \
--enable_batching --batching_parameters_file=/models/config_batch.txt &
```

Finally, use your favorite multiprocessing approach. One way is to use ``gnu parallel`` as for example:
```
parallel -N 10 -a smiles.txt -j 8 "printf '%s\n' {} | searchcli --config <path/to/>/parallel/config_remote.yml --depth 2 --routes"
```
that takes a ``smiles.txt`` file with SMILES strings as input and runs ``searchcli`` on 8 CPUs and batches of 10 SMILES. 

On my home PC with 1 GPU NVIDIA RTX4070Ti, the [smiles.txt](shallowtree/smiles.txt) file with 80 SMILES took ~7 mins to run which is about 5 sec per molecule.
