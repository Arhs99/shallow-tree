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

On my home PC with 1 GPU NVIDIA RTX4070Ti, the [smiles.txt](/shallowtree/smiles.txt) file with 80 SMILES took ~7 mins to run which is about 5 sec per molecule.

Further performance enhancement is achieved by persistent caching of intermediates with their routes via Redis as described in the next section.

On the same PC and command as above, with Redis caching enabled and starting from a completely empty cache, we can achieve doubling of performance with 3m 11sec for the run or 2.4 sec per molecule


## Redis Cache Setup

For shared caching across parallel workers, install and start Redis:

```bash
# Install (Ubuntu/Debian)
sudo apt update && sudo apt install redis-server -y

# Enable persistence
sudo sed -i 's/appendonly no/appendonly yes/' /etc/redis/redis.conf

# Start and enable on boot
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Verify Redis is running
redis-cli ping  # Should return PONG
```

Install the cache extra:
```bash
poetry install -E cache
```

Update your config.yml:

```yaml
cache:
  enabled: true
  host: localhost
  port: 6379
```

The cache uses a config hash prefix, so different configurations automatically get separate cache entries. All parallel workers share the same Redis cache, avoiding redundant computations across processes.

### Inspecting Cached Data

**Using redis-cli:**
```bash
# List all shallowtree keys
redis-cli KEYS "shallowtree:*"

# Get a specific cache entry (shows depth, score, timestamp)
redis-cli GET "shallowtree:<config_hash>:cache:<inchi_key>"

# Get a solved route entry (shows reactants SMILES, score, classification)
redis-cli GET "shallowtree:<config_hash>:solved:<inchi_key>"

# Count all entries
redis-cli KEYS "shallowtree:*" | wc -l

# Clear all shallowtree entries
redis-cli KEYS "shallowtree:*" | xargs redis-cli DEL
```

**Using Python:**
```python
import redis
import json

r = redis.Redis(decode_responses=True)

# List all keys
for key in r.scan_iter("shallowtree:*"):
    print(key)

# Get and parse a specific entry
data = r.get("shallowtree:<config_hash>:cache:<inchi_key>")
if data:
    parsed = json.loads(data)
    print(f"Depth: {parsed['depth']}, Score: {parsed['score']}")
```

**Key format:**
- `shallowtree:<config_hash>:cache:<inchi_key>` - Branch pruning data
  ```json
  {"depth": 1, "score": 0.95, "ts": 1706900000}
  ```
- `shallowtree:<config_hash>:solved:<inchi_key>` - Route reconstruction data
  ```json
  {"reactants_smiles": ["CCO", "CC(=O)O"], "score": 0.95, "classification": "Amide bond formation", "ts": 1706900000}
  ```

The `config_hash` is a 16-character hash of your config (model paths, stock paths, cutoff values) so different configurations get isolated namespaces.
