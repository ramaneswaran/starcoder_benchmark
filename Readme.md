## LLM Benchmarking Tool

This tool benchmarks LLM inferencing tools on two different benchmarks

1) [CoNala](https://huggingface.co/datasets/neulab/conala/): This is a dataset of short code snippets
2) [Code Contests](https://huggingface.co/datasets/deepmind/code_contests): This is a dataset of long form code contest code.

These two represent two extremities of code generation.

### Scripts For Benchmarking

We have provided the following scripts to run the benchmark

| Inference Platform                                                                                           | Scripts                              |
|--------------------------------------------------------------------------------------------------------------|--------------------------------------|
| [TensorRT-LLM with Triton](https://github.com/triton-inference-server/tensorrtllm_backend)                  | parallel_requests.py                 |
| [vLLM with Triton](https://github.com/triton-inference-server/vllm_backend)                                  | parallel_requests_vllm.py            |
| [vLLM with FastAPI](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server) | parallel_requests_vllm_entrypoint.py |

### Pre-requisite

It is required to setup the LLM servers before you can run the benchmarking scripts. For more information on setting up the servers, use the links provided in the table above.

### Running The Benchmarking

Use the following command to run the benchmark

```
python3 parallel_requests_vllm.py --dataset_name <DATASET_NAME>
```

The DATASET_NAME can be either neulab/conala or deepmind/code_contests

Note: You would have to manually set the max tokens according in the benchmarking script typically in the `send_request` method for each dataset. Reccomended to set 10 for conala and 810 for code_contests.