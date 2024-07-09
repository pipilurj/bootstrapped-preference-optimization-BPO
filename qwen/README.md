# BPO for Silkie

Modified code from [[VLFeedback]](https://github.com/vlf-silkie/VLFeedback). Perform BPO on Qwen-VL-Chat using BPO data.

### Training data
Download ShareGPT4V from [here](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V)

Download COCO from [here](https://cocodataset.org/#home)

Download dataset annotation from [here](https://huggingface.co/datasets/renjiepi/BPO)

Extract  data from ShareGPT4V and organize the images as follows:

```
Image_root
├── coco/
        train2017/
├── llava/
          llava_pretrain /
├── sam/
├── share_textvqa/
                images/
├── web-celebrity/
                  images/
├── web-landmark/
                 images/
├── wikiart/
            images/
```

### Installation

To run our training scripts, create a virtual environment and install the dependencies first.

```bash
conda create -n silkie python=3.10  && conda activate silkie
pip install -r requirements.txt
```

### Training

Our training scripts support both single-node and multi-node training.
We provide a `launch_dpo.py` script that handles both cases. If you want to launch a job locally, you can use:

```bash
python launch_dpo.py --config dpo_config/example.yaml --working $WORKING_DIR
```

If you want to launch a job on a Slurm cluster, specify `GPUS_PER_NODE` in `launch_dpo.py` and run:

```bash
python launch_dpo.py --config dpo_config/example.yaml --working $WORKING_DIR --gpus $NUM_GPUS
```

## Citations

```bib
@article{2023vlfeedback,
  author      = {Lei Li and Zhihui Xie and Mukai Li and Shunian Chen and Peiyi Wang and Liang Chen and  Yazheng Yang and  Benyou Wang and  Lingpeng Kong},
  title       = {Silkie: Preference Distillation for Large Visual Language Models},
  publisher   = {arXiv:2312.10665},
  year        = {2023}
}
```

## Acknowledgements

We would like to thank the authors of [trl](https://github.com/huggingface/trl) and [Qwen-VL](https://github.com/QwenLM/Qwen-VL) for their great work.