# Data Generation for BPO
We utilize [VCD](https://github.com/DAMO-NLP-SG/VCD) as noise maker and [LLaVA-Phi](https://github.com/zhuyiche/llava-phi) as the base model to generate negative answers with pretrained bias.


## Installation
```bash
git clone https://github.com/zhuyiche/llava-phi.git
```

## Merge VCD into LLaVA-Phi
1. Copy the folder `vcd_utils` into `/llava-phi/llava_phi/eval/` directory.
2. Follow the [instructions](https://github.com/DAMO-NLP-SG/VCD?tab=readme-ov-file#how-to-use-vcd-in-lvlms), Add the following at the beginning of the start-up script:
```python
from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()
```
The `evolve_vcd_sampling` function replaces the sampling function in the transformers library. The modified sampling function includes an option for visual contrastive decoding, while keeping the rest unchanged.

3. Slightly modify `/llava-phi/llava_phi/model/language_model/llava_phi.py`:

   a. Add contrastive decoding parameters in the `LlavaPhiForCausalLM` class's `forward` function to avoid exceptions in `model.generate`.
   ```python
    (
        images_cd=None,
        cd_alpha=None,
        cd_beta=None
    ）
   ```
   
   b. Add the `prepare_inputs_for_generation_cd` function.
    ```python
        def prepare_inputs_for_generation_cd(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images_cd", None),
            }
        )
        return model_inputs
    ```

4. Move `run_llava_phi_vcd.py` to `/llava-phi/llava_phi/eval/`

## Install environments
Following the [steps](https://github.com/zhuyiche/llava-phi?tab=readme-ov-file#install) to install the environments.


## Generate
Run 
```bash
cd /llava-phi/llava_phi/eval
python run_llava_phi_vcd.py --image_file YOUR_IMAGE_PATH --query YOUR_JSON_PATH --save_path OUTPUT_PATH
```

`YOUR_JSON_PATH` should be a list of json file and has the following format
```
[
    {'prompt': 'What do you see happening in this image?\n<image>',
        'image': 'coco/train2017/000000000009.jpg',
        'completions': [{'score': 1,
        'response': "xxxx",
        'type': 'gt'}]},

    {'prompt': 'question\n<image>',
        'image': 'coco/train2017/0000000000010.jpg',
        'completions': [{'score': 1,
        'response': "xxxx",
        'type': 'gt'}]}, 
    .......
]
```