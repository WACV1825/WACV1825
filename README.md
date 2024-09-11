
# Anonymous Repository for WACV 2025 Submission 1825

This repository has been anonymized to comply with the double-blind review process for WACV 2025.

## Train the Teacher Model Using FixMatch
This part of the code is based on the implementation provided by SRST-AWR. Due to the absence of a formal license for their code, we cannot include it in this repository. You can find the commands for training the teacher model by FixMatch in the [original SRST-AWR repository](https://github.com/dyoony/SRST_AWR).

## Train Robust Models Using UAT++ and RST
To train robust models using UAT++ or RST, use the following commands:

```bash
python train_semi_base.py --model-dir {directory_to_save_model} --dataset {dataset} --model wideresnet --depth {depth} --widen-factor {widen_factor} --num-labels-per-class {number_of_labeled_data_per_class} --ges const --lambd {lambd} --loss-inner ce --outer {uat or rst} --gpu_id {gpu_id}
```

## Train Robust Model Using RST+Ours
To train robust models using our method (RST+Ours), use the following command:

```bash
python train_semi_interp.py --model-dir {directory_to_save_model} --dataset {dataset} --model wideresnet --depth {depth} --widen-factor {widen_factor} --rho-setup {0 or 1} --num-labels-per-class {number_of_labeled_data_per_class} --ges {global epsilon scheduling strategy} --lambd {lambd} --loss-inner ce --tau {tau} --intp-steps {K} {--mixed} --mixed-beta {beta} --wandb-project {wandb project name} --wandb-entity {wandb entity name} --gpu_id {gpu_id}
```

## Test Natural Accuracy and Robustness
To test natural accuracy and robustness against adversarial attacks, use the following command:

```bash
python test.py --dataset {dataset} --model wideresnet --depth {depth} --widen-factor {widen_factor} --model-dir {path to the checkpoint} --attack-method {pgd or autoattack} --pgd-steps {number of steps for pgd} --gpu_id {gpu_id}
```

## Notice Regarding SRST-AWR Code
The experiments related to SRST-AWR are based on code from the [SRST-AWR repository](https://github.com/dyoony/SRST_AWR). However, due to the absence of a formal license, their code is not included in this repository. We plan to release the related code once the licensing issue is resolved.

