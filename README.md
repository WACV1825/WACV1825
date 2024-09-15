
# Anonymous Repository for WACV 2025 Submission 1825

This repository has been anonymized to comply with the double-blind review process for WACV 2025.

## Train the Teacher Model Using FixMatch

To train the teacher model using FixMatch, run the following command:

```bash
python SRST_AWR/train_teacher.py --dataset {dataset} --model {model} --depth {depth} --widen_factor {widen_factor} --num_labels {num_labels} --algo fixmatch --lamb 1 --eta 0.95
```

This part of the code is based on the implementation provided by [SRST-AWR](https://github.com/dyoony/SRST_AWR).

Once the teacher model is trained, the path to the teacher model must be manually added into `train_semi_base.py`, `train_semi_interp.py`, `SRST_AWR/main.py`, and `SRST_AWR/main_intp.py`.

## Train Robust Models Using UAT++ and RST

To train robust models using UAT++ or RST, use the following command:

```bash
python train_semi_base.py --model-dir {directory_to_save_model} --dataset {dataset} --model wideresnet --depth {depth} --widen-factor {widen_factor} --num-labels-per-class {number_of_labeled_data_per_class} --ges const --lambd {lambd} --loss-inner ce --outer {uat or rst} --gpu_id {gpu_id}
```

## Train Robust Model Using RST+Ours

To train robust models using our method (RST+Ours), use the following command:

```bash
python train_semi_interp.py --model-dir {directory_to_save_model} --dataset {dataset} --model wideresnet --depth {depth} --widen-factor {widen_factor} --rho-setup {0 or 1} --num-labels-per-class {number_of_labeled_data_per_class} --ges {global epsilon scheduling strategy} --lambd {lambd} --loss-inner ce --tau {tau} --intp-steps {K} {--mixed} --mixed-beta {beta} --wandb-project {wandb project name} --wandb-entity {wandb entity name} --gpu_id {gpu_id}
```

## Train Robust Models Using SRST-AWR

To train robust models using SRST-AWR, use the following command:

```bash
python SRST_AWR/main.py --dataset {dataset} --model wideresnet --depth {depth} --widen_factor {widen_factor} --num_labels {num_labels} --algo srst-awr --perturb_loss {perturb_loss} --teacher fixmatch --tau {tau_value_defined_in_SRSTAWR} --smooth 0.2 --lamb {lambd} --gamma 4 --beta 0.5 --lr 0.05 --swa --wandb-project {wandb project name} --wandb-entity {wandb entity name} --wandb-run-name {wandb run name} --gpu {gpu_id}
```

This part is adapted from the implementation in the [SRST-AWR repository](https://github.com/dyoony/SRST_AWR). 

## Train Robust Models Using SRST-AWR+Ours

To train robust models using SRST-AWR+Ours, use the following command:

```bash
python SRST_AWR/main_intp.py --dataset {dataset} --model wideresnet --depth {depth} --widen_factor {widen_factor} --num_labels {num_labels} --algo srst-awr --perturb_loss {perturb_loss} --teacher fixmatch --tau {tau defined in SRSTAWR paper} --tau-margin {tau defined in our paper} --smooth 0.2 --lamb {lambd} --gamma 4 --beta {beta defined in SRSTAWR paper} --lr 0.05 --swa --lr-setup {lr modes: 0, 1, or 2} --rho-setup {rho modes: 0 or 1} --intp-steps {K} --ges {global epsilon scheduling strategy} {--mixed} --mixed-beta {beta defined in our paper} --inner-target {hard or soft} --wandb-project {wandb project name} --wandb-entity {wandb entity name} --wandb-run-name {wandb run name} --gpu {gpu_id}
```

This part is adapted from the implementation in the [SRST-AWR repository](https://github.com/dyoony/SRST_AWR).


## Test Natural Accuracy and Robustness

To test natural accuracy and robustness against adversarial attacks, use the following command:

```bash
python test.py --dataset {dataset} --model wideresnet --depth {depth} --widen-factor {widen_factor} --model-dir {path to the checkpoint} --attack-method {pgd or autoattack} --pgd-steps {number of steps for pgd} --gpu_id {gpu_id}
```


