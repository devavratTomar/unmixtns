#!/bin/bash -l
methods=(source)           # choose from: source, tent, cotta, rotta, note, lame, roid
settings=(reset_each_shift_correlated) # choose from: reset_each_shift_correlated, correlated, mixed_domains_correlated    # 
seeds=(2020 2021 2022)  # to reproduce the benchmark results, use: (2020 2021 2022)
datasets=(cifar10_c) # choose from: cifar10_c cifar100_c imagenet_c


for dataset in ${datasets[*]}; do
      if [[ $dataset = "cifar10_c" ]]
      then
            default_options="TEST.BATCH_SIZE 64 TEST.DELTA_DIRICHLET 0.1 OPTIM.LR 1e-5"
      fi
      if [[ $dataset = "cifar100_c" ]]
      then
            default_options="TEST.BATCH_SIZE 64 TEST.DELTA_DIRICHLET 0.01 OPTIM.LR 1e-5"
      fi
      if [[ $dataset = "imagenet_c" ]]
      then
            default_options="TEST.BATCH_SIZE 16 TEST.DELTA_DIRICHLET 0.01 OPTIM.LR 2.5e-6 UNMIXTNS.BATCH_SIZE_MAX 16"
      fi

      for setting in ${settings[*]}; do
            if [ $setting == "mixed_domains_correlated" ]
            then
                  mix_k_options="UNMIXTNS.NUM_COMPONENTS 128"
            else
                  mix_k_options="UNMIXTNS.NUM_COMPONENTS 16"
            fi
            for method in ${methods[*]}; do
                  for seed in ${seeds[*]}; do
                        config=("cfgs/${dataset}/${method}.yaml")
                        python test_time.py --cfg $config SETTING $setting RNG_SEED $seed ${options} ${default_options} ${mix_k_options} SAVE_DIR ./output_unmixtns/${setting}
                  done
            done
      done
done