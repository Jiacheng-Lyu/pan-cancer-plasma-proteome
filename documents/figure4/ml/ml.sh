#! /bin/bash

target="LC_subtype"
features="MCHC WBC AFP"
for feature in $features; do
#     # python ml_pipeline.py --feature $feature --data_inpath ./ml_data.csv --output_path ./${target%/}_${feature%/}/ --prefix_name ${target%/} --topn 10 --n_jobs 6 --cutoffs 0.9 --ranked 0 --cv_folds 5 --preprocessing no
    python model_evaluation.py --name $target --model_inpath ./${target%/}_${feature%/}/
done

# echo all_clinical
# python ml_pipeline.py --feature MCHC WBC AFP --data_inpath ./ml_data.csv --output_path ./${target%/}_clinical/ --prefix_name ${target%/} --topn 10 --n_jobs 6 --cutoffs 0.9 --ranked 0 --cv_folds 5 --preprocessing no
python model_evaluation.py --name $target --model_inpath ./${target%/}_clinical/

target="LC_subtype"
echo all_feature
# python ml_pipeline.py --data_inpath ./${target%/}_ml_data.csv --output_path ./${target%/}_clinical_proteome/ --prefix_name ${target%/} --topn 20 --n_jobs 6 --cutoffs 0.9 --ranked 0 --cv_folds 5 --preprocessing no
python model_evaluation.py --name $target --model_inpath ./${target%/}_clinical_proteome/
