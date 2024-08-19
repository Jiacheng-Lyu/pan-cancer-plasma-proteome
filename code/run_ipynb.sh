#! /bin/bash

files="1-cohort_description-QC 2-proteome_clustering 3-digestive_cluster_characterization 4-system_group_characterization 5-LC_subtypes_characterization 6-core_tumor_marker 7_ML_model"
for file in $files; do
    ipython -c "run ${number%/}.ipynb"
done


