#!/usr/bin/env nextflow

tomls_ch = Channel.of('metadata_LCO.toml','metadata_LDO.toml','metadata_LPO.toml')
toml_path_ch = Channel.of('LCO','LDO','LPO')

params.prediction_task = 'both'
params.global_output_dir = '~/nf/output'

log.info """\
    BASELINE    MODEL    EVALUATION
    ===============================
    prediciton task: ${params.prediction_task}
    output directory: ${params.global_output_dir}
    """
    .stripIndent(true)

include { lin_reg } from './modules.nf'
include { lin_clf } from './modules.nf'
include { svc } from './modules.nf'
include { svr } from './modules.nf'
include { rfc } from './modules.nf'
include { rfr } from './modules.nf'
include { gbc } from './modules.nf'
include { gbr } from './modules.nf'

/*
process lin_reg { 
    input:
    val toml
    val toml_path

    output: 
    stdout 

    when: 
    params.prediction_task == 'regression' || params.prediction_task == 'both'

    script: 
    """
    source ~/.virtualenvs/bin/activate
    cd ~/drp_model_suite/Baselines/linear_regression
    python -m main -t $toml -d $params.global_output_dir/linreg/$toml_path/
    """
} 

process lin_clf { 
    input:
    val toml
    val toml_path

    output: 
    stdout 

    when: 
    params.prediction_task == 'classification' || params.prediction_task == 'both'

    script: 
    """
    source ~/.virtualenvs/bin/activate
    cd ~/drp_model_suite/Baselines/logistic_regression_classifier
    python -m main -t $toml -d $params.global_output_dir/linclf/$toml_path/
    """
} 
*/

workflow { 
    lin_reg(tomls_ch, toml_path_ch) 
    lin_clf(tomls_ch, toml_path_ch)
    svc(tomls_ch, toml_path_ch)
    svr(tomls_ch, toml_path_ch)
    rfc(tomls_ch, toml_path_ch)
    rfr(tomls_ch, toml_path_ch)
    gbc(tomls_ch, toml_path_ch)
    gbr(tomls_ch, toml_path_ch)
} 
