process lin_reg {                                                                  
    input:                                                                         
    val toml                                                                       
    val toml_path                                                                  
                                                                                   
    output:                                                                        
    stdout                                                                         
                                                                                   
    when:                                                                          
    params.prediction_task == 'regression' || params.prediction_task == 'both' || params.prediction_task == 'lin_reg'  
                                                                                   
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
    params.prediction_task == 'classification' || params.prediction_task == 'both' || params.prediction_task == 'lin_clf'
                                                                                   
    script:                                                                        
    """                                                                            
    source ~/.virtualenvs/bin/activate                                             
    cd ~/drp_model_suite/Baselines/logistic_regression_classifier                  
    python -m main -t $toml -d $params.global_output_dir/linclf/$toml_path/        
    """                                                                            
}

process svc {                                                                  
    input:                                                                         
    val toml                                                                       
    val toml_path                                                                  
                                                                                   
    output:                                                                        
    stdout                                                                         
                                                                                   
    when:                                                                          
    params.prediction_task == 'classification' || params.prediction_task == 'both' || params.prediction_task == 'svc'
                                                                                   
    script:                                                                        
    """                                                                            
    source ~/.virtualenvs/bin/activate                                             
    cd ~/drp_model_suite/Baselines/support_vector_classifier                              
    python -m main -t $toml -d $params.global_output_dir/svc/$toml_path/        
    """                                                                            
}
                                                                                    
process svr {                                                                  
    input:                                                                         
    val toml                                                                       
    val toml_path                                                                  
                                                                                   
    output:                                                                        
    stdout                                                                         
                                                                                   
    when:                                                                          
    params.prediction_task == 'regression' || params.prediction_task == 'both' || params.prediction_task == 'svr' 
                                                                                   
    script:                                                                        
    """                                                                            
    source ~/.virtualenvs/bin/activate                                             
    cd ~/drp_model_suite/Baselines/support_vector_regressor 
    python -m main -t $toml -d $params.global_output_dir/svr/$toml_path/        
    """                                                                            
}

process rfc {                                                                  
    input:                                                                         
    val toml                                                                       
    val toml_path                                                                  
                                                                                   
    output:                                                                        
    stdout                                                                         
                                                                                   
    when:                                                                          
    params.prediction_task == 'classification' || params.prediction_task == 'both' || params.prediction_task == 'rfc' 
                                                                                   
    script:                                                                        
    """                                                                            
    source ~/.virtualenvs/bin/activate                                             
    cd ~/drp_model_suite/Baselines/random_forest_classifier                              
    python -m main -t $toml -d $params.global_output_dir/rfc/$toml_path/        
    """                                                                            
}

process rfr {                                                                  
    input:                                                                         
    val toml                                                                       
    val toml_path                                                                  
                                                                                   
    output:                                                                        
    stdout                                                                         
                                                                                   
    when:                                                                          
    params.prediction_task == 'regression' || params.prediction_task == 'both' || params.prediction_task == 'rfr' 
                                                                                   
    script:                                                                        
    """                                                                            
    source ~/.virtualenvs/bin/activate                                             
    cd ~/drp_model_suite/Baselines/random_forest_regressor
    python -m main -t $toml -d $params.global_output_dir/rfr/$toml_path/        
    """                                                                            
}

process gbc {                                                                  
    input:                                                                         
    val toml                                                                       
    val toml_path                                                                  
                                                                                   
    output:                                                                        
    stdout                                                                         
                                                                                   
    when:                                                                          
    params.prediction_task == 'classification' || params.prediction_task == 'both' || params.prediction_task == 'gbc' 
                                                                                   
    script:                                                                        
    """                                                                            
    source ~/.virtualenvs/bin/activate                                             
    cd ~/drp_model_suite/Baselines/gradient_boost_classifier
    python -m main -t $toml -d $params.global_output_dir/gbc/$toml_path/        
    """                                                                            
}

process gbr {                                                                  
    input:                                                                         
    val toml                                                                       
    val toml_path                                                                  
                                                                                   
    output:                                                                        
    stdout                                                                         
                                                                                   
    when:                                                                          
    params.prediction_task == 'regression' || params.prediction_task == 'both' || params.prediction_task == 'gbr' 
                                                                                   
    script:                                                                        
    """                                                                            
    source ~/.virtualenvs/bin/activate                                             
    cd ~/drp_model_suite/Baselines/gradient_boost_regressor
    python -m main -t $toml -d $params.global_output_dir/gbr/$toml_path/        
    """                                                                            
}
