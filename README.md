
# LSTM for tweet classification

*MLCS 2019 - Workshop on Machine Learning for CyberSecurity*
*Competition on Multi-Task Learning in Natural Language Processing for Cybersecurity Threat Awareness*

**Solution based on LSTM by**
TU Wien, Inst. of Telec., CN Group
FIV, Aug 2019 

## Checking dependencies

Extract the zip and save your file for testing anywhere in the same folder. From a terminal, check Python3 dependencies. You can use...

    python3 check_dependencies.py


## Analyzing test data with a trained classifier

Execute...

    python3 lstm_based_classifiers.py -e test_file

where "test_file" is a new file only containing tweets. The [data] folder contains some example files. By default, results are saved as "results.csv" in the same folder.

## Retraining classification models

In case that you want to retrain models, use new training data, or start everything from scratch, execute...

    python3 descriptive.py train_file entities_file

    python3 lstm_based_classifiers.py -t test_file -e test_file -c config_file

You can find examples for the train_file, entities_file, test_file and config_file in [data/train.csv], [obj/entities], [data/test_file], and [config/config.txt] respectively. New files should keep the same format as the example files. The test_file is not required to contain evaluation/label columns, i.e.,  'relevant', 'entities', 'A', 'B', and 'C' fields.

## Configuration parameters and default values

    HEADER_TRAIN:1       # 1 if the training_file has header  
    HEADER_TEST:0        # 1 if the test_file has header
    OUT2FILE:1           # 1 if the results/outputs are to be saved in a CSV file
    EVAL:0               # 1 if the test_file must be evaluated (data for validation is required) 
    VERBOSE:0            # 0: only predictions, 1: predictions and real (if EVAL), 2: display complete info (if EVAL), 3: display only evaluation results (if EVAL)
    SAVE_MODEL:0         # 1 saves trained models

    ENT_FILE:obj/entities.csv,      # file with entities tables for entity prediction
    OUTPUT_FILE:results.csv,        # file to save outputs/results (if OUT2FILE)
    R_MOD_FILE:obj/model_r.h5,      # file with the LSTM model for "relevance" prediction 
    A_MOD_FILE:obj/model_A.h5,      # file with the LSTM model for "A" prediction
    B_MOD_FILE:obj/model_B.h5,      # file with the LSTM model for "B" prediction
    C_MOD_FILE:obj/model_C.h5,      # file with the LSTM model for "C" prediction
    DICT_FILE:obj/lstm_dict.pkl    # file with the LSTM-dictionary


## Main files
- *check_dependencies.py*, check python dependencies
- *descriptive.py*, extracts the entity file with entities (i.e., words) and frequency values related to classification and identificaiton labels 
- *lstm_based_classif.py*, executes lstm-nn-based classification and model training 
- *README.md*, this file
- *text_processing.py*, functions for tokenizing sentences and word extraction
- *tweet_manager.py*, functions to extract and display tweet and tweet analysis information 
- The *[data]* folder contain competition data and the *create_small_datasets.sh* script for creating smaller training, test and validation splits for quick algorithm testing. 

