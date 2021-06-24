from recurrent_NN_sentence_classifier import *

#Experiment code
#For Table 1
params = parser.parse_args()
if params.min_complexity is None: params.min_complexity=params.complexity
params.rel_num=4
for d in ["l","r"]: 
    params.branching=d
    for a in ["LSTM","GRU","SRN"]:
        params.architecture=a
        for c in range(3,8):
            params.complexity=c
            params.min_complexity=c
            for cu in ["gentle_curriculum","no_curriculum","slow_curriculum","steep_curriculum"]:
                params.curriculum=cu
                run_with(params)

#For Table 2
params = parser.parse_args()
if params.min_complexity is None: params.min_complexity=params.complexity
params.rel_num=4
architectures=["LSTM","GRU","SRN"]
for i in range(5):
    params.top_complexity_share_in_training=i*0.2
    for a in architectures:
        params.architecture=a
        run_with(params)


#running zero-shot generalization from 3 to 4+ level of complexity

from LSTM_sentence_classifier import *
params = parser.parse_args()
if params.min_complexity is None: params.min_complexity=params.complexity
params.rel_num=4
params.top_complexity_share_in_training=0.0
params.min_complexity=4
for complexity_diff in range(4):
    params.complexity=4+complexity_diff
    for a in ["SRN","GRU","LSTM"]:
        params.architecture=a
        run_with(params)

