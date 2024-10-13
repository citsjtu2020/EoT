DATA_PATH="/root/data/huaqin/"
import os
checkpoint = os.path.join(DATA_PATH,"BLEURT-20")
CODE_PATH="/root/code/huaqin/Reasoning/EoT-ENG/"
references = ["This is a test."]
candidates = ["This is the test."]
import numpy as np
import os
from prompt_design import CON_PART_STRUCT,NO_STRUCT
from Faithfulity_score_compute import compute_clipped_scores,compute_nli_based_score_for_batch
from prompt_design import generate_history_reasoning_process_item,EX_COMP_start_token,EX_COMP_end_token,NLI_PROMPT_TEMPLATE
from evolution_procedure import evo_reasoning_procedure
from evolution_generation import  BASE_URL_GPT,BASE_URL_QWEN
from scoring_metrics import MASK,MODIFY,HOLD
import copy
from prompt_design import PRE_start_token,PRE_end_token,HYP_end_token,HYP_start_token,NLI_start_token,NLI_end_token


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import pandas as pd
import tensorflow as tf
from bleurt import score
'''
The codes which implement the experiments of EoT and the corresponding results analysis
'''

def evaluate_global_metric(his_rp_set, scorer_0: score.BleurtScorer,
                           scorer_1: score.BleurtScorer,
                           scorer_2: score.BleurtScorer,
                           scorer_3: score.BleurtScorer,
                           faith_col="faith_score_hyb", fact_col="fact_score",
                           truth_col="truth_score_hyb", raw_answer_col="manual_answers",
                           answer_col="hist_replay_answer",
                           question_col="input",
                           add_metrics=["bleurt", "rouge", "nli"], repeat_time=2, base_url=BASE_URL_GPT,
                           gpt_model_name="gpt-4-turbo-128k",
                           connect_repeat_time=2,
                           nli_repeat_time=2,
                           nli_thread_num=3,
                           computing_thread_num=4, scorer_num=4, default_device='/device:GPU:1',
                           add_structure=NO_STRUCT):
    '''

    This function analyzes each reasoning process produced in iterations of evolutions.
    :param his_rp_set: generated results
    :param scorer_0: BERT model instance 0
    :param scorer_1: BERT model instance 1
    :param scorer_2: BERT model instance 2
    :param scorer_3: BERT model instance 3
    :param faith_col: column name of fidelity score
    :param fact_col: column name of factuality score
    :param truth_col: column name of reliability score
    :param raw_answer_col: column name of reference answer
    :param answer_col: column name of generated answer under the guideline of produced reasoning process
    :param question_col: column name of question context
    :param add_metrics: considered widely used metrics for the performance evaluation
    :param repeat_time: repeat time of evaluate performance
    :param base_url: Base url to connect GPT
    :param gpt_model_name: name of used GPT-model
    :param connect_repeat_time: repeat time of connect GPT
    :param nli_repeat_time: repeat time of NLI inference
    :param nli_thread_num: number of threads for NLI inference
    :param computing_thread_num: number of threads for computing metrics
    :param scorer_num: number of BERT instance
    :param default_device: the default device to deploy BERT model
    :param add_structure: if considering the window granularity for BLEURT estimation
    :return: the aggregated results and source results of performance analysis
    '''
    total_questions = list(his_rp_set.keys())
    total_out_pdf = pd.DataFrame()
    total_out_source_pdf = pd.DataFrame()
    for tq in total_questions:
        out_pdf = {"iter": [], question_col: [], fact_col: [], truth_col: [],
                   faith_col: [], f"{faith_col}_raw": [], "reasoning": [], answer_col: [], raw_answer_col: []}
        qes_his_rp_set = his_rp_set[tq].copy()
        out_pdf[question_col] = [tq] * len(list(qes_his_rp_set.keys()))
        qes_iter_set = list(qes_his_rp_set.keys())
        qes_iter_set.sort()
        for qi in qes_iter_set:
            out_pdf["iter"].append(int(qi))
            # print(qes_his_rp_set[qi]["source"].columns)
            out_pdf[fact_col].append(qes_his_rp_set[qi]["source"][fact_col].mean())
            out_pdf[faith_col].append(qes_his_rp_set[qi]["source"][faith_col].mean())
            out_pdf[truth_col].append(qes_his_rp_set[qi]["source"][truth_col].mean())
            out_pdf[f"{faith_col}_raw"].append(qes_his_rp_set[qi]["source"][f"{faith_col}_raw"].mean())
            out_pdf[answer_col].append(qes_his_rp_set[qi]["source"][answer_col].values[0])
            out_pdf[raw_answer_col].append(qes_his_rp_set[qi]["source"][raw_answer_col].values[0])
            out_pdf["reasoning"].append(generate_history_reasoning_process_item(
                input_pdf=qes_his_rp_set[qi]["source"]))
            tmp_source_pdf = qes_his_rp_set[qi]["source"].reset_index(drop=True).copy()
            tmp_source_pdf["iter"] = qi

            if total_out_source_pdf.shape[0] < 1:
                total_out_source_pdf = tmp_source_pdf.copy()
            else:
                total_out_source_pdf = pd.concat([total_out_source_pdf.reset_index(drop=True),
                                                  tmp_source_pdf.copy().reset_index(drop=True)],
                                                 axis=0).reset_index(drop=True)

        out_pdf = pd.DataFrame(out_pdf)
        if total_out_pdf.shape[0] < 1:
            total_out_pdf = out_pdf.copy().reset_index(drop=True)
        else:
            total_out_pdf = pd.concat([total_out_pdf.reset_index(drop=True),
                                       out_pdf.copy().reset_index(drop=True)], axis=0)
    for am in add_metrics:
        if "nli" not in am:
            total_out_pdf_metric = compute_clipped_scores(x_pdf=total_out_pdf,
                                                          scorer_0=scorer_0,
                                                          scorer_1=scorer_1,
                                                          scorer_2=scorer_2,
                                                          scorer_3=scorer_3,
                                                          metric=am, ref_col=raw_answer_col,
                                                          base_col=raw_answer_col,
                                                          hyp_col=answer_col,
                                                          id_col=[question_col, "iter"],
                                                          method=2, aggr_method="mean",
                                                          aim_col=f"{am}_score",
                                                          worker_num=computing_thread_num,
                                                          default_device=default_device,
                                                          scorer_num=scorer_num)
            total_out_pdf = pd.merge(total_out_pdf, total_out_pdf_metric[[question_col, "iter", f"{am}_score"]],
                                     on=[question_col, "iter"], how="inner")
        else:
            total_out_pdf_metric = pd.DataFrame()
            for rp_id in range(repeat_time):
                if "nli_" not in am:
                    tmp_out_pdf_metric = compute_nli_based_score_for_batch(x_pdf=total_out_pdf.copy(),
                                                                           scorer_0=scorer_0,
                                                                           scorer_1=scorer_1,
                                                                           scorer_2=scorer_2,
                                                                           scorer_3=scorer_3,
                                                                           pre_col=raw_answer_col,
                                                                           hyp_col=answer_col,
                                                                           metric="nli",
                                                                           group_key_cols=[question_col, "iter"],
                                                                           nli_prompt=NLI_PROMPT_TEMPLATE["ENG"],
                                                                           pre_start_token=PRE_start_token,
                                                                           pre_end_token=PRE_end_token,
                                                                           hyp_start_token=HYP_start_token,
                                                                           hyp_end_token=HYP_end_token,
                                                                           label_start_token=NLI_start_token,
                                                                           label_end_token=NLI_end_token,
                                                                           comp_start_token=EX_COMP_start_token,
                                                                           comp_end_token=EX_COMP_end_token,
                                                                           base_url=base_url,
                                                                           model_name=gpt_model_name,
                                                                           connect_repeat_time=connect_repeat_time,
                                                                           nli_repeat_time=nli_repeat_time,
                                                                           split_tokens=["\n", "。"],
                                                                           method=2, item_limit=-1, add_semantic=False,
                                                                           aim_col=f"{am}_score_{rp_id}",
                                                                           nli_thread_num=nli_thread_num,
                                                                           computing_thread_num=computing_thread_num,
                                                                           default_device=default_device,
                                                                           scorer_num=scorer_num,
                                                                           add_structure=NO_STRUCT)
                    # "entail_out":[],"contrad_out":[],"neural_out":[]
                    tmp_out_pdf_metric.rename(columns={"entail_out": f"entail_out_{rp_id}",
                                                       "contrad_out": f"contrad_out_{rp_id}",
                                                       "neural_out": f"neural_out_{rp_id}",
                                                       }, inplace=True)
                else:
                    tmp_out_pdf_metric = compute_nli_based_score_for_batch(x_pdf=total_out_pdf.copy(),
                                                                           pre_col=raw_answer_col,
                                                                           hyp_col=answer_col,
                                                                           scorer_0=scorer_0,
                                                                           scorer_1=scorer_1,
                                                                           scorer_2=scorer_2,
                                                                           scorer_3=scorer_3,
                                                                           metric="nli",
                                                                           group_key_cols=[question_col, "iter"],
                                                                           nli_prompt=NLI_PROMPT_TEMPLATE["ENG"],
                                                                           pre_start_token=PRE_start_token,
                                                                           pre_end_token=PRE_end_token,
                                                                           hyp_start_token=HYP_start_token,
                                                                           hyp_end_token=HYP_end_token,
                                                                           label_start_token=NLI_start_token,
                                                                           label_end_token=NLI_end_token,
                                                                           comp_start_token=EX_COMP_start_token,
                                                                           comp_end_token=EX_COMP_end_token,
                                                                           base_url=base_url,
                                                                           model_name=gpt_model_name,
                                                                           connect_repeat_time=connect_repeat_time,
                                                                           nli_repeat_time=nli_repeat_time,
                                                                           split_tokens=["\n", "。"],
                                                                           method=2, item_limit=-1, add_semantic=True,
                                                                           aim_col=f"{am}_score_{rp_id}",
                                                                           nli_thread_num=nli_thread_num,
                                                                           computing_thread_num=computing_thread_num,
                                                                           default_device=default_device,
                                                                           scorer_num=scorer_num,
                                                                           add_structure=add_structure)

                if "nli_" not in am:
                    if total_out_pdf_metric.shape[0] < 1:
                        total_out_pdf_metric = tmp_out_pdf_metric[[question_col, "iter",
                                                                   f"{am}_score_{rp_id}", f"entail_out_{rp_id}",
                                                                   f"contrad_out_{rp_id}",
                                                                   f"neural_out_{rp_id}"]].copy().reset_index(drop=True)
                    else:
                        total_out_pdf_metric = pd.merge(total_out_pdf_metric.reset_index(drop=True),
                                                        tmp_out_pdf_metric[[question_col, "iter",
                                                                            f"{am}_score_{rp_id}",
                                                                            f"entail_out_{rp_id}",
                                                                            f"contrad_out_{rp_id}",
                                                                            f"neural_out_{rp_id}"]].copy().reset_index(
                                                            drop=True),
                                                        on=[question_col, "iter"], how="inner").reset_index(drop=True)

                else:
                    if total_out_pdf_metric.shape[0] < 1:
                        total_out_pdf_metric = tmp_out_pdf_metric[
                            [question_col, "iter", f"{am}_score_{rp_id}"]].copy().reset_index(drop=True)
                    else:
                        total_out_pdf_metric = pd.merge(total_out_pdf_metric.reset_index(drop=True),
                                                        tmp_out_pdf_metric[[question_col, "iter",
                                                                            f"{am}_score_{rp_id}"]].copy().reset_index(
                                                            drop=True),
                                                        on=[question_col, "iter"], how="inner").reset_index(drop=True)

            total_out_pdf_metric[f"{am}_score"] = 0.0
            if "nli_" not in am:
                total_out_pdf_metric["entail_out"] = 0.0
                total_out_pdf_metric["contrad_out"] = 0.0
                total_out_pdf_metric["neural_out"] = 0.0

            for rp_id in range(repeat_time):

                total_out_pdf_metric[f"{am}_score"] = total_out_pdf_metric[f"{am}_score"] + total_out_pdf_metric[
                    f"{am}_score_{rp_id}"]
                if "nli_" not in am:
                    total_out_pdf_metric["entail_out"] = total_out_pdf_metric["entail_out"] + total_out_pdf_metric[
                        f"entail_out_{rp_id}"]
                    total_out_pdf_metric["contrad_out"] = total_out_pdf_metric["contrad_out"] + total_out_pdf_metric[
                        f"contrad_out_{rp_id}"]
                    total_out_pdf_metric["neural_out"] = total_out_pdf_metric["neural_out"] + total_out_pdf_metric[
                        f"neural_out_{rp_id}"]

            total_out_pdf_metric[f"{am}_score"] = total_out_pdf_metric[f"{am}_score"] / repeat_time

            if "nli_" not in am:
                total_out_pdf_metric["entail_out"] = total_out_pdf_metric["entail_out"] / repeat_time
                total_out_pdf_metric["contrad_out"] = total_out_pdf_metric["contrad_out"] / repeat_time
                total_out_pdf_metric["neural_out"] = total_out_pdf_metric["neural_out"] / repeat_time

            if "nli_" not in am:
                total_out_pdf = pd.merge(total_out_pdf, total_out_pdf_metric[[question_col, "iter", f"{am}_score",
                                                                              "entail_out", "contrad_out",
                                                                              "neural_out"]],
                                         on=[question_col, "iter"], how="inner")
            else:
                total_out_pdf = pd.merge(total_out_pdf, total_out_pdf_metric[[question_col, "iter", f"{am}_score"]],
                                         on=[question_col, "iter"], how="inner")

    return total_out_pdf, total_out_source_pdf


import argparse
'''
The parameters of experiments
'''
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--start_id', type=int,default=0,help="start of index")
parser.add_argument("--exp",type=str,default="1-CON_STRUCT",help="exp name")
parser.add_argument('--proc_num', type=int,default=10,help="total proc qes num")
parser.add_argument('--evo_model_name', type=str,default="gpt-4-turbo-128k",help="evo model name")
parser.add_argument("--is_fact",type=int,default=1)
parser.add_argument("--is_truth",type=int,default=1)
parser.add_argument("--strategy",type=str,default="MASK",help="strategy of mask/modify/hold")
parser.add_argument("--is_super",type=int,default=1)
parser.add_argument("--use_reliability",type=int,default=1)
parser.add_argument("--use_factual",type=int,default=1)
parser.add_argument("--use_fidelity",type=int,default=1)
# is_fact=True,
#                                is_truth=True,




args = parser.parse_args()



with tf.device('/device:GPU:1'):
    scorer_1 = score.BleurtScorer(checkpoint)

with tf.device('/device:GPU:0'):
    scorer_0 = score.BleurtScorer(checkpoint)

with tf.device('/device:GPU:2'):
    scorer_2 = score.BleurtScorer(checkpoint)

with tf.device('/device:GPU:3'):
    scorer_3 = score.BleurtScorer(checkpoint)

with tf.device('/device:GPU:1'):
    scores = scorer_1.score(references=references, candidates=candidates)
assert isinstance(scores, list) and len(scores) == 1
print(scores)

history_reasoning_process_set_gpt = {}
history_reasoning_process_set_qwen = {}
history_reasoning_process_set_gpt_raw2 = {}
history_reasoning_process_set_qwen_raw2 = {}

history_reasoning_process_set_gpt_splited_struct = {}
history_reasoning_process_set_qwen_splited_struct = {}
history_reasoning_process_set_gpt_splited_struct_raw2 = {}
history_reasoning_process_set_qwen_splited_struct_raw2 = {}


history_reasoning_process_set_gpt_con_struct = {}
history_reasoning_process_set_qwen_con_struct0 = {}
history_reasoning_process_set_qwen_con_struct1 = {}
history_reasoning_process_set_qwen_con_struct2 = {}
history_reasoning_process_set_qwen_con_struct3 = {}
history_reasoning_process_set_qwen_con_struct4 = {}
history_reasoning_process_set_qwen_con_struct5 = {}
history_reasoning_process_set_qwen_con_struct6 = {}
history_reasoning_process_set_qwen_con_struct7 = {}
history_reasoning_process_set_qwen_con_struct8 = {}
history_reasoning_process_set_qwen_con_struct9 = {}

history_reasoning_process_set_gpt_con_struct_raw2 = {}
history_reasoning_process_set_qwen_con_struct_raw2 = {}

history_reasoning_process_set_gpt_total_struct = {}
history_reasoning_process_set_qwen_total_struct = {}
history_reasoning_process_set_gpt_total_struct_raw2 = {}
history_reasoning_process_set_qwen_total_struct_raw2 = {}

test_initial_data = pd.read_csv(os.path.join(CODE_PATH,"long_bench_test_data.csv"))
if "initial_reasoning" in test_initial_data.columns:
    test_initial_data_cols = list(test_initial_data.columns)
    test_initial_data_cols.remove("initial_reasoning")
    test_initial_data = test_initial_data[test_initial_data_cols].reset_index(drop=True)

test_question_series = test_initial_data.input.unique().tolist()[args.start_id:args.start_id+args.proc_num]
print(test_question_series)
print(len(test_question_series))


if args.is_fact > 0:
    is_fact = True
else:
    is_fact = False

if args.is_truth > 0:
    is_truth = True
else:
    is_truth = False

if args.is_super > 0:
    is_super = True
else:
    is_super = False

if args.use_fidelity > 0:
    use_fidelity = True
else:
    use_fidelity = False

if args.use_reliability > 0:
    use_reliability = True
else:
    use_reliability = False

if args.use_factual:
    use_factual = True
else:
    use_factual = False

strategy_str = args.strategy.lower()
if "ma" in strategy_str:
    strategy = MASK
elif "mo" in strategy_str:
    strategy = MODIFY
else:
    strategy = HOLD

raw_name = "source_results"
if is_super:
    raw_name = f"supervise_{raw_name}"
else:
    raw_name = f"unsupervise_{raw_name}"

if is_truth:
    raw_name = f"{raw_name}_weight"
else:
    raw_name = f"{raw_name}_no_weight"

if strategy == MODIFY:
    raw_name = f"{raw_name}_modify"
else:
    raw_name = f"{raw_name}_mask"

if not use_reliability:
    raw_name = f"{raw_name}_unreli"

if not use_fidelity:
    raw_name = f"{raw_name}_unfaith"

if not use_factual:
    raw_name = f"{raw_name}_unfact"



out_source_path = os.path.join(DATA_PATH,raw_name)

EXP = 2
import os
if not os.path.exists(out_source_path):
    os.makedirs(out_source_path)

if not os.path.exists(os.path.join(out_source_path,"gpt")):
    os.makedirs(os.path.join(out_source_path,"gpt"))

if not os.path.exists(os.path.join(out_source_path,"qwen")):
    os.makedirs(os.path.join(out_source_path,"qwen"))

gpt_root_path = os.path.join(out_source_path,"gpt")
qwen_root_path = os.path.join(out_source_path,"qwen")

if not os.path.exists(os.path.join(gpt_root_path,"weight")):
    os.makedirs(os.path.join(gpt_root_path,"weight"))

if not os.path.exists(os.path.join(gpt_root_path,"no_weight")):
    os.makedirs(os.path.join(gpt_root_path,"no_weight"))


if not os.path.exists(os.path.join(qwen_root_path,"weight")):
    os.makedirs(os.path.join(qwen_root_path,"weight"))

if not os.path.exists(os.path.join(qwen_root_path,"no_weight")):
    os.makedirs(os.path.join(qwen_root_path,"no_weight"))


if "gpt" in args.evo_model_name:
    aim_root_path = gpt_root_path
else:
    aim_root_path = qwen_root_path




TOTAL_EXP_NUM = args.proc_num
NOW_EXP_NUM= 0
EXP = args.exp

import time

AA = 0


'''
Executing experiments on a group of aim questions to evaluate the performance and effectiveness of EoT
'''

for i in range(NOW_EXP_NUM,NOW_EXP_NUM + TOTAL_EXP_NUM):
    if AA > 0:
        time.sleep(120)
    print(test_question_series[i:i+1])

    print(f"evo model name: {args.evo_model_name}")

    if args.is_fact > 0:
        is_fact = True
    else:
        is_fact = False

    if args.is_truth > 0:
        is_truth = True
    else:
        is_truth = False

    if args.is_super > 0:
        is_super = True
    else:
        is_super = False

    strategy_str = args.strategy.lower()
    if "ma" in strategy_str:
        strategy = MASK
    elif "mo" in strategy_str:
        strategy = MODIFY
    else:
        strategy = HOLD

    '''
    invoke the evo_reasoning_procedure to implement the reasoning process evolution
    '''

    history_reasoning_process_set_qwen_series,global_item_id_qwen_series = evo_reasoning_procedure(
                               initial_data=test_initial_data,
                               few_shot_input_data=test_initial_data,
                               scorer_0=scorer_0,scorer_1=scorer_1,scorer_2=scorer_2,scorer_3=scorer_3,
                               hist_rp_set={},iter_limit=10,
                               aim_qes=test_question_series[i:i+1],
                              question_col="input", answer_col="manual_answers",
                              reference_col="context",hist_item_limit=5,
                               prompt_template="[PROMPT] [INSTRUCTION]",
                               now_reasoning_col="evo_reasoning_process",
                               faith_col="faith_score_hyb",
                               truth_col="truth_score_hyb",
                               fact_col="fact_score",
                               evo_base_url=BASE_URL_GPT,
                               gpt_model_name="gpt-4-turbo-128k",
                               evo_model_name=args.evo_model_name,
                               connect_repeat_time=2,
                               evo_repeat_time=2,
                               evo_thread_num=3,
                               is_fact=is_fact,
                               is_truth=is_truth,
                               score_strategy=strategy,
                               faith_metric="nli_bleurt",
                               modify_repeat_time=2,
                               modify_worker_num=3,
                               hist_replay_answer_col="hist_replay_answer",
                               faith_replay_answer_col="replay_answer",
                               replay_thread_num = 4,
                               nli_thread_num = 3,
                               computing_thread_num=4,
                               nli_repeat_time=2,
                               split_tokens=["\n","。"],
                               faith_method=2,
                               item_limit=100,
                               add_semantic=True,
                               default_device='/device:GPU:1',
                               scorer_num=4,score_repeat_time=2,
                               add_structure=CON_PART_STRUCT,
                               is_replay=True,is_super=is_super,use_factual=use_factual,
                               use_fidelity=use_fidelity,use_reliability=use_reliability)

    for aq in history_reasoning_process_set_qwen_series.keys():
        if aq not in history_reasoning_process_set_qwen_con_struct0.keys():
            history_reasoning_process_set_qwen_con_struct0[aq] = copy.deepcopy(
                history_reasoning_process_set_qwen_series[aq])

    time.sleep(60)

    tmp_total_qes = list(history_reasoning_process_set_qwen_con_struct0.keys())
    '''
    invoke the evaluate_global_metric to analyze the performance on the three factors of factuality, fidelity and reliability for each generated reasoning process
    '''
    if len(list(history_reasoning_process_set_qwen_con_struct0.keys())) > 0:
        evo_qwen_results_con_struct, evo_qwen_raw_source_con_struct = evaluate_global_metric(
            scorer_0=scorer_0, scorer_1=scorer_1, scorer_2=scorer_2, scorer_3=scorer_3,
            his_rp_set=history_reasoning_process_set_qwen_con_struct0,
            faith_col="faith_score_hyb", fact_col="fact_score",
            truth_col="truth_score_hyb", raw_answer_col="manual_answers",
            answer_col='hist_replay_answer',
            question_col="input",
            add_metrics=["bleurt", "rouge", "nli"], repeat_time=2, base_url=BASE_URL_GPT,
            gpt_model_name="gpt-4-turbo-128k",
            connect_repeat_time=2,
            nli_repeat_time=2,
            nli_thread_num=3,
            computing_thread_num=4, scorer_num=4, default_device='/device:GPU:1')

        if not os.path.exists(os.path.join(os.path.join(aim_root_path, "weight"), "supervised")):
            os.makedirs(os.path.join(os.path.join(aim_root_path, "weight"), "supervised"))

        qwen_super_path = os.path.join(os.path.join(aim_root_path, "weight"), "supervised")

        if not os.path.exists(os.path.join(qwen_super_path, f"{EXP}")):
            os.makedirs(os.path.join(qwen_super_path, f"{EXP}"), exist_ok=True)

        if os.path.exists(os.path.join(os.path.join(qwen_super_path, f"{EXP}"), "aggre_results.csv")):
            evo_qwen_results_con_struct_old = pd.read_csv(
                os.path.join(os.path.join(qwen_super_path, f"{EXP}"), "aggre_results.csv"))
            evo_qwen_results_con_struct_old = evo_qwen_results_con_struct_old[
                ~(evo_qwen_results_con_struct_old["input"].isin(tmp_total_qes))].reset_index(drop=True)
        else:
            evo_qwen_results_con_struct_old = pd.DataFrame()

        if os.path.exists(os.path.join(os.path.join(qwen_super_path, f"{EXP}"), "source_results.csv")):
            evo_qwen_raw_source_con_struct_old = pd.read_csv(
                os.path.join(os.path.join(qwen_super_path, f"{EXP}"), "source_results.csv"))
            evo_qwen_raw_source_con_struct_old = evo_qwen_raw_source_con_struct_old[
                ~(evo_qwen_raw_source_con_struct_old["input"].isin(tmp_total_qes))].reset_index(drop=True)
        else:
            evo_qwen_raw_source_con_struct_old = pd.DataFrame()

        if evo_qwen_results_con_struct_old.shape[0] > 0:
            evo_qwen_results_con_struct = pd.concat([evo_qwen_results_con_struct_old, evo_qwen_results_con_struct],
                                                axis=0).reset_index(drop=True)

        if evo_qwen_raw_source_con_struct_old.shape[0] > 0:
            evo_qwen_raw_source_con_struct = pd.concat([evo_qwen_raw_source_con_struct_old, evo_qwen_raw_source_con_struct],
                                                   axis=0).reset_index(drop=True)

        evo_qwen_results_con_struct.to_csv(os.path.join(os.path.join(qwen_super_path, f"{EXP}"), "aggre_results.csv"),
                                       index=False)
        evo_qwen_raw_source_con_struct.to_csv(os.path.join(os.path.join(qwen_super_path, f"{EXP}"), "source_results.csv"),
                                          index=False)

    history_reasoning_process_set_qwen_con_struct0 = {}
    AA += 1