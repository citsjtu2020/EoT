from rouge import Rouge
import pandas as pd
from bleurt import score
import tensorflow as tf
# from scoring_for_reasoning import MASK

checkpoint = "../BLEURT-20"

references = ["This is a test."]
candidates = ["This is the test."]
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

# Compute the similarity scores


def compute_rouge_score(x_pdf, ref_col="summary", hyp_col="summary_llm"):
    """
        调用 Rouge 方法来评测 两个结果之间的差异与相似度
    """
    rouger = Rouge()
    ref_data = list(x_pdf[ref_col].values)
    hyp_data = list(x_pdf[hyp_col].values)

    assert len(ref_data) == len(hyp_data)

    res_rouge_1_r = []
    res_rouge_1_p = []
    res_rouge_1 = []
    res_rouge_2_r = []
    res_rouge_2_p = []
    res_rouge_2 = []
    res_rouge_l_r = []
    res_rouge_l_p = []
    res_rouge_l = []

    for k in range(len(ref_data)):
        scores = rouger.get_scores(' '.join(list(hyp_data[k])), ' '.join(list(ref_data[k])))
        res_rouge_1_r.append(scores[0]['rouge-1']['r'])
        res_rouge_1_p.append(scores[0]['rouge-1']['p'])
        res_rouge_1.append(scores[0]['rouge-1']['f'])
        res_rouge_2_r.append(scores[0]['rouge-2']['r'])
        res_rouge_2_p.append(scores[0]['rouge-2']['p'])
        res_rouge_2.append(scores[0]['rouge-2']['f'])
        res_rouge_l_r.append(scores[0]['rouge-l']['r'])
        res_rouge_l_p.append(scores[0]['rouge-l']['p'])
        res_rouge_l.append(scores[0]['rouge-l']['f'])

    out_pdf = x_pdf.copy()
    out_pdf = out_pdf.reset_index(drop=True)

    # out_pdf["rouge-1_recall"] = pd.Series(res_rouge_1_r)
    # out_pdf["rouge-1_precision"] = pd.Series(res_rouge_1_p)
    # out_pdf["rouge-1"] = pd.Series(res_rouge_1)

    # out_pdf["rouge-2_recall"] = pd.Series(res_rouge_2_r)
    # out_pdf["rouge-2_precision"] = pd.Series(res_rouge_2_p)
    # out_pdf["rouge-2"] = pd.Series(res_rouge_2)

    out_pdf["rouge-l_recall"] = pd.Series(res_rouge_l_r)
    out_pdf["rouge-l_precision"] = pd.Series(res_rouge_l_p)
    out_pdf["rouge-l"] = pd.Series(res_rouge_l)

    return out_pdf

def compute_bert_score_new(x_pdf,scorer_0:score.BleurtScorer,
                       scorer_1:score.BleurtScorer,
                       scorer_2:score.BleurtScorer,
                       scorer_3:score.BleurtScorer,
                       ref_col="answer",hyp_col="summary_turing3",
                       out_col="bleurt_score3",
                       device='/device:GPU:1',
                       aggr_method="mean"):
    references_vals = x_pdf[ref_col].values
    candidates_vals = x_pdf[hyp_col].values
    with tf.device(device):
        if "GPU:1" in device:
            scores = scorer_1.score(references=references_vals,candidates=candidates_vals)
        elif "GPU:2" in device:
            scores = scorer_2.score(references=references_vals,candidates=candidates_vals)
        elif "GPU:3" in device:
            scores = scorer_3.score(references=references_vals,candidates=candidates_vals)
        else:
            scores = scorer_0.score(references=references_vals,candidates=candidates_vals)
    if "max" in aggr_method:
        out_pdf = pd.DataFrame({out_col:[np.max(scores)]})
    else:
        out_pdf = pd.DataFrame({out_col:[np.mean(scores)]})
    return out_pdf