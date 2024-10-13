import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from NLI_socre_compute import nli_distribution_aggr,NLI_PROMPT_TEMPLATE
from construct_reasoning_sources import aggr_splited_steps

from prompt_design import NLI_start_token,NLI_end_token,PRE_end_token,PRE_start_token,HYP_end_token,HYP_start_token
from prompt_design import EX_COMP_end_token,EX_COMP_start_token,generate_fine_grained_semantic_compoenments,EXTRACT_RELATION_COMP_PROP
from NLI_socre_compute import generate_nli_relation_single
from prompt_design import NO_STRUCT,CON_PART_STRUCT,TOTAL_STRUCT,SPLITED_STRUCT

import numpy as np
import time
from related_extract_generation import extracted_related_componments_single
from similarity_scores import  compute_rouge_score,compute_bert_score_new
from bleurt import score


def clipped_scores(x_pdf, upper_limit=1.0, lower_limit=0.0, aim_col="score"):
    clipped_upper_pdf = x_pdf[x_pdf[aim_col] >= upper_limit].reset_index(drop=True)
    if clipped_upper_pdf.shape[0] > 0:
        clipped_upper_pdf[aim_col] = upper_limit

    clipped_lower_pdf = x_pdf[x_pdf[aim_col] <= lower_limit].reset_index(drop=True)
    if clipped_lower_pdf.shape[0] > 0:
        clipped_lower_pdf[aim_col] = lower_limit

    clipped_remain_pdf = x_pdf[(x_pdf[aim_col] > lower_limit) & (x_pdf[aim_col] < upper_limit)].reset_index(drop=True)
    out_pdf = pd.concat([clipped_upper_pdf, clipped_remain_pdf, clipped_lower_pdf], axis=0).reset_index(drop=True)
    return out_pdf


import math


def compute_clipped_scores(x_pdf, scorer_0: score.BleurtScorer,
                           scorer_1: score.BleurtScorer,
                           scorer_2: score.BleurtScorer,
                           scorer_3: score.BleurtScorer, metric="nli_bleurt", ref_col="ref", base_col="base",
                           hyp_col="hyp", id_col="count_id", method=2,
                           aggr_method="mean", aim_col="score",
                           worker_num=2, default_device='/device:GPU:1',
                           scorer_num=4):
    now_remaining_cols = list(x_pdf.columns)

    test_input_pdf = pd.DataFrame()

    group_size = math.ceil(x_pdf.shape[0] / worker_num)

    if group_size < 1:
        group_size = 1

    use_lock = threading.Lock()

    if isinstance(id_col, str):
        use_id_col = [id_col]
    else:
        use_id_col = id_col[:]

    def compute_slip_metric(input_pdf, device):
        try:
            in_pdf = input_pdf.copy()

            if isinstance(id_col, str):
                use_id_col = [id_col]
            else:
                use_id_col = id_col[:]

            # if len(use_id_col) < 2:
            #     use_id_col = id_col

            if "rouge" in metric:
                tmp_input_pdf = compute_rouge_score(x_pdf=in_pdf, ref_col=ref_col, hyp_col=base_col)
                tmp_input_pdf.rename(columns={"rouge-l_recall": "base_rouge-l_recall",
                                              "rouge-l_precision": "base_rouge-l_precision",
                                              "rouge-l": "base_rouge-l"
                                              }, inplace=True)

                tmp_input_pdf = compute_rouge_score(x_pdf=tmp_input_pdf, ref_col=ref_col, hyp_col=hyp_col)
                tmp_input_pdf[aim_col] = tmp_input_pdf["rouge-l"] / tmp_input_pdf["base_rouge-l"]

            else:
                # compute_bert_score(x_col,ref_col="answer",hyp_col="summary_turing3",out_col="bleurt_score3",device='/device:GPU:1')
                try:
                    base_metric_pdf = in_pdf.groupby(use_id_col).apply(compute_bert_score_new, scorer_0=scorer_0,
                                                                   scorer_1=scorer_1, scorer_2=scorer_2,
                                                                   scorer_3=scorer_3,
                                                                   ref_col=ref_col, hyp_col=base_col,
                                                                   out_col="base_bleurt", device=device).reset_index()
                    # print(in_pdf.columns)
                    # print(base_metric_pdf.columns)
                    # ,group_keys=False

                    test_metric_pdf = in_pdf.groupby(use_id_col).apply(compute_bert_score_new, scorer_0=scorer_0,
                                                                   scorer_1=scorer_1, scorer_2=scorer_2,
                                                                   scorer_3=scorer_3,
                                                                   ref_col=ref_col, hyp_col=hyp_col,
                                                                   out_col="test_bleurt", device=device).reset_index()

                    # print(in_pdf.columns)
                    # print(test_metric_pdf.columns)

                    tmp_input_pdf = input_pdf.copy()

                    tmp_input_pdf = pd.merge(tmp_input_pdf, base_metric_pdf[use_id_col + ["base_bleurt"]],
                                             on=use_id_col,
                                             how="inner")
                    tmp_input_pdf = pd.merge(tmp_input_pdf, test_metric_pdf[use_id_col + ["test_bleurt"]],
                                             on=use_id_col,
                                             how="inner")

                    # print(tmp_input_pdf)

                    tmp_input_pdf[aim_col] = tmp_input_pdf["test_bleurt"] / tmp_input_pdf["base_bleurt"]

                except Exception as eee:
                    print(eee)
                    print(in_pdf.shape)
                    print(in_pdf.columns)
                    print(in_pdf.head())
                    tmp_input_pdf = input_pdf.copy()


            with use_lock:
                progress_0.update(1)

        except Exception as ee0:
            print(ee0)

        return tmp_input_pdf

    computed_item_num = 0

    if worker_num < 2:
        progress_0 = tqdm(total=1, position=1, leave=True)
        test_input_pdf = compute_slip_metric(x_pdf, device=default_device)

    else:
        pool_worker_outputs = []
        device_id = 0
        progress_0 = tqdm(total=len([jjj for jjj in range(0, x_pdf.shape[0], group_size)]), position=1, leave=True)
        with ThreadPoolExecutor(max_workers=worker_num) as executor_compute:
            for j in range(0, x_pdf.shape[0], group_size):
                tmp_in_pdf = x_pdf.head(computed_item_num + group_size)
                if tmp_in_pdf.shape[0] < (computed_item_num + group_size):
                    tmp_in_pdf = tmp_in_pdf.tail(tmp_in_pdf.shape[0] - computed_item_num).reset_index(drop=True)

                else:
                    tmp_in_pdf = tmp_in_pdf.tail(group_size).reset_index(drop=True)

                computed_item_num += (tmp_in_pdf.shape[0])

                use_device_id = device_id % scorer_num
                # print(f"tmp_in_pdf:\n{tmp_in_pdf}")
                pool_worker_outputs.append(
                    executor_compute.submit(compute_slip_metric, tmp_in_pdf, f'/device:GPU:{use_device_id}'))

                device_id += 1

        progress_0.close()
        pool_worker_result = [future.result() for future in as_completed(pool_worker_outputs)]
        test_input_pdf = pd.concat(pool_worker_result, axis=0).reset_index(drop=True)

    # print(test_input_pdf.columns)
    # print(test_input_pdf.shape)

    test_input_pdf = clipped_scores(x_pdf=test_input_pdf, upper_limit=1.0, lower_limit=0.0, aim_col=aim_col)

    out_pdf = test_input_pdf[now_remaining_cols + [aim_col]].reset_index(drop=True)
    if method < 3:
        out_pdf = out_pdf.sort_values(aim_col, ascending=False).reset_index(drop=True)
    else:
        out_pdf = out_pdf.sort_values(use_id_col[0], ascending=True).reset_index(drop=True)

    return out_pdf





def compute_semantic_similarity_score(pre_data_lists, hyp_data,
                                      scorer_0: score.BleurtScorer,
                                      scorer_1: score.BleurtScorer,
                                      scorer_2: score.BleurtScorer,
                                      scorer_3: score.BleurtScorer,
                                      metric="nli_bleurt", use_relation="entailment", method=2, item_limit=5,
                                      add_semantic=True, worker_num=2, default_device='/device:GPU:1', scorer_num=4,
                                      add_structure=NO_STRUCT
                                      ):
    use_metric = metric.lower()
    if "entail" in use_relation:
        score_upper_limit = 1.0
        score_lower_limit = 0.5
    else:
        score_upper_limit = 0.5
        score_lower_limit = 0.0

    if not add_semantic:
        return score_upper_limit

    test_input_pdf = {"ref": [], "base": [], "hyp": [], "count_id": [], "sentence_id": []}
    for i in range(len(pre_data_lists)):
        test_input_pdf["ref"].append(pre_data_lists[i])
        test_input_pdf["base"].append(pre_data_lists[i])
        test_input_pdf["hyp"].append(hyp_data)
        test_input_pdf["count_id"].append(i)
        test_input_pdf["sentence_id"].append(i)

    test_input_pdf = pd.DataFrame(test_input_pdf)
    # print(test_input_pdf.shape)
    # print(test_input_pdf.columns)

    test_input_pdf = compute_clipped_scores(x_pdf=test_input_pdf, metric=metric,
                                            scorer_0=scorer_0,
                                            scorer_1=scorer_1,
                                            scorer_2=scorer_2,
                                            scorer_3=scorer_3,
                                            ref_col="ref", base_col="base",
                                            hyp_col="hyp", method=method, worker_num=worker_num,
                                            default_device=default_device,
                                            scorer_num=scorer_num)

    test_input_pdf = test_input_pdf.sort_values("score", ascending=False).reset_index(drop=True)

    # print(test_input_pdf.shape)
    # print(test_input_pdf.columns)

    # print(test_input_pdf)

    add_test_input_pdf = {"ref": [], "base": [], "hyp": [], "count_id": []}
    # pio_base_data = test_input_pdf["ref"].to_list()

    if item_limit > 0:

        for i in range(np.min([test_input_pdf.shape[0], item_limit])):
            tmp_test_input_pdf = test_input_pdf.copy().head(i + 1)
            if add_structure == NO_STRUCT:
                pio_base_data = tmp_test_input_pdf["ref"].to_list()[:]
            elif add_structure == SPLITED_STRUCT:
                tmp_test_input_pdf = tmp_test_input_pdf.sort_values("sentence_id").reset_index(drop=True)
                pio_base_data = tmp_test_input_pdf["ref"].to_list()[:]
            else:
                tmp_start_index = tmp_test_input_pdf["sentence_id"].min()
                tmp_end_index = tmp_test_input_pdf["sentence_id"].max()
                tmp_test_input_pdf = test_input_pdf[
                    (test_input_pdf["sentence_id"] >= tmp_start_index) & (
                                test_input_pdf["sentence_id"] <= tmp_end_index)].copy().reset_index(drop=True)
                tmp_test_input_pdf = tmp_test_input_pdf.sort_values("sentence_id").reset_index(drop=True)
                pio_base_data = tmp_test_input_pdf["ref"].to_list()[:]

            # tmp_test_input_pdf =
            add_test_input_pdf["ref"].append(aggr_splited_steps(input_data=pio_base_data[:]))
            add_test_input_pdf["base"].append(aggr_splited_steps(input_data=pio_base_data[:]))
            add_test_input_pdf["hyp"].append(hyp_data)
            add_test_input_pdf["count_id"].append(i)

        add_test_input_pdf = pd.DataFrame(add_test_input_pdf)

        # print(add_test_input_pdf.shape)
        # print(add_test_input_pdf.columns)
        add_test_input_pdf = compute_clipped_scores(x_pdf=add_test_input_pdf, scorer_0=scorer_0,
                                                    scorer_1=scorer_1,
                                                    scorer_2=scorer_2,
                                                    scorer_3=scorer_3,
                                                    metric=metric,
                                                    ref_col="ref", base_col="base",
                                                    hyp_col="hyp", method=method, worker_num=worker_num,
                                                    default_device=default_device, scorer_num=scorer_num)
        # print(add_test_input_pdf.shape)
        # print(add_test_input_pdf.columns)

        # print(add_test_input_pdf)

        out_score = np.max([test_input_pdf["score"].max(), add_test_input_pdf["score"].max()])
    else:
        out_score = test_input_pdf["score"].max()

    out_score = score_lower_limit + out_score * (score_upper_limit - score_lower_limit)

    return out_score


BASE_URL_GPT = ""
def compute_nli_based_score_for_one_time(pre_sentences,
                                         hyp_sentences,
                                         scorer_0: score.BleurtScorer,
                                         scorer_1: score.BleurtScorer,
                                         scorer_2: score.BleurtScorer,
                                         scorer_3: score.BleurtScorer,
                                         metric="nli_bleurt",
                                         nli_prompt=NLI_PROMPT_TEMPLATE["ENG"],
                                         pre_start_token=PRE_start_token,
                                         pre_end_token=PRE_end_token,
                                         hyp_start_token=HYP_start_token,
                                         hyp_end_token=HYP_end_token,
                                         label_start_token=NLI_start_token,
                                         label_end_token=NLI_end_token,
                                         comp_start_token=EX_COMP_start_token,
                                         comp_end_token=EX_COMP_end_token,
                                         base_url=BASE_URL_GPT,
                                         model_name="gpt-4-turbo-128k",
                                         connect_repeat_time=2,
                                         nli_repeat_time=2,
                                         split_tokens=["\n", "。"],
                                         method=0, item_limit=-1, add_semantic=True,
                                         nli_thread_num=2, computing_thread_num=2,
                                         default_device='/device:GPU:1', scorer_num=4,
                                         add_structure=NO_STRUCT
                                         ):
    hyp_sentences_list = generate_fine_grained_semantic_compoenments(input_sentence=hyp_sentences,
                                                                     split_tokens=split_tokens)

    aim_score_res = {"hyp_content": hyp_sentences_list[:],
                     "nli_label": ["Unknown"] * len(hyp_sentences_list), "find_related": [[]] * len(hyp_sentences_list)}
    aim_score_res = pd.DataFrame(aim_score_res).reset_index(drop=True)
    aim_content_extracted_relation = {}
    # ,"nli_score":[]

    # if nli_thread_num > 1:
    nli_lock = threading.Lock()

    progress = tqdm(total=len(hyp_sentences_list), position=1, leave=True)

    def call_nli_relation_execute(ind, pre_sentences, hyp_line):
        time1 = time.time()
        try:
            if method < 1:
                hsl_label = generate_nli_relation_single(premise_sentences=pre_sentences,
                                                         hyp_sentences=hyp_line,
                                                         input_prompt=nli_prompt,
                                                         pre_start_token=pre_start_token,
                                                         pre_end_token=pre_end_token,
                                                         hyp_start_token=hyp_start_token,
                                                         hyp_end_token=hyp_end_token,
                                                         label_start_token=label_start_token,
                                                         label_end_token=label_end_token,
                                                         base_url=base_url,
                                                         model_name=model_name,
                                                         connect_repeat_time=connect_repeat_time,
                                                         nli_repeat_time=nli_repeat_time)
            elif method < 2:
                hsl_label, find_extracted_results = generate_nli_relation_single(premise_sentences=pre_sentences,
                                                                                 hyp_sentences=hyp_line,
                                                                                 input_prompt=nli_prompt,
                                                                                 pre_start_token=pre_start_token,
                                                                                 pre_end_token=pre_end_token,
                                                                                 hyp_start_token=hyp_start_token,
                                                                                 hyp_end_token=hyp_end_token,
                                                                                 label_start_token=label_start_token,
                                                                                 label_end_token=label_end_token,
                                                                                 base_url=base_url,
                                                                                 model_name=model_name,
                                                                                 connect_repeat_time=connect_repeat_time,
                                                                                 nli_repeat_time=nli_repeat_time,
                                                                                 add_semantic=True)
            else:
                hsl_label = generate_nli_relation_single(premise_sentences=pre_sentences,
                                                         hyp_sentences=hyp_line,
                                                         input_prompt=nli_prompt,
                                                         pre_start_token=pre_start_token,
                                                         pre_end_token=pre_end_token,
                                                         hyp_start_token=hyp_start_token,
                                                         hyp_end_token=hyp_end_token,
                                                         label_start_token=label_start_token,
                                                         label_end_token=label_end_token,
                                                         base_url=base_url,
                                                         model_name=model_name,
                                                         connect_repeat_time=connect_repeat_time,
                                                         nli_repeat_time=nli_repeat_time)

            time2 = time.time()

            # print(f"find_label:{(time2-time1)} {hsl_label}")
            hsl_label = hsl_label.lower()
            with nli_lock:
                aim_score_res.at[ind, "hyp_content"] = hyp_line
                aim_score_res.at[ind, "nli_label"] = hsl_label
                if method > 0 and method < 2:
                    aim_score_res.at[ind, "find_related"] = find_extracted_results[:]
            # print("Done")
            progress.update(1)
        except Exception as ee0:
            print(ee0)

        return

    def find_realated_componments(ind, pre_sentences, hyp_line, hyp_label):
        if "entail" in hyp_label or "neu" in hyp_label:
            if "entail" in hyp_label or "entailment" in hyp_label:
                use_relation = "entailment"

            else:
                use_relation = "neural"
            time1 = time.time()

            find_extracted_results = extracted_related_componments_single(premise_sentences=pre_sentences,
                                                                          hyp_sentences=hyp_line,
                                                                          input_prompt=EXTRACT_RELATION_COMP_PROP[
                                                                              use_relation],
                                                                          relationship=use_relation,
                                                                          pre_start_token=pre_start_token,
                                                                          pre_end_token=pre_end_token,
                                                                          hyp_start_token=hyp_start_token,
                                                                          hyp_end_token=hyp_end_token,
                                                                          comp_start_token=comp_start_token,
                                                                          comp_end_token=comp_end_token,
                                                                          base_url=base_url,
                                                                          model_name=model_name,
                                                                          connect_repeat_time=connect_repeat_time,
                                                                          nli_repeat_time=nli_repeat_time)
            time2 = time.time()
            # print(f"find_comp:{(time2-time1)}")
            # print(f"len comp:{(len(find_extracted_results))}")

            with nli_lock:
                aim_score_res.at[ind, "find_related"] = find_extracted_results[:]
                # aim_content_extracted_relation[hyp_line] = find_extracted_results[:]

            progress.update(1)
            return
        else:
            progress.update(1)
            return

    # Stage 1: 判断NLI关系
    # aim_score_res
    with ThreadPoolExecutor(max_workers=nli_thread_num) as executor_nli:
        for ind, row in aim_score_res.iterrows():
            hsl = row["hyp_content"]
            executor_nli.submit(call_nli_relation_execute, ind, pre_sentences, hsl)
            time.sleep(1.2)

    progress.close()  # 关闭进度条
    # aim_score_res = pd.DataFrame(aim_score_res)

    # Stage 2: 寻找做出判断依据的PREMISE语句

    if method < 1:
        nli_lock = threading.Lock()
        progress = tqdm(total=len(hyp_sentences_list), position=1, leave=True)
        hsl_data = aim_score_res["hyp_content"].to_list()
        hsl_label_data = aim_score_res["nli_label"].to_list()

        with ThreadPoolExecutor(max_workers=nli_thread_num) as executor_nli:
            for ind, row in aim_score_res.iterrows():
                hsl = row["hyp_content"]
                hsl_label = row["nli_label"]
                executor_nli.submit(find_realated_componments, ind, pre_sentences, hsl, hsl_label)
                time.sleep(1.3)
        progress.close()

    elif method >= 2:
        progress = tqdm(total=len(hyp_sentences_list), position=1, leave=True)
        hsl_data = aim_score_res["hyp_content"].to_list()
        hsl_label_data = aim_score_res["nli_label"].to_list()
        for ind, row in aim_score_res.iterrows():
            if "entail" in row["nli_label"] or "neu" in row["nli_label"]:

                time1 = time.time()

                find_extracted_results = generate_fine_grained_semantic_compoenments(input_sentence=pre_sentences,
                                                                                     split_tokens=split_tokens)
                aim_score_res.at[ind, "find_related"] = find_extracted_results[:]

                time2 = time.time()
                # print(f"find_comp:{(time2-time1)}")
                # print(f"len comp:{(len(find_extracted_results))}")
            else:
                continue

            progress.update(1)

        progress.close()

    for h_id in range(aim_score_res.shape[0]):
        if aim_score_res["find_related"].values[h_id]:
            aim_content_extracted_relation[aim_score_res["hyp_content"].values[h_id]] = \
            aim_score_res["find_related"].values[h_id][:]

    # print(len(list(aim_content_extracted_relation.keys())))

    aim_score_res = aim_score_res.reset_index(drop=True)
    aim_score_res["sentence_id"] = pd.Series([ii for ii in range(aim_score_res.shape[0])])

    # print(aim_score_res)

    # Stage 3 进行相似性打分的计算

    nli_scores = {"hyp_content": [], "nli_score": []}

    aim_to_compute_hyps = []
    for jj in range(aim_score_res.shape[0]):
        hsl = aim_score_res["hyp_content"].values[jj]
        hsl_label = aim_score_res["nli_label"].values[jj]

        if "contradiction" in hsl_label or "cond" in hsl_label:

            nli_scores["hyp_content"].append(hsl)
            nli_scores["nli_score"].append(0.0)
        elif hsl in list(aim_content_extracted_relation.keys()):
            if len(aim_content_extracted_relation[hsl]) > 0:
                aim_to_compute_hyps.append(hsl)

            else:
                nli_scores["hyp_content"].append(hsl)
                if "entail" in hsl_label:
                    nli_scores["nli_score"].append(0.5)
                else:
                    nli_scores["nli_score"].append(0.25)

        else:
            nli_scores["hyp_content"].append(hsl)
            nli_scores["nli_score"].append(0.25)

    nli_scores = pd.DataFrame(nli_scores)

    aim_to_compute_hyp_res = aim_score_res[aim_score_res.hyp_content.isin(aim_to_compute_hyps)].reset_index(drop=True)
    computed_score_res = aim_score_res[~aim_score_res.hyp_content.isin(aim_to_compute_hyps)].reset_index(drop=True)
    if computed_score_res.shape[0] > 0:
        # aim_to_compute_hyp_res["nli_score"]

        computed_score_res = pd.merge(computed_score_res, nli_scores, on="hyp_content", how="inner").reset_index(
            drop=True)

    aim_to_compute_hyp_res_out = {"hyp_content": [], "nli_score": []}

    for idx in range(len(aim_to_compute_hyps)):
        hsl = aim_to_compute_hyps[idx]
        if item_limit < 0:
            input_item_limit = int(len(aim_content_extracted_relation[hsl]) // 2)
        else:
            input_item_limit = np.min([int(len(aim_content_extracted_relation[hsl])), item_limit])

        aim_to_compute_hyp_res_out["hyp_content"].append(aim_to_compute_hyps[idx])
        time1 = time.time()

        tmp_label = aim_to_compute_hyp_res[aim_to_compute_hyp_res.hyp_content.isin([hsl])].values[0]

        if "entail" in tmp_label or "entailment" in tmp_label:
            use_relation = "entailment"

        else:
            use_relation = "neural"

        computed_score = compute_semantic_similarity_score(pre_data_lists=aim_content_extracted_relation[hsl],
                                                           hyp_data=hsl, scorer_0=scorer_0, scorer_1=scorer_1,
                                                           scorer_2=scorer_2, scorer_3=scorer_3,
                                                           metric=metric, use_relation=use_relation,
                                                           item_limit=input_item_limit,
                                                           add_semantic=add_semantic,
                                                           worker_num=computing_thread_num,
                                                           default_device=default_device,
                                                           scorer_num=scorer_num,
                                                           add_structure=add_structure
                                                           )

        aim_to_compute_hyp_res_out["nli_score"].append(computed_score)
        time2 = time.time()
        # print(f"compute score:{(time2-time1)}")

    aim_to_compute_hyp_res_out = pd.DataFrame(aim_to_compute_hyp_res_out)
    if aim_to_compute_hyp_res_out.shape[0] > 0 and aim_to_compute_hyp_res.shape[0] > 0:
        aim_to_compute_hyp_res = pd.merge(aim_to_compute_hyp_res, aim_to_compute_hyp_res_out, on="hyp_content",
                                          how="inner").reset_index(drop=True)

    if computed_score_res.shape[0] > 0 and aim_to_compute_hyp_res.shape[0] > 0:
        aim_score_res = pd.concat([computed_score_res, aim_to_compute_hyp_res], axis=0).reset_index(drop=True)

    elif computed_score_res.shape[0] > 0:
        aim_score_res = computed_score_res.reset_index(drop=True)

    elif aim_to_compute_hyp_res.shape[0] > 0:
        aim_score_res = aim_to_compute_hyp_res.reset_index(drop=True)

    else:
        aim_score_res = aim_to_compute_hyp_res.reset_index(drop=True)

    aim_score_res = aim_score_res.sort_values("sentence_id").reset_index(drop=True)

    return aim_score_res


def compute_nli_based_score_for_batch(x_pdf, pre_col,
                                      hyp_col,
                                      scorer_0: score.BleurtScorer,
                                      scorer_1: score.BleurtScorer,
                                      scorer_2: score.BleurtScorer,
                                      scorer_3: score.BleurtScorer,
                                      metric="nli_bleurt",
                                      group_key_cols=["generate_question", "step_id", "line_id", "sen_id"],
                                      nli_prompt=NLI_PROMPT_TEMPLATE["ENG"],
                                      pre_start_token=PRE_start_token,
                                      pre_end_token=PRE_end_token,
                                      hyp_start_token=HYP_start_token,
                                      hyp_end_token=HYP_end_token,
                                      label_start_token=NLI_start_token,
                                      label_end_token=NLI_end_token,
                                      comp_start_token=EX_COMP_start_token,
                                      comp_end_token=EX_COMP_end_token,
                                      base_url=BASE_URL_GPT,
                                      model_name="gpt-4-turbo-128k",
                                      connect_repeat_time=2,
                                      nli_repeat_time=2,
                                      split_tokens=["\n", "。"],
                                      method=0, item_limit=-1, add_semantic=True, add_structure=NO_STRUCT,
                                      aim_col="faith_score", nli_thread_num=2,
                                      computing_thread_num=2, default_device='/device:GPU:1', scorer_num=4):
    in_pdf = x_pdf.copy()
    # "entail":[extract_distribution["entailment"]],
    #    "contrad":[extract_distribution["contradiction"]],
    #    "neural":[extract_distribution["neural"]],
    #    score_col:[extract_distribution["entailment"]]
    # "entail_out":[],"contrad_out":[],"neural_out":[]

    if isinstance(group_key_cols, str):
        group_key_cols = [group_key_cols]

    if add_structure == TOTAL_STRUCT:
        use_add_semantic = False
    else:
        use_add_semantic = add_semantic

    if not use_add_semantic:
        out_res = {aim_col: [], "nli_score": [], "entail_out": [], "contrad_out": [], "neural_out": []}
    else:
        out_res = {aim_col: [], "nli_score": []}

    for gkc in group_key_cols:
        out_res[gkc] = []

    for jj in range(in_pdf.shape[0]):
        tmp_premise = in_pdf[pre_col].values[jj]
        tmp_hyp = in_pdf[hyp_col].values[jj]

        # NO_STRUCT = 0
        # SPLITED_STRUCT = 1
        # CON_PART_STRUCT = 2
        # TOTAL_STRUCT = 3

        tmp_out_res_pdf = compute_nli_based_score_for_one_time(pre_sentences=tmp_premise,
                                                               hyp_sentences=tmp_hyp,
                                                               scorer_0=scorer_0,
                                                               scorer_1=scorer_1,
                                                               scorer_2=scorer_2,
                                                               scorer_3=scorer_3,
                                                               metric=metric,
                                                               nli_prompt=nli_prompt,
                                                               pre_start_token=pre_start_token,
                                                               pre_end_token=pre_end_token,
                                                               hyp_start_token=hyp_start_token,
                                                               hyp_end_token=hyp_end_token,
                                                               label_start_token=label_start_token,
                                                               label_end_token=label_end_token,
                                                               comp_start_token=comp_start_token,
                                                               comp_end_token=comp_end_token,
                                                               base_url=base_url,
                                                               model_name=model_name,
                                                               connect_repeat_time=connect_repeat_time,
                                                               nli_repeat_time=nli_repeat_time,
                                                               split_tokens=split_tokens,
                                                               method=method, item_limit=item_limit,
                                                               add_semantic=use_add_semantic,
                                                               nli_thread_num=nli_thread_num,
                                                               computing_thread_num=computing_thread_num,
                                                               default_device=default_device, scorer_num=scorer_num,
                                                               add_structure=add_structure)

        if use_add_semantic:
            out_res[aim_col].append(tmp_out_res_pdf["nli_score"].mean())
            out_res["nli_score"].append(tmp_out_res_pdf["nli_score"].mean())
        else:
            tmp_nli_res = nli_distribution_aggr(x_pdf=tmp_out_res_pdf, label_col="nli_label", score_col=aim_col)
            out_res[aim_col].append(tmp_nli_res[aim_col].mean())
            out_res["nli_score"].append(tmp_nli_res[aim_col].mean())
            out_res["entail_out"].append(tmp_nli_res["entail"].mean())
            out_res["contrad_out"].append(tmp_nli_res["contrad"].mean())
            out_res["neural_out"].append(tmp_nli_res["neural"].mean())

        for gkc in group_key_cols:
            out_res[gkc].append(in_pdf[gkc].values[jj])

    result_pdf = pd.DataFrame(out_res).reset_index(drop=True)

    if add_structure == TOTAL_STRUCT:
        tmp_struct_metric = compute_clipped_scores(x_pdf=in_pdf.copy(),
                                                   scorer_0=scorer_0,
                                                   scorer_1=scorer_1,
                                                   scorer_2=scorer_2,
                                                   scorer_3=scorer_3,
                                                   metric=metric, ref_col=pre_col,
                                                   base_col=pre_col, hyp_col=hyp_col, id_col=group_key_cols,
                                                   method=2, aggr_method="mean", aim_col="struct_score",
                                                   worker_num=computing_thread_num,
                                                   default_device=default_device, scorer_num=scorer_num)

        result_pdf = pd.merge(result_pdf, tmp_struct_metric[group_key_cols + ["struct_score"]],
                              on=group_key_cols, how="inner").reset_index(drop=True)
        print(result_pdf.columns)
        print(result_pdf.shape)

        result_pdf[aim_col] = result_pdf[aim_col] * result_pdf["struct_score"]

    return result_pdf.reset_index(drop=True)