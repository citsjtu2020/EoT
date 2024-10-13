import pandas as pd
from prompt_design import REPLAY_META_BODY,PRE_end_token,PRE_start_token,NLI_end_token,NLI_start_token,HYP_end_token,HYP_start_token,NLI_PROMPT_TEMPLATE
from NLI_socre_compute import generate_NLI_results_set,nli_distribution_aggr
from replay_generate import generate_replay_answer_set
from prompt_design import NO_STRUCT,CON_PART_STRUCT,TOTAL_STRUCT
from Faithfulity_score_compute import compute_clipped_scores,compute_nli_based_score_for_batch
import time
from bleurt import score

# functions to estimate scoring metrics of factulity, fidelity  and reliability

BASE_URL=""


def scoring_factuality_score(input_pdf, question_col="generate_question", reference_col="reference_body2",
                             input_prompt=NLI_PROMPT_TEMPLATE["ENG"],
                             out_col="nli_label",
                             reasoning_content_col="content",
                             pre_start_token=PRE_start_token,
                             pre_end_token=PRE_end_token,
                             hyp_start_token=HYP_start_token,
                             hyp_end_token=HYP_end_token,
                             label_start_token=NLI_start_token,
                             label_end_token=NLI_end_token,
                             base_url=BASE_URL,
                             model_name="gpt-4-turbo-128k",
                             metric="nli",
                             score_col="fact_score",
                             group_key='step_id',
                             connect_repeat_time=2, nli_repeat_time=2, add_semantic=False, nli_thread_num=3):
    in_pdf = input_pdf.copy()
    metric = metric.lower()

    nli_pdf = generate_NLI_results_set(in_pdf=in_pdf,
                                       question_col=question_col, reference_col=reference_col,
                                       input_prompt=input_prompt,
                                       out_col=out_col,
                                       reasoning_content_col=reasoning_content_col,
                                       pre_start_token=pre_start_token,
                                       pre_end_token=pre_end_token,
                                       hyp_start_token=hyp_start_token,
                                       hyp_end_token=hyp_end_token,
                                       label_start_token=label_start_token,
                                       label_end_token=label_end_token,
                                       base_url=base_url,
                                       model_name=model_name,
                                       connect_repeat_time=connect_repeat_time, nli_repeat_time=nli_repeat_time,
                                       nli_thread_num=nli_thread_num)

    now_nli_cols = list(nli_pdf.columns)
    print(now_nli_cols)
    print(nli_pdf.shape)
    if add_semantic:
        return nli_pdf
    else:
        in_nli_pdf = nli_pdf.copy()
        remaining_cols = list(set(now_nli_cols) - set(["score", "entail", "contrad", "neural"]))
        if isinstance(group_key, list):
            group_key_list = group_key[:]
        else:
            group_key_list = [group_key]
        nil_dist_pdf = in_nli_pdf[remaining_cols].groupby([question_col] + group_key_list).apply(
            nli_distribution_aggr, label_col="nli_label", score_col=score_col).reset_index()
        print(nil_dist_pdf.shape)
        return nil_dist_pdf, nli_pdf


TOP_START = "**"
TOP_END = "**"
MASK = 0
MODIFY = 1
HOLD = 2

IGNORE = 0
TURTH_WEIGHT = 1
GROUND_WEIGHT = 2
GROUND_NO_WEIGHT = 3


def scoring_faithfulity_score_v2(input_pdf,
                                 scorer_0: score.BleurtScorer,
                                 scorer_1: score.BleurtScorer,
                                 scorer_2: score.BleurtScorer,
                                 scorer_3: score.BleurtScorer,
                                 prompt_body=REPLAY_META_BODY, question_col="generate_question",
                                 refer_col="reference_body2",
                                 raw_answer_col="manual_answer",
                                 refer_answer_col="hist_replay_answer",
                                 reasoning_col="reasoning_input",
                                 replay_col="replay_answer",
                                 eval_model="chat_qwen72B_pre",
                                 metric="bleurt",
                                 hist_reasoning_col="hist_reasoning_input",
                                 bert_model="bert-base-chinese", aim_col="faith_score", truth_col="truth_score",
                                 nli_prompt=NLI_PROMPT_TEMPLATE["ENG"],
                                 base_url=BASE_URL,
                                 model_name="gpt-4-turbo-128k",
                                 connect_repeat_time=2,
                                 nli_repeat_time=2,
                                 split_tokens=["\n", "。"],
                                 method=2, item_limit=100, add_semantic=True,
                                 nli_thread_num=2,
                                 computing_thread_num=2,
                                 default_device='/device:GPU:1',
                                 scorer_num=4, replay_thread_num=2, is_truth=IGNORE, repeat_time=2,
                                 add_structure=NO_STRUCT
                                 ):
    in_pdf = input_pdf.copy()
    metric = metric.lower()

    if is_truth >= 2:
        is_truth = GROUND_WEIGHT
    elif is_truth <= 0:
        is_truth = IGNORE
    else:
        is_truth = TURTH_WEIGHT

    if repeat_time < 1:
        repeat_time = 1

    repeat_time = int(repeat_time)

    # raw_answer_col=raw_answer_col,

    replay_pdf = generate_replay_answer_set(
        in_pdf=in_pdf,
        prompt_body=REPLAY_META_BODY, question_col=question_col,
        refer_col=refer_col,
        reasoning_col=reasoning_col,
        out_col=replay_col, eval_model=eval_model,
        replay_thread_num=replay_thread_num,

    )

    now_replay_cols = list(replay_pdf.columns)
    print(now_replay_cols)
    print(replay_pdf.shape)

    def catch_history_replay_answer_for_qes(x_pdf, hist_replay_col, raw_ans_col, hist_reasoning_col):
        in_x_pdf = x_pdf.copy()
        out_pdf = {hist_reasoning_col: [in_x_pdf[hist_reasoning_col].values[0]],
                   hist_replay_col: [in_x_pdf[hist_replay_col].values[0]],
                   raw_ans_col: [in_x_pdf[raw_ans_col].values[0]]}

        return pd.DataFrame(out_pdf)

    # if is_truth != IGNORE:
    hist_reasoning_pdf = replay_pdf.groupby(question_col).apply(catch_history_replay_answer_for_qes,
                                                                hist_replay_col=refer_answer_col,
                                                                raw_ans_col=raw_answer_col,
                                                                hist_reasoning_col=hist_reasoning_col).reset_index()

    if "nli" not in metric:
        in_replay_pdf = replay_pdf.copy()
        merge_add_col = [question_col, "step_id", "line_id", "sen_id", aim_col, truth_col]
        if is_truth == GROUND_WEIGHT:
            merge_add_col.append(f"hist_{aim_col}")

        if "rouge" in metric:
            input_metric = "rouge"
            remaining_cols = list(
                set(now_replay_cols) - set(["rouge-l", "rouge-l_precision", "rouge-l_recall", "score", aim_col]))
        else:
            input_metric = "bleurt"
            remaining_cols = list(set(now_replay_cols) - set(["score", "bleurt", aim_col]))

            # if is_truth != IGNORE:
        hist_metric_pdf = pd.DataFrame()
        out_metric_pdf = pd.DataFrame()
        for rp_id in range(repeat_time):
            tmp_hist_metric_pdf = compute_clipped_scores(x_pdf=hist_reasoning_pdf,
                                                         scorer_0=scorer_0,
                                                         scorer_1=scorer_1,
                                                         scorer_2=scorer_2,
                                                         scorer_3=scorer_3,
                                                         metric=input_metric,
                                                         ref_col=raw_answer_col,
                                                         base_col=raw_answer_col,
                                                         hyp_col=refer_answer_col,
                                                         id_col=[question_col],
                                                         method=2, aggr_method="mean",
                                                         aim_col=f"{truth_col}_{rp_id}",
                                                         worker_num=computing_thread_num,
                                                         default_device=default_device,
                                                         scorer_num=scorer_num)

            if is_truth != GROUND_WEIGHT:
                tmp_out_metric_pdf = compute_clipped_scores(x_pdf=in_replay_pdf[remaining_cols],
                                                            scorer_0=scorer_0, scorer_1=scorer_1,
                                                            scorer_2=scorer_2, scorer_3=scorer_3,
                                                            metric=input_metric,
                                                            ref_col=refer_answer_col,
                                                            base_col=refer_answer_col,
                                                            hyp_col=replay_col,
                                                            id_col=[question_col, "step_id", "line_id", "sen_id"],
                                                            method=2, aggr_method="mean",
                                                            aim_col=f"{aim_col}_{rp_id}",
                                                            worker_num=computing_thread_num,
                                                            default_device=default_device,
                                                            scorer_num=scorer_num)

            else:
                tmp_out_metric_pdf = compute_clipped_scores(x_pdf=in_replay_pdf[remaining_cols],
                                                            scorer_0=scorer_0, scorer_1=scorer_1,
                                                            scorer_2=scorer_2, scorer_3=scorer_3,
                                                            metric=input_metric,
                                                            ref_col=raw_answer_col,
                                                            base_col=raw_answer_col,
                                                            hyp_col=replay_col,
                                                            id_col=[question_col, "step_id", "line_id", "sen_id"],
                                                            method=2, aggr_method="mean",
                                                            aim_col=f"{aim_col}_{rp_id}",
                                                            worker_num=computing_thread_num,
                                                            default_device=default_device,
                                                            scorer_num=scorer_num)

            if rp_id == 0:
                out_metric_pdf = tmp_out_metric_pdf[
                    [question_col, "step_id", "line_id", "sen_id", f"{aim_col}_{rp_id}"]].copy()
                hist_metric_pdf = tmp_hist_metric_pdf[[question_col, f"{truth_col}_{rp_id}"]].copy()
            else:
                out_metric_pdf = pd.merge(out_metric_pdf,
                                          tmp_out_metric_pdf[
                                              [question_col, "step_id", "line_id", "sen_id", f"{aim_col}_{rp_id}"]],
                                          on=[question_col, "step_id", "line_id", "sen_id"], how="inner").reset_index(
                    drop=True)
                hist_metric_pdf = pd.merge(hist_metric_pdf,
                                           tmp_hist_metric_pdf[[question_col, f"{truth_col}_{rp_id}"]],
                                           on=[question_col], how="inner").reset_index(drop=True)
            time.sleep(1.2)

        hist_metric_pdf = hist_metric_pdf.reset_index(drop=True)
        out_metric_pdf = out_metric_pdf.reset_index(drop=True)
        hist_metric_pdf[truth_col] = 0.0
        out_metric_pdf[aim_col] = 0.0

        for rp_id in range(repeat_time):
            hist_metric_pdf[truth_col] = hist_metric_pdf[truth_col] + hist_metric_pdf[f"{truth_col}_{rp_id}"]
            out_metric_pdf[aim_col] = out_metric_pdf[aim_col] + out_metric_pdf[f"{aim_col}_{rp_id}"]

        hist_metric_pdf[truth_col] = hist_metric_pdf[truth_col] / repeat_time
        out_metric_pdf[aim_col] = out_metric_pdf[aim_col] / repeat_time

        out_metric_pdf = pd.merge(out_metric_pdf, hist_metric_pdf[[question_col, truth_col]], on=[question_col],
                                  how="inner").reset_index(drop=True)
        out_metric_pdf[f"hist_{aim_col}"] = 1.0 - out_metric_pdf[truth_col]

        # rouge_metric_pdf = compute_rouge_score(x_pdf=in_replay_pdf[remaining_cols],
        #                 ref_col=raw_answer_col,
        #                 hyp_col=replay_col)

        replay_pdf = pd.merge(replay_pdf[remaining_cols], out_metric_pdf[merge_add_col],
                              on=[question_col, "step_id", "line_id", "sen_id"], how="inner")

        replay_pdf = replay_pdf.reset_index(drop=True)

        replay_pdf[aim_col] = 1.0 - replay_pdf[aim_col]

        replay_pdf[f"{aim_col}_raw"] = replay_pdf[aim_col].copy()

        if is_truth != IGNORE:
            print("consider truth weight")
            for ii in replay_pdf.index:
                replay_pdf[aim_col].at[ii] = replay_pdf[aim_col].at[ii] * replay_pdf[truth_col].at[ii]

        # elif is_truth == GROUND_WEIGHT:
        #     replay_pdf[aim_col] = replay_pdf[aim_col] - replay_pdf
        # replay_pdf[aim_col] = (((replay_pdf[aim_col] - replay_pdf[f"hist_{aim_col}"])/replay_pdf[f"hist_{aim_col}"])-(
        #     (0 - replay_pdf[f"hist_{aim_col}"])/replay_pdf[f"hist_{aim_col}"])) / (((1-replay_pdf[f"hist_{aim_col}"])/replay_pdf[f"hist_{aim_col}"])-(
        #         (0-replay_pdf[f"hist_{aim_col}"])/replay_pdf[f"hist_{aim_col}"]))
        # replay_pdf[aim_col] = replay_col[aim_col]*replay_col[truth_col]

        print(replay_pdf.shape)
        return replay_pdf

    else:
        in_replay_pdf = replay_pdf.copy()
        remaining_cols = list(set(now_replay_cols) - set(["nli_score", aim_col]))
        in_replay_pdf = in_replay_pdf[remaining_cols].reset_index(drop=True)

        merge_add_col = [question_col, "step_id", "line_id", "sen_id", aim_col, truth_col]
        if is_truth == GROUND_WEIGHT:
            merge_add_col.append(f"hist_{aim_col}")

        hist_metric_pdf = pd.DataFrame()
        out_metric_pdf = pd.DataFrame()

        for rp_id in range(repeat_time):

            tmp_hist_metric_pdf = compute_nli_based_score_for_batch(x_pdf=hist_reasoning_pdf,
                                                                    pre_col=raw_answer_col,
                                                                    hyp_col=refer_answer_col,
                                                                    scorer_0=scorer_0,
                                                                    scorer_1=scorer_1,
                                                                    scorer_2=scorer_2,
                                                                    scorer_3=scorer_3,
                                                                    metric=metric, group_key_cols=[question_col],
                                                                    nli_prompt=nli_prompt,
                                                                    base_url=base_url,
                                                                    model_name=model_name,
                                                                    connect_repeat_time=connect_repeat_time,
                                                                    nli_repeat_time=nli_repeat_time,
                                                                    split_tokens=split_tokens,
                                                                    method=method, item_limit=item_limit,
                                                                    add_semantic=add_semantic,
                                                                    aim_col=f"{truth_col}_{rp_id}",
                                                                    nli_thread_num=nli_thread_num,
                                                                    computing_thread_num=computing_thread_num,
                                                                    default_device=default_device,
                                                                    scorer_num=scorer_num,
                                                                    add_structure=add_structure
                                                                    ).reset_index(drop=True)

            if is_truth != GROUND_WEIGHT:
                tmp_out_metric_pdf = compute_nli_based_score_for_batch(x_pdf=in_replay_pdf,
                                                                       pre_col=refer_answer_col,
                                                                       hyp_col=replay_col,
                                                                       scorer_0=scorer_0,
                                                                       scorer_1=scorer_1,
                                                                       scorer_2=scorer_2,
                                                                       scorer_3=scorer_3,
                                                                       metric=metric,
                                                                       group_key_cols=[question_col, "step_id",
                                                                                       "line_id", "sen_id"],
                                                                       nli_prompt=nli_prompt,
                                                                       base_url=base_url,
                                                                       model_name=model_name,
                                                                       connect_repeat_time=connect_repeat_time,
                                                                       nli_repeat_time=nli_repeat_time,
                                                                       split_tokens=split_tokens,
                                                                       method=method, item_limit=item_limit,
                                                                       add_semantic=add_semantic,
                                                                       aim_col=f"{aim_col}_{rp_id}",
                                                                       nli_thread_num=nli_thread_num,
                                                                       computing_thread_num=computing_thread_num,
                                                                       default_device=default_device,
                                                                       scorer_num=scorer_num,
                                                                       add_structure=add_structure
                                                                       ).reset_index(drop=True)

            else:
                tmp_out_metric_pdf = compute_nli_based_score_for_batch(x_pdf=in_replay_pdf,
                                                                       pre_col=raw_answer_col,
                                                                       hyp_col=replay_col,
                                                                       scorer_0=scorer_0,
                                                                       scorer_1=scorer_1,
                                                                       scorer_2=scorer_2,
                                                                       scorer_3=scorer_3,
                                                                       metric=metric,
                                                                       group_key_cols=[question_col, "step_id",
                                                                                       "line_id", "sen_id"],
                                                                       nli_prompt=nli_prompt,
                                                                       base_url=base_url,
                                                                       model_name=model_name,
                                                                       connect_repeat_time=connect_repeat_time,
                                                                       nli_repeat_time=nli_repeat_time,
                                                                       split_tokens=split_tokens,
                                                                       method=method, item_limit=item_limit,
                                                                       add_semantic=add_semantic,
                                                                       aim_col=f"{aim_col}_{rp_id}",
                                                                       nli_thread_num=nli_thread_num,
                                                                       computing_thread_num=computing_thread_num,
                                                                       default_device=default_device,
                                                                       scorer_num=scorer_num,
                                                                       add_structure=add_structure
                                                                       ).reset_index(drop=True)

            if rp_id == 0:
                out_metric_pdf = tmp_out_metric_pdf[
                    [question_col, "step_id", "line_id", "sen_id", f"{aim_col}_{rp_id}"]].copy()
                hist_metric_pdf = tmp_hist_metric_pdf[[question_col, f"{truth_col}_{rp_id}"]].copy()
            else:
                out_metric_pdf = pd.merge(out_metric_pdf,
                                          tmp_out_metric_pdf[
                                              [question_col, "step_id", "line_id", "sen_id", f"{aim_col}_{rp_id}"]],
                                          on=[question_col, "step_id", "line_id", "sen_id"], how="inner").reset_index(
                    drop=True)
                hist_metric_pdf = pd.merge(hist_metric_pdf,
                                           tmp_hist_metric_pdf[[question_col, f"{truth_col}_{rp_id}"]],
                                           on=[question_col], how="inner").reset_index(drop=True)

            time.sleep(1.2)

        hist_metric_pdf = hist_metric_pdf.reset_index(drop=True)
        out_metric_pdf = out_metric_pdf.reset_index(drop=True)
        hist_metric_pdf[truth_col] = 0.0
        out_metric_pdf[aim_col] = 0.0

        for rp_id in range(repeat_time):
            hist_metric_pdf[truth_col] = hist_metric_pdf[truth_col] + hist_metric_pdf[f"{truth_col}_{rp_id}"]
            out_metric_pdf[aim_col] = out_metric_pdf[aim_col] + out_metric_pdf[f"{aim_col}_{rp_id}"]

        hist_metric_pdf[truth_col] = hist_metric_pdf[truth_col] / repeat_time
        out_metric_pdf[aim_col] = out_metric_pdf[aim_col] / repeat_time

        out_metric_pdf = pd.merge(out_metric_pdf, hist_metric_pdf[[question_col, truth_col]], on=[question_col],
                                  how="inner").reset_index(drop=True)
        out_metric_pdf[f"hist_{aim_col}"] = 1.0 - out_metric_pdf[truth_col]

        replay_pdf = pd.merge(replay_pdf[remaining_cols], out_metric_pdf[merge_add_col],
                              on=[question_col, "step_id", "line_id", "sen_id"], how="inner")

        replay_pdf = replay_pdf.reset_index(drop=True)

        replay_pdf[aim_col] = 1.0 - replay_pdf[aim_col]
        replay_pdf[f"{aim_col}_raw"] = replay_pdf[aim_col].copy()
        # print(replay_pdf)

        if is_truth != IGNORE:
            print("consider truth weight")
            print(replay_pdf.index)
            print(aim_col)
            print(truth_col)
            for ii in replay_pdf.index:
                # print(ii)
                replay_pdf[aim_col].at[ii] = replay_pdf[aim_col].at[ii] * replay_pdf[truth_col].at[ii]

        # elif is_truth == GROUND_WEIGHT:
        #     replay_pdf[aim_col] = (((replay_pdf[aim_col] - replay_pdf[f"hist_{aim_col}"])/replay_pdf[f"hist_{aim_col}"])-(
        #         (0 - replay_pdf[f"hist_{aim_col}"])/replay_pdf[f"hist_{aim_col}"])) / (((1-replay_pdf[f"hist_{aim_col}"])/replay_pdf[f"hist_{aim_col}"])-(
        #             (0-replay_pdf[f"hist_{aim_col}"])/replay_pdf[f"hist_{aim_col}"]))
        #     replay_pdf[aim_col] = replay_col[aim_col]*replay_col[truth_col]

        print(replay_pdf.shape)

        return replay_pdf


def scoring_faithfulity_score_v3(input_pdf,
                                 prompt_body=REPLAY_META_BODY, question_col="generate_question",
                                 refer_col="reference_body2",
                                 raw_answer_col="manual_answer",
                                 refer_answer_col="hist_replay_answer",
                                 reasoning_col="reasoning_input",
                                 replay_col="replay_answer",
                                 eval_model="chat_qwen72B_pre",
                                 metric="bleurt",
                                 hist_reasoning_col="hist_reasoning_input",
                                 bert_model="bert-base-chinese", aim_col="faith_score", truth_col="truth_score",
                                 nli_prompt=NLI_PROMPT_TEMPLATE["ENG"],
                                 base_url=BASE_URL,
                                 model_name="gpt-4-turbo-128k",
                                 connect_repeat_time=2,
                                 nli_repeat_time=2,
                                 split_tokens=["\n", "。"],
                                 method=2, item_limit=100, add_semantic=True,
                                 nli_thread_num=2,
                                 computing_thread_num=2,
                                 default_device='/device:GPU:1',
                                 scorer_num=4, replay_thread_num=2,
                                 is_truth=IGNORE, repeat_time=2, add_structure=NO_STRUCT,
                                 is_super=True, is_replay=True,
                                 aggr_method=""
                                 ):
    in_pdf = input_pdf.copy()
    metric = metric.lower()

    if is_truth >= 3:
        is_truth = GROUND_NO_WEIGHT
    elif is_truth >= 2:
        is_truth = GROUND_WEIGHT
    elif is_truth <= 0:
        is_truth = IGNORE
    else:
        is_truth = TURTH_WEIGHT

    if repeat_time < 1:
        repeat_time = 1

    repeat_time = int(repeat_time)

    # raw_answer_col=raw_answer_col,

    if is_super:
        replay_pdf = generate_replay_answer_set(
            in_pdf=in_pdf,
            prompt_body=REPLAY_META_BODY,
            refer_col=refer_col,
            reasoning_col=reasoning_col,question_col=question_col,
            out_col=replay_col, eval_model=eval_model,
            replay_thread_num=replay_thread_num,
        )
        print("supervised replay")
    else:
        if is_replay:
            replay_pdf = in_pdf.copy()
            replay_pdf[replay_col] = in_pdf[refer_answer_col].copy()
            print("unsupervised replay")

        else:
            replay_pdf = in_pdf.copy()
            replay_pdf[replay_col] = in_pdf[refer_answer_col].copy()
            print("supervised unreplay")

    now_replay_cols = list(replay_pdf.columns)
    print(now_replay_cols)
    print(replay_pdf.shape)

    def catch_history_replay_answer_for_qes(x_pdf, hist_replay_col, raw_ans_col, hist_reasoning_col):
        in_x_pdf = x_pdf.copy()
        out_pdf = {hist_reasoning_col: [in_x_pdf[hist_reasoning_col].values[0]],
                   hist_replay_col: [in_x_pdf[hist_replay_col].values[0]],
                   raw_ans_col: [in_x_pdf[raw_ans_col].values[0]]}

        return pd.DataFrame(out_pdf)

    # if is_truth != IGNORE:
    hist_reasoning_pdf = replay_pdf.groupby(question_col).apply(catch_history_replay_answer_for_qes,
                                                                hist_replay_col=refer_answer_col,
                                                                raw_ans_col=raw_answer_col,
                                                                hist_reasoning_col=hist_reasoning_col).reset_index()

    if "nli" not in metric:
        in_replay_pdf = replay_pdf.copy()
        merge_add_col = [question_col, "step_id", "line_id", "sen_id", aim_col, truth_col]
        if is_truth > TURTH_WEIGHT:
            merge_add_col.append(f"hist_{aim_col}")

        if "rouge" in metric:
            input_metric = "rouge"
            remaining_cols = list(
                set(now_replay_cols) - set(["rouge-l", "rouge-l_precision", "rouge-l_recall", "score", aim_col]))
        else:
            input_metric = "bleurt"
            remaining_cols = list(set(now_replay_cols) - set(["score", "bleurt", aim_col]))

            # if is_truth != IGNORE:
        hist_metric_pdf = pd.DataFrame()
        out_metric_pdf = pd.DataFrame()
        for rp_id in range(repeat_time):
            print("hist: raw answer with refer answer")

            tmp_hist_metric_pdf = compute_clipped_scores(x_pdf=hist_reasoning_pdf,
                                                         metric=input_metric, ref_col=raw_answer_col,
                                                         base_col=raw_answer_col,
                                                         hyp_col=refer_answer_col,
                                                         id_col=[question_col],
                                                         method=2, aggr_method="mean", aim_col=f"{truth_col}_{rp_id}",
                                                         worker_num=computing_thread_num,
                                                         default_device=default_device, scorer_num=scorer_num)

            if is_truth < GROUND_WEIGHT:
                print("now: reasoning with replay answer")
                tmp_out_metric_pdf = compute_clipped_scores(x_pdf=in_replay_pdf[remaining_cols],
                                                            metric=input_metric, ref_col=reasoning_col,
                                                            base_col=reasoning_col,
                                                            hyp_col=replay_col,
                                                            id_col=[question_col, "step_id", "line_id", "sen_id"],
                                                            method=2, aggr_method="mean", aim_col=f"{aim_col}_{rp_id}",
                                                            worker_num=computing_thread_num,
                                                            default_device=default_device, scorer_num=scorer_num)

            else:
                print("compute diff with groundtruth")
                print("now: reasoning with raw answer")
                tmp_out_metric_pdf = compute_clipped_scores(x_pdf=in_replay_pdf[remaining_cols],
                                                            metric=input_metric, ref_col=reasoning_col,
                                                            base_col=reasoning_col,
                                                            hyp_col=raw_answer_col,
                                                            id_col=[question_col, "step_id", "line_id", "sen_id"],
                                                            method=2, aggr_method="mean", aim_col=f"{aim_col}_{rp_id}",
                                                            worker_num=computing_thread_num,
                                                            default_device=default_device, scorer_num=scorer_num)

            if rp_id == 0:
                out_metric_pdf = tmp_out_metric_pdf[
                    [question_col, "step_id", "line_id", "sen_id", f"{aim_col}_{rp_id}"]].copy()
                hist_metric_pdf = tmp_hist_metric_pdf[[question_col, f"{truth_col}_{rp_id}"]].copy()
            else:
                out_metric_pdf = pd.merge(out_metric_pdf,
                                          tmp_out_metric_pdf[
                                              [question_col, "step_id", "line_id", "sen_id", f"{aim_col}_{rp_id}"]],
                                          on=[question_col, "step_id", "line_id", "sen_id"], how="inner").reset_index(
                    drop=True)
                hist_metric_pdf = pd.merge(hist_metric_pdf,
                                           tmp_hist_metric_pdf[[question_col, f"{truth_col}_{rp_id}"]],
                                           on=[question_col], how="inner").reset_index(drop=True)
            time.sleep(1.2)

        hist_metric_pdf = hist_metric_pdf.reset_index(drop=True)
        out_metric_pdf = out_metric_pdf.reset_index(drop=True)
        hist_metric_pdf[truth_col] = 0.0
        out_metric_pdf[aim_col] = 0.0

        for rp_id in range(repeat_time):
            hist_metric_pdf[truth_col] = hist_metric_pdf[truth_col] + hist_metric_pdf[f"{truth_col}_{rp_id}"]
            out_metric_pdf[aim_col] = out_metric_pdf[aim_col] + out_metric_pdf[f"{aim_col}_{rp_id}"]

        hist_metric_pdf[truth_col] = hist_metric_pdf[truth_col] / repeat_time
        out_metric_pdf[aim_col] = out_metric_pdf[aim_col] / repeat_time

        out_metric_pdf = pd.merge(out_metric_pdf, hist_metric_pdf[[question_col, truth_col]], on=[question_col],
                                  how="inner").reset_index(drop=True)

        out_metric_pdf[f"hist_{aim_col}"] = 1.0 - out_metric_pdf[truth_col]

        # rouge_metric_pdf = compute_rouge_score(x_pdf=in_replay_pdf[remaining_cols],
        #                 ref_col=raw_answer_col,
        #                 hyp_col=replay_col)

        replay_pdf = pd.merge(replay_pdf[remaining_cols], out_metric_pdf[merge_add_col],
                              on=[question_col, "step_id", "line_id", "sen_id"], how="inner")

        replay_pdf = replay_pdf.reset_index(drop=True)

        replay_pdf[aim_col] = replay_pdf[aim_col]

        replay_pdf[f"{aim_col}_raw"] = replay_pdf[aim_col].copy()

        if is_truth != IGNORE and is_truth != GROUND_NO_WEIGHT:
            print("consider truth weight")
            for ii in replay_pdf.index:
                replay_pdf[aim_col].at[ii] = replay_pdf[aim_col].at[ii] * replay_pdf[truth_col].at[ii]
        else:
            print("do not consider truth weight")

        # elif is_truth == GROUND_WEIGHT:
        #     replay_pdf[aim_col] = replay_pdf[aim_col] - replay_pdf
        # replay_pdf[aim_col] = (((replay_pdf[aim_col] - replay_pdf[f"hist_{aim_col}"])/replay_pdf[f"hist_{aim_col}"])-(
        #     (0 - replay_pdf[f"hist_{aim_col}"])/replay_pdf[f"hist_{aim_col}"])) / (((1-replay_pdf[f"hist_{aim_col}"])/replay_pdf[f"hist_{aim_col}"])-(
        #         (0-replay_pdf[f"hist_{aim_col}"])/replay_pdf[f"hist_{aim_col}"]))
        # replay_pdf[aim_col] = replay_col[aim_col]*replay_col[truth_col]

        print(replay_pdf.shape)
        return replay_pdf

    else:
        in_replay_pdf = replay_pdf.copy()
        remaining_cols = list(set(now_replay_cols) - set(["nli_score", aim_col]))
        in_replay_pdf = in_replay_pdf[remaining_cols].reset_index(drop=True)

        merge_add_col = [question_col, "step_id", "line_id", "sen_id", aim_col, truth_col]
        if is_truth > TURTH_WEIGHT:
            merge_add_col.append(f"hist_{aim_col}")

        hist_metric_pdf = pd.DataFrame()
        out_metric_pdf = pd.DataFrame()

        for rp_id in range(repeat_time):
            print("hist: raw answer with refer answer")

            tmp_hist_metric_pdf = compute_nli_based_score_for_batch(x_pdf=hist_reasoning_pdf, pre_col=raw_answer_col,
                                                                    hyp_col=refer_answer_col, metric=metric,
                                                                    group_key_cols=[question_col],
                                                                    nli_prompt=nli_prompt,
                                                                    base_url=base_url,
                                                                    model_name=model_name,
                                                                    connect_repeat_time=connect_repeat_time,
                                                                    nli_repeat_time=nli_repeat_time,
                                                                    split_tokens=split_tokens,
                                                                    method=method, item_limit=item_limit,
                                                                    add_semantic=add_semantic,
                                                                    aim_col=f"{truth_col}_{rp_id}",
                                                                    nli_thread_num=nli_thread_num,
                                                                    computing_thread_num=computing_thread_num,
                                                                    default_device=default_device,
                                                                    scorer_num=scorer_num,
                                                                    add_structure=add_structure
                                                                    ).reset_index(drop=True)

            if is_truth < GROUND_WEIGHT:
                print("now: reasoning with replay answer")
                tmp_out_metric_pdf = compute_nli_based_score_for_batch(x_pdf=in_replay_pdf, pre_col=reasoning_col,
                                                                       hyp_col=replay_col, metric="nli",
                                                                       group_key_cols=[question_col, "step_id",
                                                                                       "line_id", "sen_id"],
                                                                       nli_prompt=nli_prompt,
                                                                       base_url=base_url,
                                                                       model_name=model_name,
                                                                       connect_repeat_time=connect_repeat_time,
                                                                       nli_repeat_time=nli_repeat_time,
                                                                       split_tokens=split_tokens,
                                                                       method=method, item_limit=item_limit,
                                                                       add_semantic=False,
                                                                       aim_col=f"{aim_col}_{rp_id}",
                                                                       nli_thread_num=nli_thread_num,
                                                                       computing_thread_num=computing_thread_num,
                                                                       default_device=default_device,
                                                                       scorer_num=scorer_num,
                                                                       add_structure=NO_STRUCT
                                                                       ).reset_index(drop=True)

            else:
                print("compute diff with groundtruth")
                print("now: reasoning with raw answer")
                tmp_out_metric_pdf = compute_nli_based_score_for_batch(x_pdf=in_replay_pdf, pre_col=reasoning_col,
                                                                       hyp_col=raw_answer_col, metric="nli",
                                                                       group_key_cols=[question_col, "step_id",
                                                                                       "line_id", "sen_id"],
                                                                       nli_prompt=nli_prompt,
                                                                       base_url=base_url,
                                                                       model_name=model_name,
                                                                       connect_repeat_time=connect_repeat_time,
                                                                       nli_repeat_time=nli_repeat_time,
                                                                       split_tokens=split_tokens,
                                                                       method=method, item_limit=item_limit,
                                                                       add_semantic=False,
                                                                       aim_col=f"{aim_col}_{rp_id}",
                                                                       nli_thread_num=nli_thread_num,
                                                                       computing_thread_num=computing_thread_num,
                                                                       default_device=default_device,
                                                                       scorer_num=scorer_num,
                                                                       add_structure=NO_STRUCT
                                                                       ).reset_index(drop=True)

            if rp_id == 0:
                out_metric_pdf = tmp_out_metric_pdf[
                    [question_col, "step_id", "line_id", "sen_id", f"{aim_col}_{rp_id}"]].copy()
                hist_metric_pdf = tmp_hist_metric_pdf[[question_col, f"{truth_col}_{rp_id}"]].copy()
            else:
                out_metric_pdf = pd.merge(out_metric_pdf,
                                          tmp_out_metric_pdf[
                                              [question_col, "step_id", "line_id", "sen_id", f"{aim_col}_{rp_id}"]],
                                          on=[question_col, "step_id", "line_id", "sen_id"], how="inner").reset_index(
                    drop=True)
                hist_metric_pdf = pd.merge(hist_metric_pdf,
                                           tmp_hist_metric_pdf[[question_col, f"{truth_col}_{rp_id}"]],
                                           on=[question_col], how="inner").reset_index(drop=True)

            time.sleep(1.2)

        hist_metric_pdf = hist_metric_pdf.reset_index(drop=True)
        out_metric_pdf = out_metric_pdf.reset_index(drop=True)
        hist_metric_pdf[truth_col] = 0.0
        out_metric_pdf[aim_col] = 0.0

        for rp_id in range(repeat_time):
            hist_metric_pdf[truth_col] = hist_metric_pdf[truth_col] + hist_metric_pdf[f"{truth_col}_{rp_id}"]
            out_metric_pdf[aim_col] = out_metric_pdf[aim_col] + out_metric_pdf[f"{aim_col}_{rp_id}"]

        hist_metric_pdf[truth_col] = hist_metric_pdf[truth_col] / repeat_time
        out_metric_pdf[aim_col] = out_metric_pdf[aim_col] / repeat_time

        out_metric_pdf = pd.merge(out_metric_pdf, hist_metric_pdf[[question_col, truth_col]], on=[question_col],
                                  how="inner").reset_index(drop=True)
        out_metric_pdf[f"hist_{aim_col}"] = 1.0 - out_metric_pdf[truth_col]

        replay_pdf = pd.merge(replay_pdf[remaining_cols], out_metric_pdf[merge_add_col],
                              on=[question_col, "step_id", "line_id", "sen_id"], how="inner")

        replay_pdf = replay_pdf.reset_index(drop=True)

        replay_pdf[aim_col] = replay_pdf[aim_col]
        # aim_result_pdf = replay_pdf[]
        replay_pdf[f"{aim_col}_raw"] = replay_pdf[aim_col].copy()
        # print(replay_pdf)

        if is_truth != IGNORE and is_truth != GROUND_NO_WEIGHT:
            print("consider truth weight")
            print(replay_pdf.index)
            print(aim_col)
            print(truth_col)
            for ii in replay_pdf.index:
                # print(ii)
                replay_pdf[aim_col].at[ii] = replay_pdf[aim_col].at[ii] * replay_pdf[truth_col].at[ii]
        else:
            print("do not consider truth weight")

        # elif is_truth == GROUND_WEIGHT:
        #     replay_pdf[aim_col] = (((replay_pdf[aim_col] - replay_pdf[f"hist_{aim_col}"])/replay_pdf[f"hist_{aim_col}"])-(
        #         (0 - replay_pdf[f"hist_{aim_col}"])/replay_pdf[f"hist_{aim_col}"])) / (((1-replay_pdf[f"hist_{aim_col}"])/replay_pdf[f"hist_{aim_col}"])-(
        #             (0-replay_pdf[f"hist_{aim_col}"])/replay_pdf[f"hist_{aim_col}"]))
        #     replay_pdf[aim_col] = replay_col[aim_col]*replay_col[truth_col]

        print(replay_pdf.shape)

        return replay_pdf