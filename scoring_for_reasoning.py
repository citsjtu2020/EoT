import pandas as pd

from construct_reasoning_sources import generate_reasoning_process_faithful_test_set
from prompt_design import REPLAY_META_BODY,PRE_end_token,PRE_start_token,NLI_end_token,NLI_start_token,HYP_end_token,HYP_start_token,NLI_PROMPT_TEMPLATE
from replay_generate import generate_replay_answer_set
from scoring_metrics import scoring_factuality_score
from scoring_metrics import scoring_faithfulity_score_v2
from scoring_metrics import IGNORE,GROUND_NO_WEIGHT,GROUND_WEIGHT,TURTH_WEIGHT
from bleurt import score
import time
MASK = 0
HOLD = 2
MODIFY = 1

NO_STRUCT = 0
SPLITED_STRUCT = 1
CON_PART_STRUCT = 2
TOTAL_STRUCT = 3

TOP_START = "**"
TOP_END = "**"


# the scoring fuction for each step of reasoning thoughts, which is the encapsulation of the three scoring metric estimation in scoring_metrics.py
BASE_GPT_URL = ""
def scoring_question_reasoning_for_each_step(input_df,
                                             in_extracted_componments,
                                             scorer_0:score.BleurtScorer,
                                             scorer_1:score.BleurtScorer,
                                             scorer_2:score.BleurtScorer,
                                             scorer_3:score.BleurtScorer,
                                             aim_qes=[],
                                             is_fact=True,
                                             is_truth=True,
                                             strategy=MASK,
                                             question_col="generate_question",
                                             reference_col="reference_body2",
                                             reasoning_type="steps",
                                             answer_col="manual_answer",
                                             faith_metric="nli_bleurt",
                                             gpt_base_url=BASE_GPT_URL,
                                             gpt_model_name="gpt-4-turbo-128k", modify_repeat_time=2,
                                             evo_model_name="gpt-4-turbo-128k",
                                             modify_worker_num=3,
                                             hist_replay_answer_col="hist_replay_answer",
                                             faith_replay_answer_col="replay_answer",
                                             faith_col="faith_score_hyb",
                                             truth_col="truth_score_hyb",
                                             hist_reasoning_col="hist_reasoning_input",
                                             replay_thread_num=3,
                                             nli_thread_num=3,
                                             computing_thread_num=4,
                                             connect_repeat_time=2,
                                             nli_repeat_time=2,
                                             split_tokens=["\n", "。"],
                                             faith_method=2,
                                             item_limit=100,
                                             add_semantic=True,
                                             default_device='/device:GPU:1',
                                             scorer_num=4,
                                             repeat_time=2,
                                             add_structure=NO_STRUCT,is_super=True,is_replay=True
                                             ):
    '''
    strategy: 0 -> Mask, 1-> MODIFY
    '''

    if isinstance(aim_qes, str):
        aim_qes = [aim_qes]

    in_df = input_df.copy()

    if repeat_time < 1:
        repeat_time = 1

    repeat_time = int(repeat_time)
    # print(in_extracted_componments)

    aim_qes_reasoning_process_test_set = generate_reasoning_process_faithful_test_set(eval_df=in_df,
                                                                                      source_reasonings=in_extracted_componments,
                                                                                      aim_questions=aim_qes,
                                                                                      question_col=question_col,
                                                                                      reference_col=reference_col,
                                                                                      reasoning_type=reasoning_type,
                                                                                      answer_col=answer_col,
                                                                                      depth=0, method=strategy,
                                                                                      top_start=TOP_START,
                                                                                      top_end=TOP_END,
                                                                                      first_token="\n",
                                                                                      second_token="。",
                                                                                      modify_template="[PROMPT] [DEMO] [INSTRUCTION]",
                                                                                      modify_start_token="[MODIFY]",
                                                                                      modify_end_token="[/MODIFY]",
                                                                                      modify_few_shot=True,
                                                                                      base_url=gpt_base_url,
                                                                                      model_name=gpt_model_name,
                                                                                      modify_repeat_time=modify_repeat_time,
                                                                                      repeat_time=connect_repeat_time,
                                                                                      worker_num=modify_worker_num,is_super=is_super,is_replay=is_replay
                                                                                      )

    aim_qes_test_set_for_nli = generate_reasoning_process_faithful_test_set(eval_df=in_df,
                                                                            source_reasonings=in_extracted_componments,
                                                                            aim_questions=aim_qes,
                                                                            question_col=question_col,
                                                                            reference_col=reference_col,
                                                                            reasoning_type=reasoning_type,
                                                                            answer_col=answer_col,
                                                                            depth=1, method=MASK,
                                                                            top_start=TOP_START,
                                                                            top_end=TOP_END,
                                                                            first_token="\n", second_token="。",
                                                                            modify_template="[PROMPT] [DEMO] [INSTRUCTION]",
                                                                            modify_start_token="[MODIFY]",
                                                                            modify_end_token="[/MODIFY]",
                                                                            modify_few_shot=True,
                                                                            base_url=gpt_base_url,
                                                                            model_name=gpt_model_name,
                                                                            modify_repeat_time=modify_repeat_time,
                                                                            repeat_time=connect_repeat_time,
                                                                            worker_num=modify_worker_num,is_super=is_super,is_replay=is_replay
                                                                            )
    ### STAGE 1: 计算原始的replay answer

    his_input_cols = [answer_col, question_col,
                      reference_col, hist_reasoning_col]

    if hist_replay_answer_col in list(in_df.columns):
        his_input_cols.append(hist_replay_answer_col)

    if not is_super and not is_replay:
        his_input_cols.append("forward_reasoning_answer")

    if hist_reasoning_col not in list(in_df.columns):
        hist_reasoning_pdf = {question_col: [], hist_reasoning_col: []}
        for aq in aim_qes:
            hist_reasoning_pdf[question_col].append(aq)
            tmp_raw_reasoning = \
            aim_qes_reasoning_process_test_set[aim_qes_reasoning_process_test_set[question_col].isin([aq])][
                "hist_reasoning"].values[0]
            hist_reasoning_pdf[hist_reasoning_col].append(tmp_raw_reasoning)

        hist_reasoning_pdf = pd.DataFrame(hist_reasoning_pdf)

        hist_pdf = pd.merge(in_df[[answer_col, question_col, reference_col]].copy().reset_index(drop=True),
                            hist_reasoning_pdf, on=question_col, how="inner").reset_index(drop=True)
        hist_pdf = hist_pdf[his_input_cols]
    else:
        hist_pdf = in_df[his_input_cols].copy().reset_index(drop=True)

    print(hist_pdf.columns)

    if is_super:

        hist_result_pdf = generate_replay_answer_set(in_pdf=hist_pdf,
                                                 prompt_body=REPLAY_META_BODY,
                                                 question_col=question_col,
                                                 refer_col=reference_col,
                                                 reasoning_col=hist_reasoning_col, out_col=hist_replay_answer_col,
                                                 replay_thread_num=replay_thread_num,eval_model=evo_model_name)
    else:
        if is_replay:
            hist_result_pdf = generate_replay_answer_set(in_pdf=hist_pdf,
                                                         prompt_body=REPLAY_META_BODY,
                                                         question_col=question_col,
                                                         refer_col=reference_col,
                                                         reasoning_col=hist_reasoning_col,
                                                         out_col=hist_replay_answer_col,
                                                         replay_thread_num=replay_thread_num,eval_model=evo_model_name)

        else:
            hist_result_pdf = hist_pdf.copy()
            hist_result_pdf[hist_replay_answer_col] = hist_pdf["forward_reasoning_answer"].copy()

    if hist_reasoning_col not in list(aim_qes_reasoning_process_test_set.columns):
        aim_qes_reasoning_process_test_set = pd.merge(aim_qes_reasoning_process_test_set, hist_result_pdf[[question_col,
                                                                                                           hist_reasoning_col,
                                                                                                           hist_replay_answer_col]],
                                                      on=question_col, how="inner")
    else:
        aim_qes_reasoning_process_test_set = pd.merge(aim_qes_reasoning_process_test_set, hist_result_pdf[[question_col,
                                                                                                           hist_replay_answer_col]],
                                                      on=question_col, how="inner")

    # Stage 2: 计算问题的Faithful Score,Factual Score,truth score
    # out_col="nli_label",
    # 若考虑factual score进行相应计算：

    NLI_fact_score_res = pd.DataFrame()
    NLI_fact_res = pd.DataFrame()
    for rp_id in range(repeat_time):
        if is_fact:
            tmp_NLI_fact_score_res, NLI_fact_res = scoring_factuality_score(input_pdf=aim_qes_test_set_for_nli,
                                                                            question_col=question_col,
                                                                            reference_col=reference_col,
                                                                            input_prompt=NLI_PROMPT_TEMPLATE["ENG"],
                                                                            reasoning_content_col="content",
                                                                            pre_start_token=PRE_start_token,
                                                                            pre_end_token=PRE_end_token,
                                                                            hyp_start_token=HYP_start_token,
                                                                            hyp_end_token=HYP_end_token,
                                                                            label_start_token=NLI_start_token,
                                                                            label_end_token=NLI_end_token,
                                                                            base_url=gpt_base_url,
                                                                            model_name=gpt_model_name,
                                                                            metric="nli",
                                                                            score_col=f"fact_score_{rp_id}",
                                                                            group_key='step_id',
                                                                            connect_repeat_time=connect_repeat_time,
                                                                            nli_repeat_time=nli_repeat_time,
                                                                            nli_thread_num=nli_thread_num)
            # entail	contrad	neural
            tmp_NLI_fact_score_res.rename(columns={"entail": f"entail_{rp_id}",
                                                   "contrad": f"contrad_{rp_id}",
                                                   "neural": f"neural_{rp_id}"
                                                   }, inplace=True)

            if rp_id == 0:
                NLI_fact_score_res = tmp_NLI_fact_score_res.copy()
            else:
                # [[question_col,f"fact_score_{rp_id}","step_id"]]
                NLI_fact_score_res = pd.merge(NLI_fact_score_res,
                                              tmp_NLI_fact_score_res[[question_col, f"entail_{rp_id}",
                                                                      f"contrad_{rp_id}", f"neural_{rp_id}",
                                                                      f"fact_score_{rp_id}", "step_id"]],
                                              on=[question_col, "step_id"], how="inner").reset_index(drop=True)
            time.sleep(1.2)

        else:
            NLI_fact_score_res = pd.DataFrame()
            NLI_fact_res = pd.DataFrame()
            aim_qes_reasoning_process_test_set["fact_score"] = -1.0

    NLI_fact_res = NLI_fact_res.copy()

    NLI_fact_score_res = NLI_fact_score_res.reset_index(drop=True)

    if is_fact:
        NLI_fact_score_res["fact_score"] = 0.0
        NLI_fact_score_res["entail"] = 0.0
        NLI_fact_score_res["contrad"] = 0.0
        NLI_fact_score_res["neural"] = 0.0

        for rp_id in range(repeat_time):
            NLI_fact_score_res["fact_score"] = NLI_fact_score_res["fact_score"] + NLI_fact_score_res[
                f"fact_score_{rp_id}"]
            NLI_fact_score_res["entail"] = NLI_fact_score_res["entail"] + NLI_fact_score_res[f"entail_{rp_id}"]
            NLI_fact_score_res["contrad"] = NLI_fact_score_res["contrad"] + NLI_fact_score_res[f"contrad_{rp_id}"]
            NLI_fact_score_res["neural"] = NLI_fact_score_res["neural"] + NLI_fact_score_res[f"neural_{rp_id}"]

        NLI_fact_score_res["fact_score"] = NLI_fact_score_res["fact_score"] / repeat_time
        NLI_fact_score_res["entail"] = NLI_fact_score_res["entail"] / repeat_time
        NLI_fact_score_res["contrad"] = NLI_fact_score_res["contrad"] / repeat_time
        NLI_fact_score_res["neural"] = NLI_fact_score_res["neural"] / repeat_time

    aim_qes_reasoning_process_test_set = pd.merge(aim_qes_reasoning_process_test_set, NLI_fact_score_res[[question_col,
                                                                                                          "fact_score",
                                                                                                          "step_id"]],
                                                  how="inner", on=[question_col, "step_id"]).reset_index(drop=True)

    # 计算faith score和问题整体的truth score:
    if not is_truth:
        truth_add_strategy = IGNORE
    else:
        truth_add_strategy = TURTH_WEIGHT
        # faith_col="faith_score_hyb",
        #  truth_col="truth_score_hyb",
    print(question_col)
    aim_qes_reasoning_process_test_set = scoring_faithfulity_score_v2(input_pdf=aim_qes_reasoning_process_test_set,
                                                                      scorer_0=scorer_0,
                                                                      scorer_1=scorer_1,
                                                                      scorer_2=scorer_2,
                                                                      scorer_3=scorer_3,
                                                                      prompt_body=REPLAY_META_BODY,
                                                                      question_col=question_col,
                                                                      refer_col=reference_col,
                                                                      raw_answer_col=answer_col,
                                                                      refer_answer_col=hist_replay_answer_col,
                                                                      reasoning_col="reasoning_input",
                                                                      replay_col=faith_replay_answer_col,
                                                                      metric=faith_metric,
                                                                      hist_reasoning_col=hist_reasoning_col,
                                                                      aim_col=faith_col, truth_col=truth_col,
                                                                      nli_prompt=NLI_PROMPT_TEMPLATE["ENG"],
                                                                      base_url=gpt_base_url,
                                                                      model_name=gpt_model_name,
                                                                      connect_repeat_time=connect_repeat_time,
                                                                      nli_repeat_time=nli_repeat_time,
                                                                      split_tokens=split_tokens,
                                                                      method=faith_method,
                                                                      item_limit=item_limit,
                                                                      add_semantic=add_semantic,
                                                                      nli_thread_num=nli_thread_num,
                                                                      computing_thread_num=computing_thread_num,
                                                                      default_device=default_device,
                                                                      scorer_num=scorer_num,
                                                                      replay_thread_num=replay_thread_num,
                                                                      is_truth=truth_add_strategy,
                                                                      repeat_time=repeat_time,
                                                                      add_structure=add_structure,eval_model=evo_model_name
                                                                      )

    return aim_qes_reasoning_process_test_set, hist_result_pdf, aim_qes_test_set_for_nli, NLI_fact_score_res, NLI_fact_res