import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from prompt_design import REPLAY_META_BODY,PRE_end_token,PRE_start_token,NLI_end_token,NLI_start_token,HYP_end_token,HYP_start_token,NLI_PROMPT_TEMPLATE,EX_COMP_start_token,EX_COMP_end_token

from prompt_design import generate_prompt_for_NLI,generate_premise_data_in_RAG
from connect_to_LLMs import model_connect
from textual_process import format_token_for_re,extract_format_comp
from construct_reasoning_sources import  aggr_splited_steps
import pandas as pd


def nli_distribution_aggr(x_pdf, label_col="nli_label", score_col="fact_score"):
    in_pdf = x_pdf.copy()

    extract_distribution = {"entailment": 0.0, "contradiction": 0.0, "neural": 0.0, "other": 0.0}
    total_label_num = in_pdf.shape[0]

    for j in range(in_pdf.shape[0]):
        tmp_label = in_pdf[label_col].values[j]
        tmp_label = tmp_label.lower()
        if "entailment" in tmp_label or "entail" in tmp_label:
            extract_distribution["entailment"] += 1
        elif "contradiction" in tmp_label or "cond" in tmp_label:
            extract_distribution["contradiction"] += 1
        elif "neural" in tmp_label or "neu" in tmp_label:
            extract_distribution["neural"] += 1
        else:
            extract_distribution["other"] += 1

    for key in extract_distribution.keys():
        extract_distribution[key] = extract_distribution[key] / total_label_num

    out_res = {"entail": [extract_distribution["entailment"]],
               "contrad": [extract_distribution["contradiction"]],
               "neural": [extract_distribution["neural"]],
               score_col: [extract_distribution["entailment"]]}

    return pd.DataFrame(out_res).reset_index(drop=True)


def single_connect_for_NLI(premise_sentences,
                           hyp_sentences,
                           input_prompt=NLI_PROMPT_TEMPLATE["ENG"],
                           pre_start_token=PRE_start_token,
                           pre_end_token=PRE_end_token,
                           hyp_start_token=HYP_start_token,
                           hyp_end_token=HYP_end_token,
                           label_start_token=NLI_start_token,
                           label_end_token=NLI_end_token,
                           comp_start_token=EX_COMP_start_token,
                           comp_end_token=EX_COMP_end_token,
                           base_url="",
                           model_name="gpt-4-turbo-128k",
                           repeat_time=2, add_semantic=False):
    tmp_NLI_input = generate_prompt_for_NLI(premise_sentences=premise_sentences,
                                            hyp_sentences=hyp_sentences,
                                            input_prompt=input_prompt,
                                            pre_start_token=pre_start_token,
                                            pre_end_token=pre_end_token,
                                            hyp_start_token=hyp_start_token,
                                            hyp_end_token=hyp_end_token)

    tmp_output_content = model_connect(prompt=tmp_NLI_input,
                                       base_url=base_url,
                                       model_name=model_name,
                                       repeat_time=repeat_time)

    use_label_start = format_token_for_re(label_start_token)
    use_label_end = format_token_for_re(label_end_token)

    use_comp_start = format_token_for_re(comp_start_token)
    use_comp_end = format_token_for_re(comp_end_token)

    tmp_label_extract = extract_format_comp(prop=tmp_output_content,
                                            start_token=use_label_start,
                                            end_token=use_label_end,
                                            raw_start_token=label_start_token,
                                            raw_end_token=label_end_token)
    extracted_label_str = aggr_splited_steps(tmp_label_extract).strip()

    if add_semantic:
        tmp_find_componments = extract_format_comp(prop=tmp_output_content,
                                                   start_token=use_comp_start,
                                                   end_token=use_comp_end,
                                                   raw_start_token=comp_start_token,
                                                   raw_end_token=comp_end_token)

        return extracted_label_str, tmp_find_componments

    else:

        return extracted_label_str


def generate_nli_relation_single(premise_sentences,
                                 hyp_sentences,
                                 input_prompt=NLI_PROMPT_TEMPLATE["ENG"],
                                 pre_start_token=PRE_start_token,
                                 pre_end_token=PRE_end_token,
                                 hyp_start_token=HYP_start_token,
                                 hyp_end_token=HYP_end_token,
                                 label_start_token=NLI_start_token,
                                 label_end_token=NLI_end_token,
                                 base_url="",
                                 model_name="gpt-4-turbo-128k",
                                 connect_repeat_time=2,
                                 nli_repeat_time=2, add_semantic=False):
    for j in range(nli_repeat_time):
        try:
            if add_semantic:
                extracted_label_str, find_related_componments = single_connect_for_NLI(
                    premise_sentences=premise_sentences,
                    hyp_sentences=hyp_sentences,
                    input_prompt=input_prompt,
                    pre_start_token=pre_start_token,
                    pre_end_token=pre_end_token,
                    hyp_start_token=hyp_start_token,
                    hyp_end_token=hyp_end_token,
                    label_start_token=label_start_token,
                    label_end_token=label_end_token,
                    base_url=base_url,
                    model_name=model_name,
                    repeat_time=connect_repeat_time, add_semantic=add_semantic)
            else:
                extracted_label_str = single_connect_for_NLI(premise_sentences=premise_sentences,
                                                             hyp_sentences=hyp_sentences,
                                                             input_prompt=input_prompt,
                                                             pre_start_token=pre_start_token,
                                                             pre_end_token=pre_end_token,
                                                             hyp_start_token=hyp_start_token,
                                                             hyp_end_token=hyp_end_token,
                                                             label_start_token=label_start_token,
                                                             label_end_token=label_end_token,
                                                             base_url=base_url,
                                                             model_name=model_name,
                                                             repeat_time=connect_repeat_time)
        except Exception as ee1:
            print(ee1)
            extracted_label_str = "Unknown"
            if add_semantic:
                find_related_componments = []

        extracted_label_str = extracted_label_str.lower()

        if "entailment" in extracted_label_str or "entail" in extracted_label_str:
            extracted_label_str = "entailment"
            if add_semantic:
                if len(find_related_componments) > 0:
                    break
            else:
                break

        elif "contradiction" in extracted_label_str or "cond" in extracted_label_str:
            extracted_label_str = "contradiction"
            if add_semantic:
                if len(find_related_componments) > 0:
                    break
            else:
                break

        elif "neural" in extracted_label_str or "neu" in extracted_label_str:
            extracted_label_str = "neural"
            if add_semantic:
                if len(find_related_componments) > 0:
                    break
            else:
                break
        else:
            time.sleep(1.83)
            extracted_label_str = "Unknown"
            if add_semantic:
                find_related_componments = find_related_componments[:]

    extracted_label_str = extracted_label_str.lower()
    if add_semantic:
        return extracted_label_str, find_related_componments
    else:
        return extracted_label_str


def generate_NLI_results_set(in_pdf, question_col, reference_col,
                             input_prompt=NLI_PROMPT_TEMPLATE["ENG"],
                             out_col="nli_label",
                             reasoning_content_col="content",
                             pre_start_token=PRE_start_token,
                             pre_end_token=PRE_end_token,
                             hyp_start_token=HYP_start_token,
                             hyp_end_token=HYP_end_token,
                             label_start_token=NLI_start_token,
                             label_end_token=NLI_end_token,
                             base_url="",
                             model_name="gpt-4-turbo-128k",
                             connect_repeat_time=2,
                             nli_repeat_time=2, nli_thread_num=3):
    tmp_res = in_pdf.copy().reset_index(drop=True)
    tmp_stand_res = pd.DataFrame()
    # tmp_prompt = prompt_pio_template[i]
    kk2 = 0
    generate_prompt_answers = []
    # print(tmp_prompt)
    print(len(generate_prompt_answers))
    print(kk2)
    if out_col in in_pdf.columns and "Unknown" not in in_pdf[out_col].unique().tolist():
        return tmp_res
    elif out_col in in_pdf.columns:
        tmp_stand_res = in_pdf[~in_pdf[out_col].isin(["Unknown"])].reset_index(drop=True)
        tmp_res = in_pdf[in_pdf[out_col].isin(["Unknown"])].reset_index(drop=True)
    else:
        tmp_res = in_pdf.copy().reset_index(drop=True)

    tmp_res[out_col] = "Unknown"

    nli_lock = threading.Lock()

    progress = tqdm(total=tmp_res.shape[0], position=1, leave=True)

    def call_for_generate_relation(ind, input_question, input_refer, input_reasoning):
        try:
            tmp_nli_pre_data = generate_premise_data_in_RAG(question_body=input_question,
                                                            reference_body=input_refer)

            extracted_label_str = generate_nli_relation_single(premise_sentences=tmp_nli_pre_data,
                                                               hyp_sentences=input_reasoning,
                                                               input_prompt=input_prompt,
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

            extracted_label_str = extracted_label_str.lower()

            # generate_prompt_answers.append(extracted_label_str)

        except Exception as ee0:
            print(ee0)
            print(input_question)
            print(input_refer)
            print(input_reasoning)
            print(input_prompt)
            extracted_label_str = "Unknown"

        # if ind == 0:
        #     print(tmp_nli_pre_data)

        #     # print(summary_out[0:30])

        # if ind < 1:
        #     print(extracted_label_str)

        with nli_lock:
            tmp_res.at[ind, out_col] = extracted_label_str
            # print("Done")
            progress.update(1)

        return

    with ThreadPoolExecutor(max_workers=nli_thread_num) as executor_nli:
        for ind, row in tmp_res.iterrows():
            print(ind)

            tmp_question = row[question_col]
            # print(tmp_question)
            tmp_refer = row[reference_col]
            tmp_reasoning = row[reasoning_content_col]
            executor_nli.submit(call_for_generate_relation, ind, tmp_question, tmp_refer, tmp_reasoning)
            time.sleep(1.2)

    # print(len(generate_prompt_answers))
    # tmp_res[out_col] = pd.Series(generate_prompt_answers)
    if tmp_stand_res.shape[0] > 0:
        tmp_res = pd.concat([tmp_stand_res.reset_index(drop=True), tmp_res.reset_index(drop=True)], axis=0).reset_index(
            drop=True)

    return tmp_res.reset_index(drop=True)