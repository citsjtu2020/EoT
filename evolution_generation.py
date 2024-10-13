import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from connect_to_LLMs import model_connect,call_LLM_response_for_prod
from prompt_design import PART_END,PART_START
from prompt_design import generate_prompt_for_evo,STEP_END,STEP_START

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm



BASE_URL_GPT = ""
BASE_URL_QWEN = ""
def single_connect_for_evo_reasoning_gpt(input_prompt,
                                         base_url=BASE_URL_GPT,
                                         model_name="gpt-4-turbo-128k",
                                         repeat_time=2, add_semantic=False):
    '''
    # A single query for GPT to realize answering/reasoning, which is an encapsulation of model connect
    :param input_prompt:
    :param base_url:
    :param model_name:
    :param repeat_time:
    :param add_semantic:
    :return:
    '''
    tmp_output_content = model_connect(prompt=input_prompt,
                                       base_url=base_url,
                                       model_name=model_name,
                                       repeat_time=repeat_time)

    return tmp_output_content






def generate_evoed_reasoning_single(
        input_prompt,
        base_url=BASE_URL_GPT,
        model_name="gpt-4-turbo-128k",
        debug=False,
        connect_repeat_time=2,
        evo_repeat_time=2):
    '''
    # a single process to query the LLM for producing an evolved reasoning process in an iteration of evolution.
    :param input_prompt:
    :param base_url:
    :param model_name:
    :param debug:
    :param connect_repeat_time:
    :param evo_repeat_time:
    :return:
    '''
    for j in range(evo_repeat_time):
        try:
            if "gpt" in model_name:
                base_url = BASE_URL_GPT

                generated_evo_reasoning = single_connect_for_evo_reasoning_gpt(input_prompt=input_prompt,
                                                                               base_url=base_url,
                                                                               model_name=model_name,
                                                                               repeat_time=connect_repeat_time)
            else:
                base_url = BASE_URL_QWEN

                generated_evo_reasoning = call_LLM_response_for_prod(prompts=input_prompt,
                                                                     base_url=base_url,
                                                                     debug=debug, repeat_time=connect_repeat_time)

        except Exception as ee1:
            print(ee1)
            generated_evo_reasoning = "Unknown"

        if PART_START in generated_evo_reasoning and PART_END in generated_evo_reasoning and STEP_START in generated_evo_reasoning and STEP_END in generated_evo_reasoning:
            break
        else:
            time.sleep(1.83)
            generated_evo_reasoning = "Unknown"

    return generated_evo_reasoning




def generate_evo_results_set(hist_rp_set, aim_qes=[],
                             question_col="generate_question", answer_col="manual_answer",
                             reference_col="reference_body2", hist_item_limit=10,
                             prompt_template="[PROMPT] [INSTRUCTION]",
                             out_col="evo_reasoning_process",
                             faith_col="faith_score_hyb",
                             truth_col="truth_score_hyb",
                             fact_col="fact_score",
                             base_url=BASE_URL_GPT,
                             model_name="gpt-4-turbo-128k",
                             connect_repeat_time=2,
                             evo_repeat_time=2, evo_thread_num=3, is_truth=True, is_super=True,
                             use_factual=True, use_fidelity=True,
                             use_reliability=True
                             ):
    '''

    A function to generate a group of evolved reasoning process concurrently in an iteration.
    :param hist_rp_set: history reasoning set
    :param aim_qes: tasks to be solved
    :param question_col: column name of question content
    :param answer_col: column name of answer content
    :param reference_col: column name of knowledge context content
    :param hist_item_limit: number of history items to be referenced
    :param prompt_template: prompting template
    :param out_col: column name of generated reasoning process content
    :param faith_col: column name of fidelity score
    :param truth_col: column name of reliability score
    :param fact_col: column name of factuality score
    :param base_url: URL to connect with LLMs
    :param model_name: used LLM name
    :param connect_repeat_time: number of connect repeat times
    :param evo_repeat_time: number of evolve repeat times
    :param evo_thread_num: number of concurrent evolve threads
    :param is_truth: use the weighed fidelity score
    :param is_super: if EoT is aware to reference answers supervisedly
    :param use_factual: if considering the factuality factor
    :param use_fidelity: if considering the fidelity factor
    :param use_reliability: if considering the reliability factor
    :return: a group of evolved reasoning processes
    '''
    now_to_evo_pdf = {question_col: [], answer_col: [], reference_col: []}
    if isinstance(aim_qes, str):
        aim_qes = [aim_qes]

    for aq in aim_qes:
        now_to_evo_pdf[question_col].append(aq)
        aq_initial_data = hist_rp_set[aq]
        aq_initial_keys = list(aq_initial_data.keys())
        aq_initial_keys.sort()
        aq_initial_pdf = aq_initial_data[aq_initial_keys[0]]["source"]
        # print(aq_initial_pdf)
        now_to_evo_pdf[answer_col].append(aq_initial_pdf[answer_col].values[0])
        now_to_evo_pdf[reference_col].append(aq_initial_pdf[reference_col].values[0])

    now_to_evo_pdf = pd.DataFrame(now_to_evo_pdf).reset_index(drop=True)
    now_to_evo_pdf[out_col] = "Unknown"

    tmp_res = now_to_evo_pdf.copy().reset_index(drop=True)
    tmp_stand_res = pd.DataFrame()
    # tmp_prompt = prompt_pio_template[i]
    kk2 = 0
    # generate_prompt_answers = []
    # print(tmp_prompt)
    # print(len(generate_prompt_answers))
    print(kk2)

    # tmp_res[out_col] = "Unknown"

    evo_lock = threading.Lock()

    progress = tqdm(total=tmp_res.shape[0], position=1, leave=True)

    def call_for_generate_evo_results(ind, hist_rp_set, input_question):
        tmp_evo_pre_data = ""
        try:
            if "gpt" in model_name:
                model_type = "gpt"
                base_url = BASE_URL_GPT
                print("use gpt")
            else:
                model_type = "qwen"
                base_url = BASE_URL_QWEN
                print("use qwen")

            tmp_evo_pre_data = generate_prompt_for_evo(hist_rp_set=hist_rp_set, model_type=model_type,
                                                       question_body=input_question,
                                                       answer_col=answer_col,
                                                       reference_col=reference_col,
                                                       hist_item_limit=hist_item_limit,
                                                       template=prompt_template,
                                                       faith_col=faith_col,
                                                       truth_col=truth_col,
                                                       fact_col=fact_col, is_truth=is_truth,
                                                       is_super=is_super,use_factual=use_factual,
                                                       use_fidelity=use_fidelity,
                                                       use_reliability=use_reliability)

            generated_evo_reasoning = generate_evoed_reasoning_single(
                input_prompt=tmp_evo_pre_data,
                base_url=base_url,
                model_name=model_name,
                debug=False,
                connect_repeat_time=connect_repeat_time,
                evo_repeat_time=evo_repeat_time)

            # generate_prompt_answers.append(extracted_label_str)

        except Exception as ee0:
            print(ee0)
            print(input_question)

            generated_evo_reasoning = "Unknown"

        # if ind == 0:
        #     print(tmp_evo_pre_data)

        #     # print(summary_out[0:30])

        # if ind < 1:
        #     print(generated_evo_reasoning)

        with evo_lock:
            tmp_res.at[ind, out_col] = generated_evo_reasoning
            # print("Done")
            progress.update(1)

        return

    with ThreadPoolExecutor(max_workers=evo_thread_num) as executor_evo:
        for ind, row in tmp_res.iterrows():
            print(ind)

            tmp_question = row[question_col]
            # print(tmp_question)
            # tmp_refer = row[reference_col]
            # tmp_reasoning = row[reasoning_content_col]
            executor_evo.submit(call_for_generate_evo_results, ind, hist_rp_set, tmp_question)
            time.sleep(0.5)

    # print(len(generate_prompt_answers))
    # tmp_res[out_col] = pd.Series(generate_prompt_answers)
    if tmp_stand_res.shape[0] > 0:
        tmp_res = pd.concat([tmp_stand_res.reset_index(drop=True), tmp_res.reset_index(drop=True)], axis=0).reset_index(
            drop=True)

    return tmp_res.reset_index(drop=True)