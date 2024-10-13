import time
import pandas as pd
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from construct_reasoning_sources import aggr_splited_steps

from connect_to_LLMs import call_LLM_response_for_prod,model_connect
from textual_process import format_token_for_re,extract_format_comp




def generate_reasong_process_set(in_pdf, prompt_body, question_col="generate_question", refer_col="reference_body2",
                                 answer_col="manual_answer", out_col="manual_reasoning",
                                 metric="rouge-l", eval_model="qwen72B",
                                 reasoning_thread_num=4, repeat_time=2,
                                 is_super=True, ans_start="[ANS]",
                                 ans_end="[/ANS]"):
    '''

    Generate a set of reasoning process
    :param in_pdf:
    :param prompt_body:
    :param question_col:
    :param refer_col:
    :param answer_col:
    :param out_col:
    :param metric:
    :param eval_model:
    :param reasoning_thread_num:
    :param repeat_time:
    :param is_super:
    :param ans_start:
    :param ans_end:
    :return:
    '''
    tmp_res = in_pdf.copy().reset_index(drop=True)
    tmp_stand_res = pd.DataFrame()
    # tmp_prompt = prompt_pio_template[i]
    kk2 = 0
    generate_prompt_answers = []
    # print(tmp_prompt)
    out_ans_col = f"{out_col}_answer"

    print(len(generate_prompt_answers))
    print(kk2)

    if out_col in in_pdf.columns and "Unknown" not in in_pdf[out_col].unique().tolist() and is_super:
        return tmp_res

    elif out_col in in_pdf.columns:
        tmp_stand_res = in_pdf[~in_pdf[out_col].isin(["Unknown"])].reset_index(drop=True)
        tmp_res = in_pdf[in_pdf[out_col].isin(["Unknown"])].reset_index(drop=True)

    else:
        tmp_res = in_pdf.copy().reset_index(drop=True)

    if not is_super and tmp_stand_res.shape[0] > 0:
        if out_ans_col in tmp_stand_res.columns and "Unknown" not in tmp_stand_res[out_ans_col].unique().tolist():
            tmp_stand_res = tmp_stand_res.copy()
        elif out_ans_col in tmp_stand_res.columns:
            tmp_stand_res_true = tmp_stand_res[~tmp_stand_res[out_ans_col].isin(["Unknown"])].reset_index(drop=True)
            tmp_stand_res_false = tmp_stand_res[tmp_stand_res[out_ans_col].isin(["Unknown"])].reset_index(drop=True)
            tmp_res = pd.concat([tmp_res, tmp_stand_res_false], axis=0).reset_index(drop=True)
            tmp_stand_res = tmp_stand_res_true.copy()
        else:
            tmp_res = pd.concat([tmp_stand_res, tmp_res], axis=0).reset_index(drop=True)

    tmp_res[out_col] = "Unknown"
    if not is_super:
        tmp_res[out_ans_col] = "Unknown"

    reasoning_lock = threading.Lock()

    progress = tqdm(total=tmp_res.shape[0], position=1, leave=True)

    def call_for_generate_reasoning_out(ind, input_question, input_refer, input_answer):
        try:
            tmp_question_input = generate_reasoing_process_for_qa(prompt_body=prompt_body, body=input_refer,
                                                                  question_text=input_question,
                                                                  answer_text=input_answer,
                                                                  is_super=is_super,evo_model_name=eval_model)
            print(tmp_question_input)
            if "gpt" in eval_model:
                print("use gpt")
                summary_out = model_connect(prompt=tmp_question_input,model_name=eval_model,repeat_time=repeat_time)
            else:
                print("use qwen")
                summary_out = call_LLM_response_for_prod(prompts=tmp_question_input, repeat_time=repeat_time)
            print(summary_out)

            if not summary_out:
                summary_out = "Unknown"



        except Exception as ee0:
            print(ee0)
            print(tmp_question)
            print(prompt_body)
            print(tmp_refer)
            summary_out = "Unknown"

        # if ind == 0:
        #     print(tmp_question_input)

        #     # print(summary_out[0:30])

        # if ind < 1:
        #     print(summary_out)

        with reasoning_lock:
            tmp_res.at[ind, out_col] = summary_out
            try:
                if not is_super:
                    re_start_token = format_token_for_re(ans_start)
                    re_end_token = format_token_for_re(ans_end)
                    if ans_end not in summary_out:
                        summary_out = summary_out.strip() + "\n" + ans_end

                    print(summary_out)
                    raw_ans_res_list = extract_format_comp(prop=summary_out, start_token=re_start_token,
                                                           end_token=re_end_token,
                                                           raw_start_token=ans_start,
                                                           raw_end_token=ans_end)
                    if raw_ans_res_list:
                        forward_ans = aggr_splited_steps(raw_ans_res_list).strip()
                    else:
                        forward_ans = "Unknown"

                    tmp_res.at[ind, out_ans_col] = forward_ans
            except Exception as ee00:
                print(ee00)

            progress.update(1)

        return

    with ThreadPoolExecutor(max_workers=reasoning_thread_num) as executor_reason:
        for ind, row in tmp_res.iterrows():
            print(ind)

            tmp_question = row[question_col]
            # print(tmp_question)

            tmp_refer = row[refer_col]
            tmp_answer = row[answer_col]

            executor_reason.submit(call_for_generate_reasoning_out, ind, tmp_question, tmp_refer, tmp_answer)

            time.sleep(1.2)

    if tmp_stand_res.shape[0] > 0:
        tmp_res = pd.concat([tmp_stand_res.reset_index(drop=True), tmp_res.reset_index(drop=True)], axis=0).reset_index(
            drop=True)

    return tmp_res.reset_index(drop=True)


import re


def generate_reasoing_process_for_qa(prompt_body, body, question_text,
                                     answer_text, is_super=True, evo_model_name="qwen"):
    '''

    generate a single reasoning process to guide the question answering
    :param prompt_body:
    :param body:
    :param question_text:
    :param answer_text:
    :param is_super:
    :param evo_model_name:
    :return:
    '''
    if is_super:
        user_content = ' The reference materials and documents provided by the user are as follows Documents:\n' + body + '\n The specific question input by the user is: ' + question_text + '. \n The answer is: ' + answer_text + '\n Please output the Reasoning Process you believe you would use to obtain the aforementioned Answer when answering the Question based on the Documents:'
    else:
        user_content = ' The reference materials and documents provided by the user are as follows Documents:\n' + body + '\n The specific question input by the user is: ' + question_text + '. \n ' + 'Please output the Reasoning Process you believe you would use or consider to obtain the correct Answer when answering the Question based on the Documents:'

    sys_prompt = 'Hello, you are the intelligent robot of the smart Q&A project.' + prompt_body

    if "gpt" in evo_model_name:
        template = "[PROMPT] [INSTRUCTION]"
        sys_prompt = sys_prompt + "\n"
        # gen_prompt = re.sub("\[PROMPT\]", sys_prompt, template)

        instruction = user_content

        # gen_prompt = re.sub("\[INSTRUCTION\]", instruction, gen_prompt)
        gen_prompt = sys_prompt + instruction
        print(gen_prompt)
        prompts = gen_prompt
    else:
        prompts = [{"role": "system", "content": sys_prompt},
                   {"role": "user", "content": user_content}]

    return prompts

# re_start_token = format_token_for_re(start_token)
# re_end_token = format_token_for_re(end_token)

# modified_proc_items = extract_format_comp(prop=modified_proc_items_res, start_token=re_start_token,
#                                           end_token=re_end_token,
#                                           raw_start_token=start_token,
#                                           raw_end_token=end_token)