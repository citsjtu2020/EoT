import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import re
from textual_process import format_token_for_re,extract_format_comp,split_step_thinking
from connect_to_LLMs import model_connect
import pandas as pd
# from prompt_design import PRE_end_token,PRE_start_token,PART_END,PART_START,NLI_start_token,NLI_end_token,NLI_PROMPT_TEMPLATE
# # from NLI_socre_compute import
# from prompt_design import HYP_end_token,HYP_start_token



# process the reasoning steps with operations of modify, mask and concatenate

def construct_modified_step(input_step_params,give_id=1,top_start="**",top_end="**"):
    tmp_content = input_step_params["content"]
    tmp_out_content = aggr_splited_steps(input_data=tmp_content)
    tmp_out_content = tmp_out_content.strip()

    tmp_title = input_step_params["top"]
    if not tmp_title:
        tmp_title = f"{give_id}. {top_start}{tmp_title}{top_end}"
    else:
        tmp_title = f"{give_id}. {top_start}{tmp_title}{top_end}: \n"
    tmp_out_data = f"{tmp_title}{tmp_out_content}"
    return tmp_out_data


def extract_reasoning_componments(input_pdf, question_col="generate_question",
                                  reasoning_col="manual_reasoning",
                                  part_start="[PART]",
                                  part_end="[/PART]",
                                  step_start="[STEP]",
                                  step_end="[/STEP]", aim_res_part_id=["start", "steps", "summ"]):
    use_part_start = format_token_for_re(part_start)
    use_part_end = format_token_for_re(part_end)
    use_step_start = format_token_for_re(step_start)
    use_step_end = format_token_for_re(step_end)

    start_list = {}
    step_list = {}
    summ_list = {}
    # total_results = {"start":{},"steps":{},"summ":{}}
    if isinstance(aim_res_part_id, str):
        aim_res_part_id = [aim_res_part_id]

    aim_part_num = len(aim_res_part_id)
    total_results = {}
    tmp_extracted_lists = {}
    for arpi in aim_res_part_id:
        total_results[arpi] = {}
        tmp_extracted_lists[arpi] = {}

    for i in range(input_pdf.shape[0]):

        tmp_qs = input_pdf[question_col].values[i]
        tmp_reasoning = input_pdf[reasoning_col].values[i]
        if not tmp_reasoning.strip().endswith(part_end) and part_end not in tmp_reasoning:
            tmp_reasoning = tmp_reasoning + part_end
        if not tmp_reasoning.strip().startswith(part_start) and part_start not in tmp_reasoning:
            tmp_reasoning = part_start + tmp_reasoning

        if part_start not in tmp_reasoning:
            tmp_reasoning = part_start + tmp_reasoning

        if part_end not in tmp_reasoning:
            tmp_reasoning = tmp_reasoning + part_end

        split_reasoning_comp = extract_format_comp(tmp_reasoning,
                                                   start_token=use_part_start,
                                                   end_token=use_part_end,
                                                   raw_end_token=part_end,
                                                   raw_start_token=part_start)
        # print(tmp_qs)
        # print(split_reasoning_comp)

        # if len(split_reasoning_comp) == aim_part_num:
        # print(i)
        for j in range(np.min([len(split_reasoning_comp), aim_part_num])):
            arpi = aim_res_part_id[j]
            tmp_extracted_lists[arpi][tmp_qs] = split_reasoning_comp[j]
            if step_start in split_reasoning_comp[j] and step_end not in split_reasoning_comp[j]:
                tmp_split_reasoning_comp = split_reasoning_comp[j] + step_end
            elif step_start not in split_reasoning_comp[j] and step_end in split_reasoning_comp[j]:
                tmp_split_reasoning_comp = step_start + split_reasoning_comp[j]
            else:
                tmp_split_reasoning_comp = split_reasoning_comp[j]

            if step_start in tmp_split_reasoning_comp and step_end in tmp_split_reasoning_comp:
                tmp_steps = extract_format_comp(tmp_split_reasoning_comp,
                                                start_token=use_step_start,
                                                end_token=use_step_end,
                                                raw_start_token=step_start,
                                                raw_end_token=step_end)
                tmp_extracted_lists[arpi][tmp_qs] = tmp_steps

    # arpi = aim_res_part_id[0]
    # .intersection(set(list(summ_list)))

    inter_qs_keys = set(list(tmp_extracted_lists["steps"].keys()))
    # for jj in range(aim_part_num):
    #     arpi = aim_res_part_id[jj]
    #     if jj == 0 or len(list(inter_qs_keys))<1:
    #         inter_qs_keys = set(list(tmp_extracted_lists[arpi].keys()))
    #     else:
    #         if len(list(tmp_extracted_lists[arpi].keys())) > 0:
    #             inter_qs_keys = inter_qs_keys.intersection(set(list(tmp_extracted_lists[arpi].keys())))

    for it in list(inter_qs_keys):
        for arpi in aim_res_part_id:
            total_results[arpi][it] = tmp_extracted_lists[arpi][it]

    return total_results



def replace_item_in_list(raw_list,index,new_value):
    if index >= len(raw_list):
        index = -1
    if index <= (0-len(raw_list)):
        index = 0
    input_list = raw_list[:]
    if index == -1 or index == len(raw_list)-1:
        out_list = input_list[:index]+[new_value]
    else:
        out_list = input_list[:index]+[new_value]+input_list[index+1:]
    return out_list[:]

def aggr_splited_steps(input_data):
    in_stentences = input_data[:]
    out_sentence = ""
    for j in range(len(in_stentences)):
        if isinstance(in_stentences[j],list):
            tmp_out_sentence = aggr_splited_steps(in_stentences[j][:])
        else:
            tmp_out_sentence = in_stentences[j]
        out_sentence += tmp_out_sentence
    return out_sentence




def drop_item(input_procs, idx=0):
    tmp_to_drop_inputs = input_procs[:]
    tmp_to_drop_inputs.pop(idx)
    return tmp_to_drop_inputs[:]


def generate_prompt_for_modify(input_sentences, template="[PROMPT] [DEMO] [INSTRUCTION]", start_token="[MODIFY]",
                               end_token="[/MODIFY]", few_shot=True):
    meta_prompt = """Your task is to rewrite the input sentences SETENCES as examples with an opposite or contradictory semantic/meaning. Please note: (1) Make sure to generate an example with an opposite meaning or semantic contradiction for each individual sentence. Then combine the generated sentences in their original order as much as possible while ensuring the output remains coherent. (2) For each sentence, you are allowed to understand its semantics and generate the rewrite with the opposite meaning based on the context provided in this input."""
    format_prompt = f"""For each input, your output must adhere to the following format requirements:\n 1. The beginning of each output should be marked with {start_token}, and the end with {end_token}.\n """
    input_examples = """First, identify from the problem that the user encountered a timeout issue while trying to use SSH to connect to a git repository."""
    demonstration = """\nSpecific output format examples are shown below as Demonstrations:\n\nDemonstrations:\nDemonstration 1: Input: SENTENCES: First, identify from the problem that the user encountered a timeout issue while trying to use SSH to connect to a git repository. Output: [MODIFY]\nFirst, identify from the problem that the user encountered no timeout issue at all when attempting to use SSH to connect to the git repository.\n[/MODIFY] \nDemonstration 2: Input: SENTENCES: - The document mentioned several possible causes, such as network outage, SSH configuration errors, DNS resolution issues, firewall restrictions, etc.\n - Provided the use of `ssh -v` command for debugging to check the detailed information during the connection process.\n - Emphasized the correct configuration and permission setting of SSH keys.\n - Mentioned possible DNS resolution issues, calling for a check of the hosts file or a network ping test.\n - Also mentioned solutions like proxy settings adjustments and temporarily disabling SSH host checks.\n - Lastly, if all methods are ineffective, suggested using the HTTPS protocol or contacting technical support. Output: [MODIFY]\nThe document ruled out all common causes, such as the network being up, correct SSH configuration, accurate DNS resolution, no firewall restrictions, etc.\n Suggest not using the ssh -v command for debugging, since it will not show detailed information during the connection process.\n Ignored the configuration and permission settings of SSH keys, implying they are irrelevant to the problem.\n Overlooked the DNS resolution issues, and there is no need to check the hosts file or perform a network ping test.\n Did not mention the need to adjust proxy settings or temporarily disable SSH host checks.\n Lastly, if you have tried all methods, advise against using the HTTPS protocol and not contacting technical support.\n[/MODIFY]"""

    if few_shot:
        use_demonstration = demonstration
    else:
        demonstration = ""

    # gen_prompt = re.sub("\[PROMPT\]", f"{meta_prompt}\n{format_prompt}", template)
    # gen_prompt = re.sub("\[DEMO\]", demonstration, gen_prompt)
    instruction = f"""Specifically, the sentences input on this occasion SETENCES are: \"{input_sentences}\" \n Therefore, your corresponding rewrite should be:"""
    # gen_prompt = re.sub("\[INSTRUCTION\]", instruction, gen_prompt)
    gen_prompt = f"{meta_prompt}\n{format_prompt} {demonstration}\n{instruction}"
    return gen_prompt


BASE_URL_GPT = ""
def modify_item_by_calling_LLM(input_process,
                               template="[PROMPT] [DEMO] [INSTRUCTION]",
                               start_token="[MODIFY]", end_token="[/MODIFY]", few_shot=True,
                               base_url=BASE_URL_GPT,
                               model_name="gpt-4-turbo-128k", connect_repeat_time=2, modify_repeat_time=2):
    tmp_input_procs = input_process
    inter_to_LLM_prompts = generate_prompt_for_modify(input_sentences=tmp_input_procs,
                                                      start_token=start_token,
                                                      end_token=end_token,
                                                      few_shot=few_shot)
    # print(inter_to_LLM_prompts)
    errs = None
    modified_proc_items = []
    for jj in range(modify_repeat_time):
        try:
            modified_proc_items_res = model_connect(prompt=inter_to_LLM_prompts,
                                                    base_url=base_url,
                                                    model_name=model_name, repeat_time=connect_repeat_time)

            # if modified_proc_items:
            #     print("Done")

            re_start_token = format_token_for_re(start_token)
            re_end_token = format_token_for_re(end_token)

            modified_proc_items = extract_format_comp(prop=modified_proc_items_res, start_token=re_start_token,
                                                      end_token=re_end_token,
                                                      raw_start_token=start_token,
                                                      raw_end_token=end_token)
        except Exception as ee01:
            print(ee01)
            errs = ee01
            modified_proc_items = []

        if not errs:
            if (len(modified_proc_items) > 0):
                break
            else:
                time.sleep(1.5)
        else:
            time.sleep(1.5)

    return modified_proc_items


import time
import threading
from concurrent.futures import ThreadPoolExecutor,as_completed
from tqdm import tqdm

def generate_reasoning_process_faithful_test_set(eval_df, source_reasonings,
                                                 aim_questions,
                                                 question_col="generate_question",
                                                 reference_col="reference_body2",
                                                 reasoning_type="steps",
                                                 answer_col="manual_answer",
                                                 depth=1,
                                                 method=0,
                                                 raw_reasoning_col="initial_reasoning",
                                                 metric="bleurt",
                                                 top_start="**",
                                                 top_end="**",
                                                 first_token="\n", second_token="。",
                                                 modify_template="[PROMPT] [DEMO] [INSTRUCTION]",
                                                 modify_start_token="[MODIFY]", modify_end_token="[/MODIFY]",
                                                 modify_few_shot=True,
                                                 base_url="",
                                                 model_name="gpt-4-turbo-128k",
                                                 repeat_time=2, worker_num=2,modify_repeat_time=2,is_super=True,is_replay=True):
    '''
    Divided the generated reasoning process into a group of thoughts or statements.
    Take MASK/MODIFY/HOLD operations.
    This function construct the list of splited reasoning thoughts for each reasoning process to realize its scoring

    method: 0 MASK, 1: MODIFY claims to that of contradicted semantic,2: HOLD: No changes
    depth: 0:  MASK/MODIFY for the complete thought，1: MASK/MODIFY for each line of thoughts; 2: MASK/MODIFY for each statement of thoughts
    base_url: the url to connect with GPT models. Beacaus of anonymity requirement, we do not provide the specific url
    '''
    results = {}
    if isinstance(aim_questions, str):
        aim_questions = [aim_questions]

    input_eval_pdf = eval_df[eval_df[question_col].isin(aim_questions)].copy().reset_index(drop=True)
    modify_reasoning_sources = source_reasonings[reasoning_type].copy()
    aim_modify_reasoning_sources = {}
    for aq in aim_questions:
        aim_modify_reasoning_sources[aq] = {}
        aim_modify_reasoning_sources[aq]["reasoning"] = modify_reasoning_sources[aq][:]
        aim_modify_reasoning_sources[aq][question_col] = aq
        reference_bodys = input_eval_pdf[input_eval_pdf[question_col].isin([aq])][reference_col].values[0]
        raw_answer = input_eval_pdf[input_eval_pdf[question_col].isin([aq])][answer_col].values[0]
        aim_modify_reasoning_sources[aq][reference_col] = reference_bodys
        aim_modify_reasoning_sources[aq][answer_col] = raw_answer
        if not is_super and not is_replay:
            raw_reasoning_answer = input_eval_pdf[input_eval_pdf[question_col].isin([aq])][f"{raw_reasoning_col}_answer"].values[0]
            aim_modify_reasoning_sources[aq]["forward_reasoning_answer"] = raw_reasoning_answer

    if method <= 0:
        method = 0

    elif method >= 2:
        method = 2
    else:
        method = 1


    test_reasoning_data = {}

    for aq in aim_modify_reasoning_sources.keys():
        tmp_reasoning_list = aim_modify_reasoning_sources[aq]["reasoning"][:]
        tmp_split_reasoning_list = []
        for tr in tmp_reasoning_list:
            tmp_split_reasoning = split_step_thinking(tr, depth=depth)
            if tmp_split_reasoning:
                tmp_split_reasoning_list.append(tmp_split_reasoning.copy())

        split_sentence_meta = {"step_title": [], "content": [], "step_id": [], "line_id": [], "sen_id": []}

        for tsr_id in range(len(tmp_split_reasoning_list)):
            tsr = tmp_split_reasoning_list[tsr_id]
            if depth < 2:
                for item_id in range(len(tsr["content"])):
                    split_sentence_meta["step_title"].append(tsr["top"])
                    split_sentence_meta["content"].append(tsr["content"][item_id])
                    split_sentence_meta["line_id"].append(item_id)
                    split_sentence_meta["sen_id"].append(-1)
                    split_sentence_meta["step_id"].append(tsr_id)
            else:
                for item_id in range(len(tsr["content"])):
                    item = tsr[item_id]
                    for sen_id in range(len(item)):
                        split_sentence_meta["step_title"].append(tsr["top"])
                        split_sentence_meta["content"].append(tsr["content"][item_id][sen_id])
                        split_sentence_meta["line_id"].append(item_id)
                        split_sentence_meta["sen_id"].append(sen_id)
                        split_sentence_meta["step_id"].append(tsr_id)

        split_sentence_meta_pdf = pd.DataFrame(split_sentence_meta)
        test_reasoning_data[aq] = {}
        test_reasoning_data[aq]["frame"] = split_sentence_meta_pdf
        test_reasoning_data[aq]["source"] = tmp_split_reasoning_list[:]

    final_output_test_set = pd.DataFrame()

    def call_for_generate_modified_reasoning(input_aq_test_reasoning, ind):
        try:
            modified_aq_test_reasoning = input_aq_test_reasoning[:]
            modifying_reasoning_step = modified_aq_test_reasoning[ind].copy()
            modifying_content = modifying_reasoning_step["content"][:]
            modifying_content_str = aggr_splited_steps(modifying_content).strip()
            modifying_content_str = modifying_content_str + "\n"

            modified_content = modify_item_by_calling_LLM(
                input_process=modifying_content_str,
                template=modify_template,
                start_token=modify_start_token,
                end_token=modify_end_token,
                few_shot=modify_few_shot,
                base_url=base_url,
                model_name=model_name,
                connect_repeat_time=repeat_time,
                modify_repeat_time=modify_repeat_time)

            if isinstance(modified_content, list):
                if len(modified_content) > 0:
                    modifying_reasoning_step["content"] = modified_content[:]
                    modified_content_str = aggr_splited_steps(modified_content).strip()
                else:
                    modifying_reasoning_step["content"] = ["Unknown"]
                    modified_content_str = "Unknown"

            else:
                # if modified_content:
                modified_content_str = modified_content.strip()
                if modified_content_str:
                    modifying_reasoning_step["content"] = [modified_content]
                else:
                    modifying_reasoning_step["content"] = ["Unknown"]
                    modified_content_str = "Unknown"

            raw_aq_test_reasoning = input_aq_test_reasoning[:]

            modified_aq_test_reasoning = replace_item_in_list(raw_list=modified_aq_test_reasoning[:], index=ind,
                                                              new_value=modifying_reasoning_step)

            single_test_input_reasoning = ""
            for maqtr_id in range(len(modified_aq_test_reasoning)):
                maqtr = modified_aq_test_reasoning[maqtr_id]
                maqtr_input_data = construct_modified_step(input_step_params=maqtr, give_id=maqtr_id+1, top_start="",
                                                           top_end="")
                maqtr_input_data = maqtr_input_data + "\n"
                single_test_input_reasoning += maqtr_input_data

            single_test_raw_reasoning = ""
            for raqtr_id in range(len(raw_aq_test_reasoning)):
                raqtr = raw_aq_test_reasoning[raqtr_id]
                raqtr_input_data = construct_modified_step(input_step_params=raqtr, give_id=raqtr_id+1, top_start="",
                                                           top_end="")
                raqtr_input_data = raqtr_input_data + "\n"
                single_test_raw_reasoning += raqtr_input_data

            with modify_lock:
                out_test_results.at[ind, "step_id"] = ind
                out_test_results.at[ind, "reasoning_input"] = single_test_input_reasoning
                out_test_results.at[ind, "modified_content"] = modified_content_str
                out_test_results.at[ind, "hist_reasoning"] = single_test_raw_reasoning
                progress.update(1)
        except Exception as ee:
            print(ee)

        return

    def call_for_generate_modified_reasoning_fine_grained(input_aq_test_reasoning, step_id, line_id, total_ind):
        try:
            modified_line_test_reasoning = input_aq_test_reasoning[:]

            raw_aq_test_reasoning = input_aq_test_reasoning[:]

            modifying_reasoning_step = modified_line_test_reasoning[step_id].copy()

            modifying_content = modifying_reasoning_step["content"][line_id]

            if isinstance(modifying_content, list):
                modifying_content_str = aggr_splited_steps(modifying_content).strip()
            else:
                modifying_content_str = modifying_content.strip()

            modifying_content_str = modifying_content_str + "\n"

            modified_content = modify_item_by_calling_LLM(
                input_process=modifying_content_str,
                template=modify_template,
                start_token=modify_start_token,
                end_token=modify_end_token,
                few_shot=modify_few_shot,
                base_url=base_url,
                model_name=model_name,
                connect_repeat_time=repeat_time,
                modify_repeat_time=modify_repeat_time)

            if isinstance(modified_content, list):
                modified_content_str = aggr_splited_steps(modified_content).strip()
            else:
                modified_content_str = modified_content.strip()

            tmp_modifying_reasoning_step_content = modifying_reasoning_step["content"][:]

            tmp_modifying_reasoning_step_content = replace_item_in_list(
                raw_list=tmp_modifying_reasoning_step_content[:], index=line_id, new_value=modified_content_str)
            modifying_reasoning_step["content"] = tmp_modifying_reasoning_step_content[:]
            modified_line_test_reasoning = replace_item_in_list(raw_list=modified_line_test_reasoning[:], index=step_id,
                                                                new_value=modifying_reasoning_step.copy())

            single_test_input_reasoning = ""
            for maqtr_id in range(len(modified_line_test_reasoning)):
                maqtr = modified_line_test_reasoning[maqtr_id]
                maqtr_input_data = construct_modified_step(input_step_params=maqtr, give_id=maqtr_id+1, top_start="",
                                                           top_end="")
                maqtr_input_data = maqtr_input_data + "\n"
                single_test_input_reasoning += maqtr_input_data

            single_test_raw_reasoning = ""
            for raqtr_id in range(len(raw_aq_test_reasoning)):
                raqtr = raw_aq_test_reasoning[raqtr_id]
                raqtr_input_data = construct_modified_step(input_step_params=raqtr, give_id=raqtr_id+1, top_start="",
                                                           top_end="")
                raqtr_input_data = raqtr_input_data + "\n"
                single_test_raw_reasoning += raqtr_input_data

            with modify_lock:
                out_test_results.at[total_ind, "step_id"] = step_id
                out_test_results.at[total_ind, "line_id"] = line_id
                out_test_results.at[total_ind, "reasoning_input"] = single_test_input_reasoning
                out_test_results.at[total_ind, "modified_content"] = modified_content_str
                progress.update(1)
        except Exception as ee:
            print(ee)

        return

    if method <= 0:
        for aq in aim_modify_reasoning_sources.keys():
            aq_test_reasoning_data = test_reasoning_data[aq]["source"][:]

            if depth < 1:
                out_test_results = {"step_id": [], "reasoning_input": [], "hist_reasoning": []}
                for j in range(len(aq_test_reasoning_data)):

                    tmp_aq_test_reasoning = aq_test_reasoning_data[:]
                    raw_aq_test_reasoning = tmp_aq_test_reasoning[:]

                    masked_aq_test_reasoning = drop_item(input_procs=tmp_aq_test_reasoning[:], idx=j)

                    single_test_input_reasoning = ""
                    for maqtr_id in range(len(masked_aq_test_reasoning)):
                        maqtr = masked_aq_test_reasoning[maqtr_id]
                        maqtr_input_data = construct_modified_step(input_step_params=maqtr, give_id=maqtr_id+1,
                                                                   top_start="", top_end="")
                        maqtr_input_data = maqtr_input_data + "\n"
                        single_test_input_reasoning += maqtr_input_data

                    single_test_raw_reasoning = ""
                    for raqtr_id in range(len(raw_aq_test_reasoning)):
                        raqtr = raw_aq_test_reasoning[raqtr_id]
                        raqtr_input_data = construct_modified_step(input_step_params=raqtr, give_id=raqtr_id+1,
                                                                   top_start="", top_end="")
                        raqtr_input_data = raqtr_input_data + "\n"
                        single_test_raw_reasoning += raqtr_input_data

                    out_test_results["step_id"].append(j)
                    out_test_results["reasoning_input"].append(single_test_input_reasoning)
                    out_test_results["hist_reasoning"].append(single_test_raw_reasoning)

                out_test_results = pd.DataFrame(out_test_results)

                test_reasoning_data[aq]["frame"] = pd.merge(test_reasoning_data[aq]["frame"], out_test_results,
                                                            on=["step_id"], how="inner")

            elif depth < 2:
                out_test_results = {"step_id": [], "reasoning_input": [], "line_id": [], "hist_reasoning": []}
                for j in range(len(aq_test_reasoning_data)):
                    tmp_aq_test_reasoning = aq_test_reasoning_data[j].copy()

                    # masked_aq_test_reasoning = drop_item(input_procs=tmp_aq_test_reasoning[:],idx=j)

                    for line_id in range(len(tmp_aq_test_reasoning["content"])):
                        masked_line_test_reasoning = aq_test_reasoning_data[:]
                        raw_aq_test_reasoning = aq_test_reasoning_data[:]

                        masked_line_reasoning = masked_line_test_reasoning[j].copy()
                        masked_line_content = drop_item(input_procs=tmp_aq_test_reasoning["content"][:], idx=line_id)
                        masked_line_reasoning["content"] = masked_line_content[:]

                        masked_line_test_reasoning = replace_item_in_list(raw_list=masked_line_test_reasoning[:],
                                                                          index=j, new_value=masked_line_reasoning)

                        single_test_input_reasoning = ""
                        single_test_raw_reasoning = ""

                        for maqtr_id in range(len(masked_line_test_reasoning)):
                            maqtr = masked_line_test_reasoning[maqtr_id]
                            maqtr_input_data = construct_modified_step(input_step_params=maqtr, give_id=maqtr_id+1,
                                                                       top_start="", top_end="")
                            maqtr_input_data = maqtr_input_data + "\n"
                            single_test_input_reasoning += maqtr_input_data

                        for raqtr_id in range(len(raw_aq_test_reasoning)):
                            raqtr = raw_aq_test_reasoning[raqtr_id]
                            raqtr_input_data = construct_modified_step(input_step_params=raqtr, give_id=raqtr_id+1,
                                                                       top_start="", top_end="")
                            raqtr_input_data = raqtr_input_data + "\n"
                            single_test_raw_reasoning += raqtr_input_data

                        out_test_results["step_id"].append(j)
                        out_test_results["line_id"].append(line_id)
                        out_test_results["reasoning_input"].append(single_test_input_reasoning)
                        out_test_results["hist_reasoning"].append(single_test_raw_reasoning)

                out_test_results = pd.DataFrame(out_test_results)
                test_reasoning_data[aq]["frame"] = pd.merge(test_reasoning_data[aq]["frame"], out_test_results,
                                                            on=["step_id", "line_id"], how="inner")


            else:
                continue

            test_reasoning_data[aq]["frame"][reference_col] = aim_modify_reasoning_sources[aq][reference_col]
            test_reasoning_data[aq]["frame"][answer_col] = aim_modify_reasoning_sources[aq][answer_col]
            test_reasoning_data[aq]["frame"][question_col] = aim_modify_reasoning_sources[aq][question_col]
            test_reasoning_data[aq]["frame"]["modified_content"] = ""
            if not is_super and not is_replay:
                test_reasoning_data[aq]["frame"]["forward_reasoning_answer"] = aim_modify_reasoning_sources[aq]["forward_reasoning_answer"]

            if final_output_test_set.shape[0] < 1:
                final_output_test_set = test_reasoning_data[aq]["frame"].copy().reset_index(drop=True)
            else:
                final_output_test_set = pd.concat(
                    [final_output_test_set, test_reasoning_data[aq]["frame"].copy().reset_index(drop=True)],
                    axis=0).reset_index(drop=True)

    elif method >= 2:
        for aq in aim_modify_reasoning_sources.keys():
            aq_test_reasoning_data = test_reasoning_data[aq]["source"][:]

            if depth < 1:
                out_test_results = {"step_id": [], "reasoning_input": [], "hist_reasoning": []}
                for j in range(len(aq_test_reasoning_data)):

                    tmp_aq_test_reasoning = aq_test_reasoning_data[:]
                    raw_aq_test_reasoning = tmp_aq_test_reasoning[:]

                    single_test_raw_reasoning = ""
                    for raqtr_id in range(len(raw_aq_test_reasoning)):
                        raqtr = raw_aq_test_reasoning[raqtr_id]
                        raqtr_input_data = construct_modified_step(input_step_params=raqtr, give_id=raqtr_id+1,
                                                                   top_start="", top_end="")
                        raqtr_input_data = raqtr_input_data + "\n"
                        single_test_raw_reasoning += raqtr_input_data

                    out_test_results["step_id"].append(j)
                    out_test_results["reasoning_input"].append(single_test_raw_reasoning)
                    out_test_results["hist_reasoning"].append(single_test_raw_reasoning)

                out_test_results = pd.DataFrame(out_test_results)

                test_reasoning_data[aq]["frame"] = pd.merge(test_reasoning_data[aq]["frame"], out_test_results,
                                                            on=["step_id"], how="inner")

            elif depth < 2:
                out_test_results = {"step_id": [], "reasoning_input": [], "line_id": [], "hist_reasoning": []}
                for j in range(len(aq_test_reasoning_data)):
                    tmp_aq_test_reasoning = aq_test_reasoning_data[j].copy()

                    # masked_aq_test_reasoning = drop_item(input_procs=tmp_aq_test_reasoning[:],idx=j)

                    for line_id in range(len(tmp_aq_test_reasoning["content"])):
                        masked_line_test_reasoning = aq_test_reasoning_data[:]
                        raw_aq_test_reasoning = aq_test_reasoning_data[:]

                        masked_line_reasoning = masked_line_test_reasoning[j].copy()
                        masked_line_content = drop_item(input_procs=tmp_aq_test_reasoning["content"][:], idx=line_id)
                        masked_line_reasoning["content"] = masked_line_content[:]

                        single_test_raw_reasoning = ""

                        for raqtr_id in range(len(raw_aq_test_reasoning)):
                            raqtr = raw_aq_test_reasoning[raqtr_id]
                            raqtr_input_data = construct_modified_step(input_step_params=raqtr, give_id=raqtr_id+1,
                                                                       top_start="", top_end="")
                            raqtr_input_data = raqtr_input_data + "\n"
                            single_test_raw_reasoning += raqtr_input_data

                        out_test_results["step_id"].append(j)
                        out_test_results["line_id"].append(line_id)
                        out_test_results["reasoning_input"].append(single_test_raw_reasoning)
                        out_test_results["hist_reasoning"].append(single_test_raw_reasoning)

                out_test_results = pd.DataFrame(out_test_results)
                test_reasoning_data[aq]["frame"] = pd.merge(test_reasoning_data[aq]["frame"], out_test_results,
                                                            on=["step_id", "line_id"], how="inner")

            else:
                continue

            test_reasoning_data[aq]["frame"][reference_col] = aim_modify_reasoning_sources[aq][reference_col]
            test_reasoning_data[aq]["frame"][answer_col] = aim_modify_reasoning_sources[aq][answer_col]
            test_reasoning_data[aq]["frame"][question_col] = aim_modify_reasoning_sources[aq][question_col]
            test_reasoning_data[aq]["frame"]["modified_content"] = ""
            if not is_super and not is_replay:
                test_reasoning_data[aq]["frame"]["forward_reasoning_answer"] = aim_modify_reasoning_sources[aq]["forward_reasoning_answer"]

            if final_output_test_set.shape[0] < 1:
                final_output_test_set = test_reasoning_data[aq]["frame"].copy().reset_index(drop=True)
            else:
                final_output_test_set = pd.concat(
                    [final_output_test_set, test_reasoning_data[aq]["frame"].copy().reset_index(drop=True)],
                    axis=0).reset_index(drop=True)
    else:
        modify_lock = threading.Lock()

        for aq in aim_modify_reasoning_sources.keys():
            aq_test_reasoning_data = test_reasoning_data[aq]["source"][:]

            if depth < 1:
                out_test_results = {"step_id": [None] * len(aq_test_reasoning_data),
                                    "reasoning_input": [None] * len(aq_test_reasoning_data),
                                    "modified_content": [None] * len(aq_test_reasoning_data),
                                    "hist_reasoning": [None] * len(aq_test_reasoning_data)}
                out_test_results = pd.DataFrame(out_test_results)
                progress = tqdm(total=len(aq_test_reasoning_data), position=1, leave=True)

                with ThreadPoolExecutor(max_workers=worker_num) as executor_modify:
                    for ind in range(len(aq_test_reasoning_data)):
                        tmp_aq_test_reasoning = aq_test_reasoning_data[:]
                        executor_modify.submit(call_for_generate_modified_reasoning, tmp_aq_test_reasoning[:], ind)

                progress.close()
                test_reasoning_data[aq]["frame"] = pd.merge(test_reasoning_data[aq]["frame"], out_test_results,
                                                            on=["step_id"], how="inner")

            elif depth < 2:
                out_test_results = {"step_id": [], "reasoning_input": [], "line_id": [], "modified_content": [],
                                    "hist_reasoning": []}
                total_out_len = 0
                for j in range(len(aq_test_reasoning_data)):
                    total_out_len += len(tmp_aq_test_reasoning["content"])

                out_test_results = {"step_id": [None] * total_out_len,
                                    "reasoning_input": [None] * total_out_len,
                                    "line_id": [None] * total_out_len,
                                    "modified_content": [None] * total_out_len,
                                    "hist_reasoning": [None] * total_out_len
                                    }

                out_test_results = pd.DataFrame(out_test_results)

                total_ind = 0

                progress = tqdm(total=total_out_len, position=1, leave=True)

                with ThreadPoolExecutor(max_workers=worker_num) as executor_modify:
                    for j in range(len(aq_test_reasoning_data)):
                        tmp_aq_test_reasoning = aq_test_reasoning_data[j].copy()
                        # masked_aq_test_reasoning = drop_item(input_procs=tmp_aq_test_reasoning[:],idx=j)

                        for line_id in range(len(tmp_aq_test_reasoning["content"])):
                            modified_line_test_reasoning_input = aq_test_reasoning_data[:]
                            executor_modify.submit(call_for_generate_modified_reasoning_fine_grained,
                                                   modified_line_test_reasoning_input[:], j, line_id, total_ind)
                            total_ind += 1

                progress.close()
                test_reasoning_data[aq]["frame"] = pd.merge(test_reasoning_data[aq]["frame"], out_test_results,
                                                            on=["step_id", "line_id"], how="inner")


            else:
                continue

            test_reasoning_data[aq]["frame"][reference_col] = aim_modify_reasoning_sources[aq][reference_col]
            test_reasoning_data[aq]["frame"][answer_col] = aim_modify_reasoning_sources[aq][answer_col]
            test_reasoning_data[aq]["frame"][question_col] = aim_modify_reasoning_sources[aq][question_col]
            if not is_super and not is_replay:
                test_reasoning_data[aq]["frame"]["forward_reasoning_answer"] = aim_modify_reasoning_sources[aq]["forward_reasoning_answer"]

            if final_output_test_set.shape[0] < 1:
                final_output_test_set = test_reasoning_data[aq]["frame"].copy().reset_index(drop=True)
            else:
                final_output_test_set = pd.concat(
                    [final_output_test_set, test_reasoning_data[aq]["frame"].copy().reset_index(drop=True)],
                    axis=0).reset_index(drop=True)

    return final_output_test_set