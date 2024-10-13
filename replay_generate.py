import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from prompt_design import REPLAY_META_BODY,generate_replay_prompt
from connect_to_LLMs import call_LLM_response_for_prod,model_connect


def generate_replay_answer_set(in_pdf, prompt_body=REPLAY_META_BODY, question_col="generate_question",
                               refer_col="reference_body2",
                               reasoning_col="reasoning_input",
                               out_col="replay_answer", eval_model="qwen72B",
                               replay_thread_num=2):
    '''
    generate the answers under the guideline of reasoning process

    :param in_pdf: input data
    :param prompt_body: crafted prompt for solution generation under guideline
    :param question_col: column name of question context
    :param refer_col: column name of knowledge context
    :param reasoning_col: column name of reasoning process input
    :param out_col: column name of output answer
    :param eval_model: namely of used LLM
    :param replay_thread_num: number of concurrent thread for QA
    :return:
    '''
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

    replay_lock = threading.Lock()

    progress = tqdm(total=tmp_res.shape[0], position=1, leave=True)

    def call_for_generate_replay_out(ind, input_question, input_refer, input_reasoning):
        try:
            tmp_question_input = generate_replay_prompt(reference_body=input_refer,
                                                        question_text=input_question,
                                                        reasoning_text=input_reasoning,
                                                        prompt_body=prompt_body,evo_model_name=eval_model)

            if "gpt" in eval_model:
                print("use gpt")
                summary_out = model_connect(prompt=tmp_question_input,model_name=eval_model)
            else:
                print("use qwen")
                summary_out = call_LLM_response_for_prod(prompts=tmp_question_input)
            if not summary_out:
                summary_out = "Unknown"

        except Exception as ee0:
            print(ee0)
            print(input_question)
            print(prompt_body)
            print(input_refer)
            summary_out = "Unknown"

        # if ind == 0:
        #     print(tmp_question_input)

        #     # print(summary_out[0:30])

        # if ind < 1:
        #     print(summary_out)

        with replay_lock:
            tmp_res.at[ind, out_col] = summary_out
            # print("Done")
            progress.update(1)

        return

    with ThreadPoolExecutor(max_workers=replay_thread_num) as executor_replay:
        for ind, row in tmp_res.iterrows():
            print(ind)

            tmp_question = row[question_col]
            # print(tmp_question)
            tmp_refer = row[refer_col]
            tmp_reasoning = row[reasoning_col]
            executor_replay.submit(call_for_generate_replay_out, ind, tmp_question, tmp_refer, tmp_reasoning)
            time.sleep(1.2)
    progress.close()

    # print(len(generate_prompt_answers))
    # tmp_res[out_col] = pd.Series(generate_prompt_answers)

    if tmp_stand_res.shape[0] > 0:
        tmp_res = pd.concat([tmp_stand_res.reset_index(drop=True), tmp_res.reset_index(drop=True)], axis=0).reset_index(
            drop=True)

    return tmp_res.reset_index(drop=True)