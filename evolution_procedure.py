from evolution_generation import BASE_URL_GPT,BASE_URL_QWEN
from tqdm import tqdm
MASK = 0
MODIFY = 1
HOLD = 2
from prompt_design import  NO_STRUCT,TOTAL_STRUCT,CON_PART_STRUCT
import numpy as np
from construct_reasoning_sources import extract_reasoning_componments
from evolution_generation import generate_evo_results_set
from prompt_design import PART_END,PART_START,STEP_START,STEP_END,ANS_START,ANS_END
from scoring_for_reasoning import scoring_question_reasoning_for_each_step
from prompt_design import generate_reasoning_few_shot,generate_reasoning_generate_prompt
from reasoning_generate import generate_reasong_process_set
from bleurt import score
def evo_reasoning_process_step(scorer_0:score.BleurtScorer,
                               scorer_1:score.BleurtScorer,
                               scorer_2:score.BleurtScorer,
                               scorer_3:score.BleurtScorer,
                               hist_rp_set={},
                               aim_qes=[],
                               question_col="generate_question", answer_col="manual_answer",
                               reference_col="reference_body2", hist_item_limit=10,
                               prompt_template="[PROMPT] [INSTRUCTION]",
                               now_reasoning_col="evo_reasoning_process",
                               faith_col="faith_score_hyb",
                               truth_col="truth_score_hyb",
                               fact_col="fact_score",
                               evo_base_url=BASE_URL_GPT,
                               gpt_model_name="gpt-4-turbo-128k",
                               evo_model_name="gpt-4-turbo-128k",
                               connect_repeat_time=2,
                               evo_repeat_time=2,
                               evo_thread_num=3,
                               is_fact=True,
                               is_truth=True,
                               score_strategy=MASK,
                               faith_metric="nli_bleurt",
                               modify_repeat_time=2,
                               modify_worker_num=3,
                               hist_replay_answer_col="hist_replay_answer",
                               faith_replay_answer_col="replay_answer",
                               replay_thread_num=3,
                               nli_thread_num=3,
                               computing_thread_num=4,
                               nli_repeat_time=2,
                               split_tokens=["\n", "。"],
                               faith_method=2,
                               item_limit=100,
                               add_semantic=True,
                               default_device='/device:GPU:1',
                               scorer_num=4, score_repeat_time=2,
                               add_structure=NO_STRUCT,is_super=True,is_replay=True,
                               use_reliability=True,use_factual=True,use_fidelity=True
                               ):
    '''

    This function realize the evolution iteration for each time of evolving reasoning process
    :param scorer_0:
    :param scorer_1:
    :param scorer_2:
    :param scorer_3:
    :param hist_rp_set:
    :param aim_qes:
    :param question_col:
    :param answer_col:
    :param reference_col:
    :param hist_item_limit:
    :param prompt_template:
    :param now_reasoning_col:
    :param faith_col:
    :param truth_col:
    :param fact_col:
    :param evo_base_url:
    :param gpt_model_name:
    :param evo_model_name:
    :param connect_repeat_time:
    :param evo_repeat_time:
    :param evo_thread_num:
    :param is_fact:
    :param is_truth:
    :param score_strategy:
    :param faith_metric:
    :param modify_repeat_time:
    :param modify_worker_num:
    :param hist_replay_answer_col:
    :param faith_replay_answer_col:
    :param replay_thread_num:
    :param nli_thread_num:
    :param computing_thread_num:
    :param nli_repeat_time:
    :param split_tokens:
    :param faith_method:
    :param item_limit:
    :param add_semantic:
    :param default_device:
    :param scorer_num:
    :param score_repeat_time:
    :param add_structure:
    :param is_super:
    :param is_replay:
    :param use_reliability:
    :param use_factual:
    :param use_fidelity:
    :return:
    '''
    now_upper_global_id = 0
    now_upper_limit_set = []
    if isinstance(aim_qes, str):
        aim_qes = [aim_qes]
    for aq in aim_qes:
        passed_ids = list(hist_rp_set[aq].keys())
        for pd in passed_ids:
            now_upper_limit_set.append(int(pd))
    now_upper_global_id = np.max(now_upper_limit_set)
    now_upper_global_id = now_upper_global_id + 1
    if "gpt" in evo_model_name:
        evo_base_url = BASE_URL_GPT
    else:
        evo_base_url = BASE_URL_QWEN

    now_step_reasoning_results = generate_evo_results_set(hist_rp_set=hist_rp_set,
                                                          aim_qes=aim_qes,
                                                          question_col=question_col,
                                                          answer_col=answer_col,
                                                          reference_col=reference_col,
                                                          hist_item_limit=hist_item_limit,
                                                          prompt_template=prompt_template,
                                                          out_col=now_reasoning_col,
                                                          faith_col=faith_col,
                                                          truth_col=truth_col,
                                                          fact_col=fact_col,
                                                          base_url=evo_base_url,
                                                          model_name=evo_model_name,
                                                          connect_repeat_time=connect_repeat_time,
                                                          evo_repeat_time=evo_repeat_time,
                                                          evo_thread_num=evo_thread_num,
                                                          is_truth=is_truth,is_super=is_super,
                                                          use_reliability=use_reliability,
                                                          use_factual=use_factual,
                                                          use_fidelity=use_fidelity
                                                          )

    now_extracted_step_comps = extract_reasoning_componments(input_pdf=now_step_reasoning_results,
                                                             question_col=question_col,
                                                             reasoning_col=now_reasoning_col,
                                                             part_start=PART_START,
                                                             part_end=PART_END,
                                                             step_start=STEP_START,
                                                             step_end=STEP_END,
                                                             aim_res_part_id=["steps"])

    aim_qes_reasoning_process_test_set, hist_result_pdf, aim_qes_test_set_for_nli, NLI_fact_score_res, NLI_fact_res = scoring_question_reasoning_for_each_step(
        input_df=now_step_reasoning_results,
        in_extracted_componments=now_extracted_step_comps,
        scorer_0=scorer_0,
        scorer_1=scorer_1,
        scorer_2=scorer_2,
        scorer_3=scorer_3,
        aim_qes=aim_qes,
        is_fact=is_fact,
        is_truth=is_truth,
        strategy=score_strategy,
        question_col=question_col,
        reference_col=reference_col,
        reasoning_type="steps",
        answer_col=answer_col,
        faith_metric=faith_metric,
        gpt_base_url=BASE_URL_GPT,
        gpt_model_name=gpt_model_name,
        modify_repeat_time=modify_repeat_time,
        modify_worker_num=modify_worker_num,
        hist_replay_answer_col=hist_replay_answer_col,
        faith_replay_answer_col=faith_replay_answer_col,
        faith_col=faith_col,
        truth_col=truth_col,
        hist_reasoning_col="hist_reasoning_input",
        replay_thread_num=replay_thread_num,
        nli_thread_num=nli_thread_num,
        computing_thread_num=computing_thread_num,
        connect_repeat_time=connect_repeat_time,
        nli_repeat_time=nli_repeat_time,
        split_tokens=split_tokens,
        faith_method=faith_method,
        item_limit=item_limit,
        add_semantic=add_semantic,
        default_device=default_device,
        scorer_num=scorer_num,
        repeat_time=score_repeat_time,
        add_structure=add_structure,
        is_super=is_super,
        is_replay=is_replay,
        evo_model_name=evo_model_name
    )
    for aq in aim_qes:
        if aq not in hist_rp_set.keys():
            hist_rp_set[aq] = {}
        hist_rp_set[aq][now_upper_global_id] = {}
        hist_rp_set[aq][now_upper_global_id]["source"] = aim_qes_reasoning_process_test_set[
            aim_qes_reasoning_process_test_set[question_col].isin([aq])].reset_index(drop=True)
        hist_rp_set[aq][now_upper_global_id]["fact"] = NLI_fact_score_res[
            NLI_fact_score_res[question_col].isin([aq])].reset_index(drop=True)
        # hist_rp_set[aq][now_upper_global_id]["truth_score"] = hist_rp_set[aq][now_upper_global_id]["source"][
        #     truth_col].mean()
        if use_reliability:
            hist_rp_set[aq][now_upper_global_id]["truth_score"] = hist_rp_set[aq][now_upper_global_id]["source"][
                truth_col].mean()
        else:
            if use_fidelity and use_factual:
                hist_rp_set[aq][now_upper_global_id]["truth_score"] = (hist_rp_set[aq][now_upper_global_id]["source"][
                    fact_col].mean() + hist_rp_set[aq][now_upper_global_id]["source"][
                    faith_col].mean())/2
            elif use_factual and not use_fidelity:
                hist_rp_set[aq][now_upper_global_id]["truth_score"] = (hist_rp_set[aq][now_upper_global_id]["source"][
                                                                           fact_col].mean())
            elif not use_factual and use_fidelity:
                hist_rp_set[aq][now_upper_global_id]["truth_score"] = (hist_rp_set[aq][now_upper_global_id]["source"][
                                                                           faith_col].mean())
            else:
                hist_rp_set[aq][now_upper_global_id]["truth_score"] = (hist_rp_set[aq][now_upper_global_id]["source"][
                                                                           faith_col].mean())


    return hist_rp_set.copy(), now_upper_global_id


def generate_inital_test_data(input_pdf, few_shot_input_data, initial_reasoning_col="inital_reasoning",
                              question_col='generate_question', reference_col='reference_body2',
                              answer_col='manual_answer', reasoning_thread_num=4,is_super=True,evo_model_name="gpt-4-turbo-128k",in_few_shot=True):
    '''
    This function initally generate a reasoning process before evolution executes
    :param input_pdf:
    :param few_shot_input_data:
    :param initial_reasoning_col:
    :param question_col:
    :param reference_col:
    :param answer_col:
    :param reasoning_thread_num:
    :param is_super:
    :param evo_model_name:
    :param in_few_shot:
    :return:
    '''
    in_pdf = input_pdf.copy()

    initial_ans_col = f"{initial_reasoning_col}_answer"

    if is_super:
        is_forward=False
    else:
        is_forward=True

    if is_super:
        few_shot_eg = generate_reasoning_few_shot(eval_pdf=few_shot_input_data,
                                                  context_col=reference_col,question_col=question_col,
                                                  answer_col=answer_col,
                                                  is_forward=is_forward,
                                                  is_super=is_super,
                                                  ans_start="",ans_end="",
                                                  step_start=STEP_START,
                                                  step_end=STEP_END,part_start=PART_START,
                                                  part_end=PART_END)
    else:
        few_shot_eg = generate_reasoning_few_shot(eval_pdf=few_shot_input_data,
                                                  context_col=reference_col,question_col=question_col,
                                                  answer_col=initial_ans_col,
                                                  is_forward=is_forward,
                                                  is_super=is_super,
                                                  ans_start=ANS_START, ans_end=ANS_END,
                                                  step_start=STEP_START,
                                                  step_end=STEP_END,
                                                  part_start=PART_START,
                                                  part_end=PART_END)
    if in_few_shot:
        r_qa_prompt = generate_reasoning_generate_prompt(few_shot_exps=few_shot_eg,is_super=is_super,ans_start=ANS_START,ans_end=ANS_END)
    else:
        r_qa_prompt = generate_reasoning_generate_prompt(few_shot_exps="", is_super=is_super,
                                                         ans_start=ANS_START, ans_end=ANS_END)
    # tmp_prompt_body = r_qa_prompt
    out_pdf = generate_reasong_process_set(in_pdf=in_pdf, prompt_body=r_qa_prompt, question_col=question_col,
                                           refer_col=reference_col, answer_col=answer_col,
                                           out_col=initial_reasoning_col,
                                           reasoning_thread_num=reasoning_thread_num,
                                           is_super=is_super,eval_model=evo_model_name)

    print(out_pdf[initial_reasoning_col].values[0])

    out_reasoning_componments = extract_reasoning_componments(input_pdf=out_pdf, question_col=question_col,
                                                              reasoning_col=initial_reasoning_col,
                                                              part_start=PART_START,
                                                              part_end=PART_END,
                                                              step_start=STEP_START,
                                                              step_end=STEP_END,
                                                              aim_res_part_id=["steps"])
    return out_pdf, out_reasoning_componments


def evo_reasoning_procedure(initial_data, few_shot_input_data,
                            scorer_0:score.BleurtScorer,
                            scorer_1:score.BleurtScorer,
                            scorer_2:score.BleurtScorer,
                            scorer_3:score.BleurtScorer,
                            hist_rp_set={}, iter_limit=10,
                            aim_qes=[],
                            question_col="generate_question", answer_col="manual_answer",
                            reference_col="reference_body2", hist_item_limit=10,
                            prompt_template="[PROMPT] [INSTRUCTION]",
                            now_reasoning_col="evo_reasoning_process",
                            faith_col="faith_score_hyb",
                            truth_col="truth_score_hyb",
                            fact_col="fact_score",
                            evo_base_url=BASE_URL_GPT,
                            gpt_model_name="gpt-4-turbo-128k",
                            evo_model_name="gpt-4-turbo-128k",
                            connect_repeat_time=2,
                            evo_repeat_time=2,
                            evo_thread_num=3,
                            is_fact=True,
                            is_truth=True,
                            score_strategy=MASK,
                            faith_metric="nli_bleurt",
                            modify_repeat_time=2,
                            modify_worker_num=3,
                            hist_replay_answer_col="hist_replay_answer",
                            faith_replay_answer_col="replay_answer",
                            replay_thread_num=3,
                            nli_thread_num=3,
                            computing_thread_num=4,
                            nli_repeat_time=2,
                            split_tokens=["\n", "。"],
                            faith_method=2,
                            item_limit=100,
                            add_semantic=True,
                            default_device='/device:GPU:1',
                            scorer_num=4, score_repeat_time=2, add_structure=NO_STRUCT,
                            is_super=True,
                            is_replay=True,use_reliability=True,
                            use_factual=True,use_fidelity=True):
    '''

    The global procedure of reasoning process evolution, including:
    1) the initial generation of reasoning process,
    2) evolution consisting of multiple iterations which assesses current reasoning process and prompts LLMs to generate refined reasoning process
    :param initial_data: the data of knowledge, question, reference and initially generated reasoning process
    :param few_shot_input_data: the input data for few-shot learning of reasoning process generation
    :param scorer_0: BERT model instance 0
    :param scorer_1: BERT model instance 1
    :param scorer_2: BERT model instance 2
    :param scorer_3: BERT model instance 3
    :param his_rp_set: dict to store generated results
    :param iter_limit: the upper limit of iteration number
    :param aim_qes: the list of aim tasks to resolve
    :param question_col: column name of question context
    :param answer_col: column name of reference answer
    :param reference_col: column name of knowledge context
    :param hist_item_limit: the upper limit of considered previously generated reasoning process
    :param prompt_template: the template of prompt for evolving reasoning process in a self-reflection way
    :param now_reasoning_col: the column name of current reasoning process
    :param faith_col: column name of fidelity score
    :param fact_col: column name of factuality score
    :param truth_col: column name of reliability score
    :param evo_base_url: URL to connect LLMs used as generator of EoT
    :param gpt_model_name: the used GPT model name
    :param evo_model_name: name of the LLM used as generator of EoT
    :param connect_repeat_time: repeat time of connect GPT
    :param evo_repeat_time: repeat time to execute evolution in an iteration
    :param evo_thread_num: the number of parallel threads to execute evolution
    :param is_fact: if considering the factuality
    :param is_truth: if using the weighted fidelity score
    :param score_strategy: employing strategy to excluding specific in fidelity scoring: MASK/MODIFY/HOLD
    :param faith_metric: the metric to estimate similarity in fidelity scoring
    :param modify_repeat_time: the number of times to modify thoughts
    :param modify_worker_num: the number of threads to modify thoughts
    :param hist_replay_answer_col: the column name of generated answers under the guideline of complete reasoning process
    :param faith_replay_answer_col: the column name of generated answers under the guideline of reasoning process where a specific thought is excluded
    :param replay_thread_num: the number of thread to answer questions under the guideline
    :param nli_thread_num: the number of thread to execute NLI task
    :param computing_thread_num: the umber of thread to compute the scorings
    :param nli_repeat_time: the repeat time to execute NLI task
    :param split_tokens: the tokens as labels to split the statements in thoughts
    :param item_limit: the upper limit of windows when estimating similarity for a thought
    :param add_semantic: if considering the alignment when estimating similarity
    :param default_device: the default device to deploy BERT model
    :param scorer_num: number of BERT instance
    :param score_repeat_time: repeat time of evaluate performance
    :param add_structure: if considering the window granularity for BLEURT estimation
    :param is_super: if providing the reference answer when evolving reasoning processes
    :param is_replay: if generating answers under the guideline
    :param use_reliability: if considering reliability factor when evolving reasoning processes
    :param use_factual: if considering factuality factor when evolving reasoning processes
    :param use_fidelity: if considering fidelity factor when evolving reasoning processes
    :return: the evolved results and the id of the latest iteration
    '''
    if isinstance(aim_qes, str):
        aim_qes = [aim_qes]

    need_to_inital_qes = []
    for aq in aim_qes:
        if aq not in list(hist_rp_set.keys()):
            need_to_inital_qes.append(aq)

    initial_input_pdf = initial_data[initial_data[question_col].isin(need_to_inital_qes)].reset_index(drop=True)
    print(initial_input_pdf.shape)

    if len(need_to_inital_qes) > 0 and initial_input_pdf.shape[0] > 0:
        try:
            initial_input_pdf, initial_extracted_reasoning_comps = generate_inital_test_data(input_pdf=initial_input_pdf,
                                                                                         few_shot_input_data=few_shot_input_data,
                                                                                         initial_reasoning_col="inital_reasoning",
                                                                                         question_col=question_col,
                                                                                         reference_col=reference_col,
                                                                                         answer_col=answer_col,
                                                                                         reasoning_thread_num=evo_thread_num,
                                                                                         is_super=is_super,
                                                                                         evo_model_name=evo_model_name
                                                                                         )

            initial_item_id = 0
            print(initial_input_pdf.columns)
            print(initial_input_pdf.shape)
            print(initial_extracted_reasoning_comps)

            aim_qes_reasoning_process_test_set, hist_result_pdf, aim_qes_test_set_for_nli, NLI_fact_score_res, NLI_fact_res = scoring_question_reasoning_for_each_step(
                input_df=initial_input_pdf,
                in_extracted_componments=initial_extracted_reasoning_comps,
                scorer_0=scorer_0,
                scorer_1=scorer_1,
                scorer_2=scorer_2,
                scorer_3=scorer_3,
                aim_qes=need_to_inital_qes,
                is_fact=is_fact,
                is_truth=is_truth,
                strategy=score_strategy,
                question_col=question_col,
                reference_col=reference_col,
                reasoning_type="steps",
                answer_col=answer_col,
                faith_metric=faith_metric,
                gpt_base_url=BASE_URL_GPT,
                gpt_model_name=gpt_model_name,
                modify_repeat_time=modify_repeat_time,
                modify_worker_num=modify_worker_num,
                hist_replay_answer_col=hist_replay_answer_col,
                faith_replay_answer_col=faith_replay_answer_col,
                faith_col=faith_col,
                truth_col=truth_col,
                hist_reasoning_col="hist_reasoning_input",
                replay_thread_num=replay_thread_num,
                nli_thread_num=nli_thread_num,
                computing_thread_num=computing_thread_num,
                connect_repeat_time=connect_repeat_time,
                nli_repeat_time=nli_repeat_time,
                split_tokens=split_tokens,
                faith_method=faith_method,
                item_limit=item_limit,
                add_semantic=add_semantic,
                default_device=default_device,
                scorer_num=scorer_num,
                repeat_time=score_repeat_time,
                add_structure=add_structure,
                is_super=is_super,is_replay=is_replay,
                evo_model_name=evo_model_name
            )
        except Exception as ee:
            print(ee)
            initial_input_pdf, initial_extracted_reasoning_comps = generate_inital_test_data(
                input_pdf=initial_input_pdf,
                few_shot_input_data=few_shot_input_data,
                initial_reasoning_col="inital_reasoning",
                question_col=question_col,
                reference_col=reference_col,
                answer_col=answer_col,
                reasoning_thread_num=evo_thread_num,
                is_super=is_super,
                evo_model_name=evo_model_name,in_few_shot=False
                )

            initial_item_id = 0
            print(initial_input_pdf.columns)
            print(initial_input_pdf.shape)
            print(initial_extracted_reasoning_comps)

            aim_qes_reasoning_process_test_set, hist_result_pdf, aim_qes_test_set_for_nli, NLI_fact_score_res, NLI_fact_res = scoring_question_reasoning_for_each_step(
                input_df=initial_input_pdf,
                in_extracted_componments=initial_extracted_reasoning_comps,
                scorer_0=scorer_0,
                scorer_1=scorer_1,
                scorer_2=scorer_2,
                scorer_3=scorer_3,
                aim_qes=need_to_inital_qes,
                is_fact=is_fact,
                is_truth=is_truth,
                strategy=score_strategy,
                question_col=question_col,
                reference_col=reference_col,
                reasoning_type="steps",
                answer_col=answer_col,
                faith_metric=faith_metric,
                gpt_base_url=BASE_URL_GPT,
                gpt_model_name=gpt_model_name,
                modify_repeat_time=modify_repeat_time,
                modify_worker_num=modify_worker_num,
                hist_replay_answer_col=hist_replay_answer_col,
                faith_replay_answer_col=faith_replay_answer_col,
                faith_col=faith_col,
                truth_col=truth_col,
                hist_reasoning_col="hist_reasoning_input",
                replay_thread_num=replay_thread_num,
                nli_thread_num=nli_thread_num,
                computing_thread_num=computing_thread_num,
                connect_repeat_time=connect_repeat_time,
                nli_repeat_time=nli_repeat_time,
                split_tokens=split_tokens,
                faith_method=faith_method,
                item_limit=item_limit,
                add_semantic=add_semantic,
                default_device=default_device,
                scorer_num=scorer_num,
                repeat_time=score_repeat_time,
                add_structure=add_structure,
                is_super=is_super, is_replay=is_replay,
                evo_model_name=evo_model_name
            )

        for qes in need_to_inital_qes:
            hist_rp_set[qes] = {}
            hist_rp_set[qes][initial_item_id] = {}
            hist_rp_set[qes][initial_item_id]["source"] = aim_qes_reasoning_process_test_set[
                aim_qes_reasoning_process_test_set[question_col].isin([qes])].reset_index(drop=True)
            hist_rp_set[qes][initial_item_id]["fact"] = NLI_fact_score_res[
                NLI_fact_score_res[question_col].isin([qes])].reset_index(drop=True)
            # hist_rp_set[qes][initial_item_id]["truth_score"] = hist_rp_set[qes][initial_item_id]["source"][
            #     truth_col].mean()
            if use_reliability:
                hist_rp_set[qes][initial_item_id]["truth_score"] = hist_rp_set[qes][initial_item_id]["source"][
                    truth_col].mean()
            else:
                if use_fidelity and use_factual:
                    hist_rp_set[qes][initial_item_id]["truth_score"] = (hist_rp_set[qes][initial_item_id][
                                                                               "source"][
                                                                               fact_col].mean() +
                                                                           hist_rp_set[qes][initial_item_id][
                                                                               "source"][
                                                                               faith_col].mean()) / 2
                elif use_factual and not use_fidelity:
                    hist_rp_set[qes][initial_item_id]["truth_score"] = (
                        hist_rp_set[qes][initial_item_id]["source"][
                            fact_col].mean())
                elif not use_factual and use_fidelity:
                    hist_rp_set[qes][initial_item_id]["truth_score"] = (
                        hist_rp_set[qes][initial_item_id]["source"][
                            faith_col].mean())
                else:
                    hist_rp_set[qes][initial_item_id]["truth_score"] = (
                        hist_rp_set[qes][initial_item_id]["source"][
                            faith_col].mean())

    progress = tqdm(total=iter_limit, position=1, leave=True)
    now_upper_limit_set = []
    for aq in aim_qes:
        passed_ids = list(hist_rp_set[aq].keys())
        for pd in passed_ids:
            now_upper_limit_set.append(int(pd))
    now_upper_global_id = np.max(now_upper_limit_set)
    global_item_id = now_upper_global_id

    for iter_id in range(iter_limit):
        try:
            iter_hist_rp_set, iter_global_item_id = evo_reasoning_process_step(
                hist_rp_set=hist_rp_set,
                aim_qes=aim_qes,
                scorer_0=scorer_0,
                scorer_1=scorer_1,
                scorer_2=scorer_2,
                scorer_3=scorer_3,
                question_col=question_col,
                answer_col=answer_col,
                reference_col=reference_col,
                hist_item_limit=hist_item_limit,
                prompt_template=prompt_template,
                now_reasoning_col=now_reasoning_col,
                faith_col=faith_col,
                truth_col=truth_col,
                fact_col=fact_col,
                evo_base_url=evo_base_url,
                gpt_model_name=gpt_model_name,
                evo_model_name=evo_model_name,
                connect_repeat_time=connect_repeat_time,
                evo_repeat_time=evo_repeat_time,
                evo_thread_num=evo_thread_num,
                is_fact=is_fact,
                is_truth=is_truth,
                score_strategy=score_strategy,
                faith_metric=faith_metric,
                modify_repeat_time=modify_repeat_time,
                modify_worker_num=modify_worker_num,
                hist_replay_answer_col=hist_replay_answer_col,
                faith_replay_answer_col=faith_replay_answer_col,
                replay_thread_num=replay_thread_num,
                nli_thread_num=nli_thread_num,
                computing_thread_num=computing_thread_num,
                nli_repeat_time=nli_repeat_time,
                split_tokens=split_tokens,
                faith_method=faith_method,
                item_limit=item_limit,
                add_semantic=add_semantic,
                default_device=default_device,
                scorer_num=scorer_num,
                score_repeat_time=score_repeat_time,
                add_structure=add_structure,
                is_super=is_super,is_replay=is_replay,
                use_reliability=use_reliability,
                use_factual=use_factual,
                use_fidelity=use_fidelity
            )

            hist_rp_set = iter_hist_rp_set.copy()
            global_item_id = iter_global_item_id
            progress.update(1)

        except Exception as ee:
            print(ee)
    progress.close()

    return hist_rp_set, global_item_id