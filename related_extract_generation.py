import time
from prompt_design import NLI_start_token,NLI_end_token,PRE_end_token,PRE_start_token,HYP_end_token,HYP_start_token
from prompt_design import EX_COMP_end_token,EX_COMP_start_token,generate_fine_grained_semantic_compoenments,EXTRACT_RELATION_COMP_PROP
from prompt_design import generate_prompt_for_extract_relation
from connect_to_LLMs import  model_connect
from textual_process import format_token_for_re,extract_format_comp

BASE_URL_GPT = ""


def single_connect_for_related_componment_extracted(premise_sentences,
                                                    hyp_sentences,
                                                    input_prompt=EXTRACT_RELATION_COMP_PROP["entailment"],
                                                    relationship="entailment_ENG",
                                                    pre_start_token=PRE_start_token,
                                                    pre_end_token=PRE_end_token,
                                                    hyp_start_token=HYP_start_token,
                                                    hyp_end_token=HYP_end_token,
                                                    comp_start_token=EX_COMP_start_token,
                                                    comp_end_token=EX_COMP_end_token,
                                                    base_url=BASE_URL_GPT,
                                                    model_name="gpt-4-turbo-128k",
                                                    repeat_time=2):
    tmp_extracted_input = generate_prompt_for_extract_relation(premise_sentences=premise_sentences,
                                                               hyp_sentences=hyp_sentences,
                                                               input_prompt=input_prompt,
                                                               relationship=relationship,
                                                               pre_start_token=pre_start_token,
                                                               pre_end_token=pre_end_token,
                                                               hyp_start_token=hyp_start_token,
                                                               hyp_end_token=hyp_end_token)

    tmp_extracted_res = model_connect(prompt=tmp_extracted_input,
                                      base_url=base_url,
                                      model_name=model_name,
                                      repeat_time=repeat_time)

    use_comp_start = format_token_for_re(comp_start_token)
    use_comp_end = format_token_for_re(comp_end_token)

    tmp_comp_extract_list = extract_format_comp(prop=tmp_extracted_res,
                                                start_token=use_comp_start,
                                                end_token=use_comp_end,
                                                raw_start_token=comp_start_token,
                                                raw_end_token=comp_end_token)

    return tmp_comp_extract_list


def extracted_related_componments_single(premise_sentences, hyp_sentences,
                                         input_prompt=EXTRACT_RELATION_COMP_PROP["entailment"],
                                         relationship="entailment",
                                         pre_start_token=PRE_start_token,
                                         pre_end_token=PRE_end_token,
                                         hyp_start_token=HYP_start_token,
                                         hyp_end_token=HYP_end_token,
                                         comp_start_token=EX_COMP_start_token,
                                         comp_end_token=EX_COMP_end_token,
                                         base_url=BASE_URL_GPT,
                                         model_name="gpt-4-turbo-128k",
                                         connect_repeat_time=2,
                                         nli_repeat_time=2):
    out_extracted_componments = []

    for j in range(nli_repeat_time):
        try:
            extracted_componments = single_connect_for_related_componment_extracted(premise_sentences=premise_sentences,
                                                                                    hyp_sentences=hyp_sentences,
                                                                                    input_prompt=input_prompt,
                                                                                    relationship=relationship,
                                                                                    pre_start_token=pre_start_token,
                                                                                    pre_end_token=pre_end_token,
                                                                                    hyp_start_token=hyp_start_token,
                                                                                    hyp_end_token=hyp_end_token,
                                                                                    comp_start_token=comp_start_token,
                                                                                    comp_end_token=comp_end_token,
                                                                                    base_url=base_url,
                                                                                    model_name=model_name,
                                                                                    repeat_time=connect_repeat_time)
        except Exception as ee1:
            print(ee1)
            extracted_componments = []

        if len(extracted_componments) > 0:
            out_extracted_componments = extracted_componments[:]
            break

        else:
            time.sleep(1.83)
            out_extracted_componments = []

    return out_extracted_componments