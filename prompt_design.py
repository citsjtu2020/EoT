import re
exp_question='Which film came out first, Una Prostituta Al Servizio Del Pubblico E In Regola Con Le Leggi Dello Stato or The Bag Man?'
ANS_START = "[ANS]"
ANS_END = "[/ANS]"
from construct_reasoning_sources import construct_modified_step

NO_STRUCT = 0
SPLITED_STRUCT = 1
CON_PART_STRUCT = 2
TOTAL_STRUCT = 3

# This file presents all the designed prompt template for NLI, solution generation, producing reasoning process and reasoning thought evolution.
# Considering the anonymity requirement, some few-shot examples related to production knowledge is not provided



exp_question='Which film came out first, Una Prostituta Al Servizio Del Pubblico E In Regola Con Le Leggi Dello Stato or The Bag Man?'


def generate_reasoning_few_shot(eval_pdf, template="Input:\n [Documents] [Qeustion] [Answer] Output:\n [Reasoning]",
                                question_col="generate_question",
                                answer_col="manual_answer",
                                context_col="", start_list={
            exp_question: "When answering the question of Which film came out first, Una Prostituta Al Servizio Del Pubblico E In Regola Con Le Leggi Dello Stato or The Bag Man, I will extract key information from the provided Documents. Here is the STEP-BY-STEP reasoning process:"},
                                step_list={exp_question: [
                                    '**Identify the Problem**：\n To determine which film was released first, we need to compare the release dates of "UnaProstituta Al Servizio Del Pubblico E In Regola Con Le Leggi Dello Stato" and "The Bag Man."',
                                    '**Extract Release Dates**: \n From the provided documents, we can gather the release year for both films: \n - "UnaProstituta Al Servizio Del Pubblico E In Regola Con Le Leggi Dello Stato" is mentioned to be a 1970 film.\n - "The Bag Man" is a 2014 film, as indicated by the cast list mentioning its premiere in 2014.',
                                    '**Compare the Years**: \n Comparing the years, 1970 comes before 2014.',
                                    '**Generate the Conclusion**: \n Therefore, "UnaProstituta Al Servizio Del Pubblico E In Regola Con Le Leggi Dello Stato" was released before "The Bag Man."',
                                    ]}, summ_list={
            exp_question: 'In summary, the Answer is based on an understanding of the information extracted from the provided documents, i.e., finding out the released date of "UnaProstituta Al Servizio Del Pubblico E In Regola Con Le Leggi Dello Stato" (1970 ) and "The Bag Man" (2014) respectively. Then the released dates are compared to solve such problems.'},
                                part_start="[PART]",
                                part_end="[/PART]",
                                step_start="[STEP]",
                                step_end="[/STEP]",
                                ans_start="[ANS]",
                                ans_end="[/ANS]", is_super=True, is_forward=False):
    start_k = list(start_list.keys())
    step_k = list(step_list.keys())
    summ_k = list(summ_list.keys())

    tmp_set_1 = set(start_k).intersection(set(step_k))
    final_set = list(tmp_set_1.intersection(set(summ_k)))
    final_start_list = {}
    final_step_list = {}
    final_summ_list = {}
    for fs in final_set:
        final_start_list[fs] = start_list[fs]
        final_step_list[fs] = step_list[fs]
        final_summ_list = summ_list[fs]

    demo_size = len(final_set)
    out_res = ""
    for i in range(demo_size):
        tmp_out_meta = f"Demonstration {i + 1}:\n"
        tmp_qs = final_set[i]
        tmp_eval_pdf = eval_pdf[eval_pdf[question_col].isin([tmp_qs])]
        # print(tmp_eval_pdf)

        question_res = f"Question: {tmp_qs}\n"
        if not context_col:
            context_res = ""
        else:
            context = tmp_eval_pdf[context_col].values[0]
            context_res = f"Documents:\n {context}\n"
        answer = tmp_eval_pdf[answer_col].values[0]
        if ans_start and ans_end:
            answer_res = f"Answer: {ans_start}\n{answer}\n{ans_end}\n"
        else:
            answer_res = f"Answer: {answer}\n"

        # tmp_start_res = f"{part_start}\n"
        # tmp_start_res += f"{start_list[tmp_qs]}\n"
        # tmp_start_res += f"{part_end}\n"
        # tmp_out_res += f"{part_start}\n"
        tmp_total_step_res = f"{part_start}\n"
        for j in range(len(step_list[tmp_qs])):
            tmp_step = f"{step_start}\n"
            tmp_step += f"{j + 1}. {step_list[tmp_qs][j]}\n"
            tmp_step += f"{step_end}\n"
            tmp_total_step_res += tmp_step
        tmp_total_step_res += f"{part_end}\n"

        # tmp_summ_res = f"{part_start}\n"
        # tmp_summ_res += f"{summ_list[tmp_qs]}\n"
        # tmp_summ_res += f"{part_end}\n"
        # + tmp_start_res + tmp_summ_res

        tmp_reasoning_res = "Reasoning Process: " + tmp_total_step_res

        if is_super:
            tmp_out_res = "Input: \n" + context_res + "\n" + question_res + "\n" + answer_res + "\n" + "Output: \n" + tmp_reasoning_res
        else:
            tmp_out_res = "Input: \n" + context_res + "\n" + question_res + "\n" + "Output: \n" + tmp_reasoning_res
            if is_forward:
                tmp_out_res = tmp_out_res + "\n" + answer_res

        # tmp_out_res = "Input: \n" + context_res + "\n" + question_res + "\n" + answer_res + "\n" + "Output: \n" + tmp_reasoning_res
        tmp_out_res += "\n"
        tmp_out_res = tmp_out_meta + tmp_out_res
        out_res += tmp_out_res

    return out_res


def generate_reasoning_generate_prompt(pro_template="[INSCTRUCT] [Demo]", part_start="[PART]", part_end="[/PART]",
                                       step_start="[STEP]", step_end="[/STEP]", top_start="**", top_end="**",
                                       few_shot_exps="", is_super=True, ans_start="[ANS]", ans_end="[/ANS]"):
    if is_super:
        # (3) a summary of the overall thought of the reasoning process.
        meta_prompt = "Your task is to observe the given Documents, Question, Answer, and try to provide what you think is the Reasoning Process to obtain the provided Answer when answering the Question based on the provided Documents.\n Please note:\n 1.Please generate the reasoning process think step by step. In this process, reflect your thinking mode or logic.\n 2.The reasoning process needs to explain your thoughts on why the Answer is generated when answering the Question based on the Document.\n 3.Your task is to explain why the provided Answer is generated, whether the given answer is wrong or correct.\n 4.Please ensure not to answer the Question or provide any optimization that you think can generate a better answer."
        format_prompt = f"Based on the task description and instructions above, please generate the reasoning process step-by-step. The generated reasoning process needs to comply with the following format requirements:\n 1.Each reasoning process consists of one part, namely: (1) the specific content of the STEP-BY-STEP reasoning process. The beginning of each part is marked with {part_start}, and the end is marked with {part_end}.\n 2.For the STEP-BY-STEP reasoning process, the beginning of each STEP is marked with {step_start}, and the end is marked with {step_end}. Each STEP must be numbered and have a title, which begins with {top_start} and ends with {top_end}."
    else:
        meta_prompt = "Your task is to observe the given Documents, Question, and try to provide: 1) What you think is the Reasoning Process that could lead to the correct answer when answering the Question based on the Documents. 2) State the answer obtained when answering the Question based on the Documents using the given Reasoning Process.\n Please note:\n1.Please generate the reasoning process think step by step. In this process, reflect your thinking mode or logic.\n 2.The reasoning process needs to explain your thoughts on why you think the answer is generated when answering the Question based on the Document.\n 3.When generating the Reasoning process, your task is to explain why you would generate the output answer, rather than directly answer or improve your output answer.\n 4.When reasoning and generating the Answer, please ensure to think and deduce the answer as much as possible following the STEP-BY-STEP Thought provided in your reasoning process.\n 5.While generating the Answer, please do not add new content, STEPs, or explanations to the reasoning process, and make sure not to improve upon the provided reasoning process. On this basis, please ensure consistency in the thinking mode or STEP-BY-STEP Thought used in reasoning with your reasoning process. Specifically, ensure that the referencing/citing of Documents and consequently the question answering is consistent with the provided reasoning process."
        format_prompt = f"Based on the task description and instructions above, please generate the reasoning process step-by-step, followed by the Answer generated on the basis of this reasoning process. The generated reasoning process needs to comply with the following format requirements:\n 1.Each reasoning process includes one part, namely: (1) the specific content of the STEP-BY-STEP reasoning process. The beginning of each part is marked with {part_start}, and the end is marked with {part_end}.\n 2.For the STEP-BY-STEP reasoning process, the beginning of each STEP is marked with {step_start}, and the end is marked with {step_end}. Each STEP must be numbered and have a title, which starts with {top_start} and ends with {top_end}.\nThe generated Answer needs to comply with the following format requirements:\n1. The overall start of the generated Answer is marked with {ans_start}, and the end is marked with {ans_end}.\n"

    demonstration = "\nSpecific output format examples are as follows in the Demonstrations:\n\nDemonstrations:\n"
    if few_shot_exps:
        demonstration += few_shot_exps
    else:
        demonstration = ""

    # gen_prompt = re.sub("\[INSCTRUCT\]", f"{meta_prompt}\n{format_prompt}", pro_template)
    # gen_prompt = re.sub("\[Demo\]", demonstration, gen_prompt)
    gen_prompt = f"{meta_prompt}\n{format_prompt} {demonstration}"
    return gen_prompt


REPLAY_META_BODY = "Your task is to observe the currently provided reasoning process, Documents, Question, and guide your reference to the valid content in Documents based on the thinking and answering method provided by this reasoning process, to achieve an answer to the Question (i.e., generating the answer Answer).\nPlease pay attention, in this process, you need to ensure the following requirements:\n(1) In the reasoning process of generating the answer, please ensure as much as possible to think and deduce answers step-by-step according to the provided reasoning process's STEP-BY-STEP Thought;\n(2) Please ensure that in the process of reasoning the answer, do not add new reasoning process or STEPS, and ensure not to improve the provided reasoning process. On this basis, please ensure as much as possible that the thinking method or STEP-BY-STEP Thought used in reasoning is consistent with the provided reasoning process. Specifically, please ensure that the way you refer to/quote Documents and then answer the question is consistent with the provided reasoning process.\n(3) Please pay attention, your task is to refer to or quote the text according to the thinking method of the reasoning process to achieve question answering, and ultimately your output is the generated answer Answer. Therefore, please note, whether the generated answer Answer is right or wrong, try not to offer additional modification suggestions for the answer generated according to the reasoning process's thinking method, nor modify the generated answer."

def generate_replay_prompt(reference_body,
                           question_text,
                           reasoning_text,prompt_body=REPLAY_META_BODY,
                           evo_model_name="qwen"):
    user_prompt = ' The reference materials and documents provided by the user are as follows Documents:\n' + reference_body + '\n The specific user input question Question is: ' + question_text + '.\n Please refer to/quote the provided Documents based on the following provided reasoning process: ' + reasoning_text + '\nFollowing the thinking and answering method of this reasoning process to answer the above question Question.\nYour Answer to the Question is:'
    sys_prompt = 'Hello, you are the intelligent robot for the smart Q&A project. ' + prompt_body
    if "gpt" in evo_model_name:
        template="[PROMPT] [INSTRUCTION]"
        sys_prompt = sys_prompt + "\n"

        # gen_prompt = re.sub("\[PROMPT\]", sys_prompt, template)
        instruction = user_prompt

        # gen_prompt = re.sub("\[INSTRUCTION\]", instruction, gen_prompt)
        gen_prompt = sys_prompt + instruction
        prompts = gen_prompt

    else:
        prompts = [{"role":"system","content":sys_prompt},
               {"role":"user","content": user_prompt}]
    return prompts


NLI_start_token = "[NLI]"
NLI_end_token = "[/NIL]"
PRE_start_token = "[PRE]"
PRE_end_token = "[/PRE]"
HYP_start_token = "[HYP]"
HYP_end_token = "[/HYP]"

NLI_PROMPT_TEMPLATE = {
    "ENG": f"""
            Hello, your task is to implement Natural Language Inference (NLI). Specifically, you are asked to analyze and understand the relationship between two pieces of input natural language texts: (1) PREMISE text and (2) HYPOTHESIS text, to achieve the corresponding semantic classification.\nPlease note: 1. The beginning and end of the input PREMISE text are marked by {PRE_start_token} and {PRE_end_token} respectively, and the PREMISE text that needs to be understood is between {PRE_start_token} and {PRE_end_token}. Similarly, the beginning and end of the HYPOTHESIS text are marked by {HYP_start_token} and {HYP_end_token} respectively, and the HYPOTHESIS text that needs to be understood is between {HYP_start_token} and {HYP_end_token}.\n 2. Your task is to label the relationship between the PREMISE text and the HYPOTHESIS text with a Label, and the legal Label has only the following 3 kinds: (1) entailment; (2) contradiction; (3) neutral.\nNext, we will explain the conditions for outputting these three types of labels: (1) entailment: The meaning or semantics of the HYPOTHESIS text is implied by/contained in/entailed by the meaning or semantics of PREMISE text, that is, the meaning or semantics of the HYPOTHESIS text can be inferred from the PREMISE text; (2) contradiction: The meaning or semantics of the HYPOTHESIS text contradicts the meaning or semantics of the PREMISE text; (3) neutral: Neutral situation, other than entailment and contradiction, that is, the relationship of entailment or contradiction of the HYPOTHESIS text relative to the PREMISE text cannot be inferred. Please note that to judge as a neutral label, the following two conditions must be met at the same time: 1) The meaning or semantics of the HYPOTHESIS text does not contradict the meaning or semantics of the PREMISE text; 2) The meaning or semantics of the HYPOTHESIS text is not implied by (nor contained in) the meaning or semantics of the PREMISE text, and the meaning of the HYPOTHESIS text cannot be inferred from the PREMISE text.\n3. For the input PREMISE text and HYPOTHESIS text, please understand their context as a whole to implement the NLI task, and according to the understanding of the expressed meaning/semantics of the PREMISE text and HYPOTHESIS text, give out the label.\nFor each input, your output needs to adhere to the following format requirements:\n 1. The beginning of each output is marked by {NLI_start_token}, and the end is marked by {NLI_end_token}.\n 2. Please ensure that the output label Label is a legal Label, that is, it comes from one of the above three labels (i.e., (1) entailment; (2) contradiction; (3) neutral)\nThe specific output format example is as shown in the following Demonstrations:\n\n
            Demonstrations:\n
            Demonstration 1: Input: PREMISE text: {PRE_start_token} Actively promoting rural reform {PRE_end_token} HYPOTHESIS text: {HYP_start_token} Stop rural reform {HYP_end_token} \n Output: {NLI_start_token}\n contradiction\n{NLI_end_token}\n
            Demonstration 2: Input: PREMISE text: {PRE_start_token} Well, we need to discuss and discuss {PRE_end_token} HYPOTHESIS text: {HYP_start_token} This is not only up to me {HYP_end_token} \n Output: {NLI_start_token}\n entailment\n{NLI_end_token}\n
            Demonstration 3: Input: PREMISE text: {PRE_start_token} I need to call him, as I am hard to find {PRE_end_token} HYPOTHESIS text: {HYP_start_token} He is my customer {HYP_end_token} \n Output: {NLI_start_token}\n neutral\n{NLI_end_token}\n
            Demonstration 4: Input: PREMISE text: {PRE_start_token} And then, it is mostly about domestic violence {PRE_end_token} HYPOTHESIS text: {HYP_start_token} Legal literacy work is underway {HYP_end_token} \n Output: {NLI_start_token}\n neutral\n{NLI_end_token}\n
            Demonstration 5: Input: PREMISE text: {PRE_start_token} The room is lit, the two are somewhat strangers, and also polite {PRE_end_token} HYPOTHESIS text: {HYP_start_token} The two people in the room are very familiar {HYP_end_token} \n Output: {NLI_start_token}\n contradiction\n{NLI_end_token}\n
            Demonstration 6: Input: PREMISE text: {PRE_start_token} Of course, building with an increased fiscal deficit is not risk-free; the question is whether this money can be well spent {PRE_end_token} HYPOTHESIS text: {HYP_start_token} The purpose of expanding the fiscal deficit is to curb the growth of total social demand {HYP_end_token} \n Output: {NLI_start_token}\n contradiction\n{NLI_end_token}\n
            Demonstration 7: Input: PREMISE text: {PRE_start_token} Never bought drinks or hairtail before, the child was dressed in rags and was famously poor in the courtyard {PRE_end_token} HYPOTHESIS text: {HYP_start_token} The child's parents are nobility, very wealthy {HYP_end_token} \n Output: {NLI_start_token}\n contradiction\n{NLI_end_token}\n
            Demonstration 8: Input: PREMISE text: {PRE_start_token} The nights at Wuqiao are very quiet; soon you can hear the sound of dew falling {PRE_end_token} HYPOTHESIS text: {HYP_start_token} The temperature at night in Wuqiao has dropped compared to the daytime {HYP_end_token} \n Output: {NLI_start_token}\n entailment\n{NLI_end_token}\n
            Demonstration 9: Input: PREMISE text: {PRE_start_token} You also can't prove it, you can't prove that you have been harmed {PRE_end_token} HYPOTHESIS text: {HYP_start_token} It is difficult for you to prove that you have been harmed {HYP_end_token} \n Output: {NLI_start_token}\n entailment\n{NLI_end_token}\n
            Demonstration 10: Input: PREMISE text: {PRE_start_token} Just like, I push you like this, why does my force go downwards? {PRE_end_token} HYPOTHESIS text: {HYP_start_token} I am giving an example to others {HYP_end_token} \n Output: {NLI_start_token}\n entailment\n{NLI_end_token}\n
            Demonstration 11: Input: PREMISE text: {PRE_start_token} To improve the state-owned capital operation budget and increase the proportion of state-owned capital returns of the central enterprises to public finances{PRE_end_token} HYPOTHESIS text: {HYP_start_token} The profits of central enterprises do not need to be turned over to the public finances{HYP_end_token} \n Output: {NLI_start_token}\n contradiction\n{NLI_end_token}\n
            Demonstration 12: Input: PREMISE text: {PRE_start_token} The grain purchase funds were listed as a key priority, and last year a total of 34.5 billion yuan in grain purchase loans was issued{PRE_end_token} HYPOTHESIS text: {HYP_start_token} The emphasis on grain purchase funds as a key priority was the case the year before last{HYP_end_token} \n Output: {NLI_start_token}\n neutral\n{NLI_end_token}\n
            Demonstration 13: Input: PREMISE text: {PRE_start_token} They say that the Forbidden City after the renovation is just like it was during the Qing Dynasty{PRE_end_token} HYPOTHESIS text: {HYP_start_token} The Forbidden City during the Qing Dynasty was the same as it is now{HYP_end_token} \n Output: {NLI_start_token}\n neutral\n{NLI_end_token}\n
            Demonstration 14: Input: PREMISE text: {PRE_start_token} But sometimes I also feel that this could not possibly be a false accusation{PRE_end_token} HYPOTHESIS text: {HYP_start_token} I always feel that there was no false accusation{HYP_end_token} \n Output: {NLI_start_token}\n contradiction\n{NLI_end_token}\n
            Demonstration 15: Input: PREMISE text: {PRE_start_token} But logically, he should have more opportunities, right?{PRE_end_token} HYPOTHESIS text: {HYP_start_token} He has many opportunities in everyone's impression{HYP_end_token} \n Output: {NLI_start_token}\n entailment\n{NLI_end_token}\n
            Demonstration 16: Input: PREMISE text: {PRE_start_token} After seeing her whole story and the entire incident, I feel very sympathetic to this girl; as a woman myself who has also been young, I feel deeply moved{PRE_end_token} HYPOTHESIS text: {HYP_start_token} I am a peer of the girl.{HYP_end_token} \n Output: {NLI_start_token}\n contradiction\n{NLI_end_token}\n
            Demonstration 17: Input: PREMISE text: {PRE_start_token} It\'s better when day breaks{PRE_end_token} HYPOTHESIS text: {HYP_start_token} It is one o\'clock in the middle of the night now{HYP_end_token} \n Output: {NLI_start_token}\n neutral\n{NLI_end_token}\n
            Demonstration 18: Input: PREMISE text: {PRE_start_token} So, I asked my wife to investigate again the next day{PRE_end_token} HYPOTHESIS text: {HYP_start_token} My wife found the previous investigation easy{HYP_end_token} \n Output: {NLI_start_token}\n neutral\n{NLI_end_token}\n
            Demonstration 19: Input: PREMISE text: {PRE_start_token} Six team members from six different countries became heroes in the eyes of people across the world{PRE_end_token} HYPOTHESIS text: {HYP_start_token} These six members are part of the Antarctic expedition team.{HYP_end_token} \n Output: {NLI_start_token}\n neutral\n{NLI_end_token}\n
            Demonstration 20: Input: PREMISE text: {PRE_start_token} The principal and school leaders at that level, right?{PRE_end_token} HYPOTHESIS text: {HYP_start_token} Speaking nervously{HYP_end_token} \n Output: {NLI_start_token}\n entailment\n{NLI_end_token}\n
            Demonstration 21: Input: PREMISE text: {PRE_start_token} Reform and opening up were born together and promote each other{PRE_end_token} HYPOTHESIS text: {HYP_start_token} Reform and opening up cannot be isolated from each other{HYP_end_token} \n Output: {NLI_start_token}\n entailment\n{NLI_end_token}\n
            Demonstration 22: Input: PREMISE text: {PRE_start_token}Zhang Yonghong couldn't help but feel ashamed, thinking: "The fashion of their era is just the leftovers from a few generations ago, they have so much to catch up on in their studies."{PRE_end_token} HYPOTHESIS text: {HYP_start_token}Zhang Yonghong despises the fashion of their era{HYP_end_token} \n Output: {NLI_start_token}\n neutral\n{NLI_end_token}\n
            Demonstration 23: Input: PREMISE text: {PRE_start_token}"Dangal" isn't a commercial film, is it?{PRE_end_token} HYPOTHESIS text: {HYP_start_token}"Dangal" conveys certain connotations{HYP_end_token} \n Output: {NLI_start_token}\n entailment\n{NLI_end_token}\n
             """,

    "CHN": f"""
您好，您的任务是实现自然语言推理（Natural Language Inference）。具体来说请您分析和理解两个输入的自然语言文本 (1) PREMISE文本和 (2)HYPOTHESIS文本之间的关系，实现相应的语义分类鉴别。\n请注意：1. 输入的PREMISE文本的开头和结尾分别由{PRE_start_token}和{PRE_end_token}标注，需要理解的PREMISE文本在{PRE_start_token}和{PRE_end_token}之间。类似地，HYPOTHESIS文本的开头和结尾分别由{HYP_start_token}和{HYP_end_token}标注，需要理解的HYPOTHESIS文本在{HYP_start_token}和{HYP_end_token}之间。\n 2. 您的任务是对PREMISE文本和HYPOTHESIS文本之间的关系给出标签Label，合法的标签Label有且仅有以下3种：(1) entailment; (2) contradiction; (3) neutral。\n接下来，我们分别说明输出这三类标签的条件: (1) entailment: HYPOTHESIS文本的含义或语义蕴含于/包含于PREMISE文本之中，即，通过PREMISE文本可以推断出HYPOTHESIS文本的含义或语义; (2) contradiction: HYPOTHESIS文本的含义或语义与PREMISE文本的含义或语义相矛盾; (3) neutral: 中立的情况，除了entailment和contradiction以外的情况, 即无法推断出HYPOTHESIS相对于PREMISE文本有entailment或contradiction的关系。请注意，判断为neutral标签时需要同时满足以下两个条件: 1) HYPOTHESIS文本的含义或语义与PREMISE文本的含义或语义没有出现矛盾；2)  HYPOTHESIS文本的含义或语义没有蕴含于（也没有包含于）PREMISE文本的含义或语义之中，通过PREMISE文本推断不出HYPOTHESIS文本的含义。\n3. 对于输入的PREMISE文本和HYPOTHESIS文本，请您分别对其上下文进行整体的含义理解以实现NLI任务，请根据对PREMISE文本和HYPOTHESIS文本所表达含义/语义的理解给出标签。\n对于每次的输入，您的输出需要遵守以下格式要求:\n 1.每次输出的开头使用{NLI_start_token}标注，结尾使用{NLI_end_token}标注。\n 2. 请保证输出的标签Label为合法Label，即来自上述三种标签（即(1) entailment; (2) contradiction; (3) neutral）之一\n具体输出格式实例如以下Demonstrations所示:\n\n
Demonstrations:\n
Demonstration 1: Input: PREMISE文本: {PRE_start_token}积极推进农村改革{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}停止农村改革{HYP_end_token} \n Ouput: {NLI_start_token}\n contradiction\n{NLI_end_token}\n
Demonstration 2: Input: PREMISE文本: {PRE_start_token}呃咱们要好好讨论讨论{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}这不是我一个人说了算的{HYP_end_token} \n Ouput: {NLI_start_token}\n entailment\n{NLI_end_token}\n
Demonstration 3: Input: PREMISE文本: {PRE_start_token}我要打给他,因为我比较难找{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}他是我的顾客{HYP_end_token} \n Ouput: {NLI_start_token}\n neutral\n{NLI_end_token}\n
Demonstration 4: Input: PREMISE文本: {PRE_start_token}然后后面大都是讲家暴{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}正在开展法律普及工作{HYP_end_token} \n Ouput: {NLI_start_token}\n neutral\n{NLI_end_token}\n
Demonstration 5: Input: PREMISE文本: {PRE_start_token}房间亮着,两人都有些不认识的,还有些客气{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}房间里的两个人非常熟悉{HYP_end_token} \n Ouput: {NLI_start_token}\n contradiction\n{NLI_end_token}\n
Demonstration 6: Input: PREMISE文本: {PRE_start_token}当然,靠扩大财政赤字搞建设也不是没有风险,问题在于能不能用好这笔钱{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}扩大财政赤字的目的是抑制社会总需求的增长{HYP_end_token} \n Ouput: {NLI_start_token}\n contradiction\n{NLI_end_token}\n
Demonstration 7: Input: PREMISE文本: {PRE_start_token}过去从来没买过饮料,也没买过带鱼,孩子穿得破烂,在院子里穷出了名{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}小孩的父母是贵族,很有钱{HYP_end_token} \n Ouput: {NLI_start_token}\n contradiction\n{NLI_end_token}\n
Demonstration 8: Input: PREMISE文本: {PRE_start_token}邬桥的夜晚,真是要多静有多静,不一会儿,就听见沙沙的下露水声{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}邬桥夜晚的温度相比白天下降了{HYP_end_token} \n Ouput: {NLI_start_token}\n entailment\n{NLI_end_token}\n
Demonstration 9: Input: PREMISE文本: {PRE_start_token}你也没法证明,你没法证明你受到伤害了{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}你很难证明你受到了伤害。{HYP_end_token} \n Ouput: {NLI_start_token}\n entailment\n{NLI_end_token}\n
Demonstration 10: Input: PREMISE文本: {PRE_start_token}就是我看,比如说我这么推您,它怎么着我的劲儿就向下了呢{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}我在给别人举例子{HYP_end_token} \n Ouput: {NLI_start_token}\n entailment\n{NLI_end_token}\n
Demonstration 11: Input: PREMISE文本: {PRE_start_token}完善国有资本经营预算,提高中央企业国有资本收益上缴公共财政比例{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}中央企业的收益都不用上缴公共财政{HYP_end_token} \n Ouput: {NLI_start_token}\n contradiction\n{NLI_end_token}\n
Demonstration 12: Input: PREMISE文本: {PRE_start_token}粮食收购资金被列为必保的重点,去年共发放粮食收购贷款34.5亿元{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}粮食收购资金列为必保的重点是前年的事情{HYP_end_token} \n Ouput: {NLI_start_token}\n neutral\n{NLI_end_token}\n
Demonstration 13: Input: PREMISE文本: {PRE_start_token}他们说故宫新修了以后,说当年清朝就是这样的{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}当年清朝时故宫和现在一样{HYP_end_token} \n Ouput: {NLI_start_token}\n neutral\n{NLI_end_token}\n
Demonstration 14: Input: PREMISE文本: {PRE_start_token}但有时候我也觉得这个没有诬告的可能吗{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}我一直都觉得这个没有诬告{HYP_end_token} \n Ouput: {NLI_start_token}\n contradiction\n{NLI_end_token}\n
Demonstration 15: Input: PREMISE文本: {PRE_start_token}但是按理说他的机会应该多吧{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}他在大家印象中机会多{HYP_end_token} \n Ouput: {NLI_start_token}\n entailment\n{NLI_end_token}\n
Demonstration 16: Input: PREMISE文本: {PRE_start_token}我看了她整个这个故事整个事件之后,我觉得我很同情这个女孩子,自己也是女人,也年轻过,所以感觉感受很深{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}我是女生的同龄人。{HYP_end_token} \n Ouput: {NLI_start_token}\n contradiction\n{NLI_end_token}\n
Demonstration 17: Input: PREMISE文本: {PRE_start_token}等天亮了,倒还好些{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}现在是半夜一点{HYP_end_token} \n Ouput: {NLI_start_token}\n neutral\n{NLI_end_token}\n
Demonstration 18: Input: PREMISE文本: {PRE_start_token}于是让老婆第二天再调查{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}老婆之前做这个调查很轻松{HYP_end_token} \n Ouput: {NLI_start_token}\n neutral\n{NLI_end_token}\n
Demonstration 19: Input: PREMISE文本: {PRE_start_token}来自六个不同国度的六名考察队员成为世界各国人民心目中的英雄{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}这六名队员是南极考察队的。{HYP_end_token} \n Ouput: {NLI_start_token}\n neutral\n{NLI_end_token}\n
Demonstration 20: Input: PREMISE文本: {PRE_start_token}校长校校校校领导那一级的么.{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}说话紧张{HYP_end_token} \n Ouput: {NLI_start_token}\n entailment\n{NLI_end_token}\n
Demonstration 21: Input: PREMISE文本: {PRE_start_token}开放与改革相伴而生、相互促进{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}改革和开放不能相互孤立{HYP_end_token} \n Ouput: {NLI_start_token}\n entailment\n{NLI_end_token}\n
Demonstration 22: Input: PREMISE文本: {PRE_start_token}张永红禁不住惭愧地想:她们这时代的时尚,只不过是前朝几代的零头,她们要补的课实在太多了{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}张永红鄙弃她们这个时代的时尚{HYP_end_token} \n Ouput: {NLI_start_token}\n neutral\n{NLI_end_token}\n
Demonstration 23: Input: PREMISE文本: {PRE_start_token}《摔跤吧爸爸》不是商业片吗{PRE_end_token} HYPOTHESIS文本: {HYP_start_token}《摔跤吧爸爸》表达了一定的内涵{HYP_end_token} \n Ouput: {NLI_start_token}\n entailment\n{NLI_end_token}\n             
    """
}

EX_COMP_start_token = "[COMP]"
EX_COMP_end_token = "[/COMP]"

def generate_premise_data_in_RAG(question_body,reference_body):
    premise_data = 'The specific user input question you need to answer is: '+ question_body+ '\nThe reference materials and documents you need to refer to when answering the above question are as follows Documents:\n' + reference_body+'\n'
    return premise_data

def generate_prompt_for_NLI(premise_sentences,hyp_sentences,input_prompt,template="[PROMPT] [INSTRUCTION]",
                            pre_start_token=PRE_start_token,
                           pre_end_token=PRE_end_token,
                           hyp_start_token=HYP_start_token,
                           hyp_end_token=HYP_end_token,add_semantic=False):

    # gen_prompt = re.sub("\[PROMPT\]", input_prompt, template)

    if not add_semantic:
        instruction = f'Specifically, the sentence PREMISE text for this input is: {pre_start_token}{premise_sentences}{pre_end_token}; The sentence HYPOTHESIS text for this input is: {hyp_start_token}{hyp_sentences}{hyp_end_token} \n Your output NLI label should be:'
    else:
        instruction = f'Specifically, the sentence PREMISE text for this input is: {pre_start_token}{premise_sentences}{pre_end_token}; The sentence HYPOTHESIS text for this input is: {hyp_start_token}{hyp_sentences}{hyp_end_token} \n Please output: 1) Relationship label between the PREMISE text and the HYPOTHESIS text and 2) The necessary textual components extracted from the PREMISE text to support the justification of the relationship label. The output should be as follows:'
    # gen_prompt = re.sub("\[INSTRUCTION\]", instruction, gen_prompt)

    gen_prompt = f"{input_prompt}\n{instruction}"
    return gen_prompt

from textual_process import split_sentences

EX_COMP_start_token = "[COMP]"
EX_COMP_end_token = "[/COMP]"


def generate_fine_grained_semantic_compoenments(input_sentence, split_tokens=["\n", "。"]):
    if isinstance(split_tokens, list):
        aim_split_tokens = split_tokens[:]
    else:
        aim_split_tokens = [split_tokens]
    aim_splited_sentences = [input_sentence]
    for k in range(len(aim_split_tokens)):
        tmp_splited_setences = []
        for aik in range(len(aim_splited_sentences)):
            aim_item = aim_splited_sentences[aik]
            if not aim_split_tokens[k].strip():
                aim_item_splited = split_sentences(input_sentences=aim_item, split_token=aim_split_tokens[k])
            else:
                aim_item_splited = split_sentences(input_sentences=aim_item, split_token=aim_split_tokens[k],
                                                   remove_last_token=True)

            if aik < (len(aim_splited_sentences) - 1) and k > 0:
                tmp_last_aim_item = aim_item_splited.pop(-1)
                if aim_split_tokens[k - 1] not in tmp_last_aim_item:
                    tmp_last_aim_item = f"{tmp_last_aim_item}{aim_split_tokens[k - 1]}"
                aim_item_splited.append(tmp_last_aim_item)

            tmp_splited_setences = tmp_splited_setences + aim_item_splited

        aim_splited_sentences = tmp_splited_setences[:]
    total_splited_sentences = []
    for aim_item in aim_splited_sentences:
        if aim_item:
            total_splited_sentences.append(aim_item)
    return aim_splited_sentences


EXTRACT_RELATION_COMP_PROP = {"CHN": f"""您好，您的任务是分析和理解两个输入的自然语言文本即(1)PREMISE文本和(2)HYPOTHESIS文本的关系,进而实现NLI推断，具体来说您需要实现类别分类并输出关系标签Label，并从PREMISE文本中摘抄出相应的依据文本 Extracted Componments。\n
在本任务中，PREMISE文本与HYPOTHESIS文本的合法关系即候选关系标签Label有且仅有以下3种：(1) entailment; (2) contradiction; (3) neutral。\n
接下来，我们分别说明判断PREMISE文本与HYPOTHESIS文本关系为这3类标签中的某一标签的条件:\n
(1) entailment: HYPOTHESIS文本的含义或语义蕴含于/包含于PREMISE文本之中，即，通过PREMISE文本可以推断出HYPOTHESIS文本的含义或语义; \n
(2) contradiction: HYPOTHESIS文本的含义或语义与PREMISE文本的含义或语义相矛盾; \n
(3) neutral: 中立的情况，除了entailment和contradiction以外的情况, 即无法推断出HYPOTHESIS相对于PREMISE文本有entailment或contradiction的关系。请注意，判断为neutral关系时需要同时满足以下两个条件: 1) HYPOTHESIS文本的含义或语义与PREMISE文本的含义或语义没有出现矛盾；2)  HYPOTHESIS文本的含义或语义没有蕴含于（也没有包含于）PREMISE文本的含义或语义之中，通过PREMISE文本推断不出HYPOTHESIS文本的含义。\n
在您对本次输入的PREMISE文本与HYPOTHESIS文本的关系做出分类并给出标签后，请您从PREMISE文本中抽取可以充分支持您对“PREMISE文本与HYPOTHESIS文本的关系做出判断”的必要的语句或者段落。\n
在理解文本并完成任务时，请注意：1. 输入的PREMISE文本的开头和结尾分别由{PRE_start_token}和{PRE_end_token}标注，需要理解的PREMISE文本在{PRE_start_token}和{PRE_end_token}之间；类似地，HYPOTHESIS文本的开头和结尾分别由{HYP_start_token}和{HYP_end_token}标注，需要理解的HYPOTHESIS文本在{HYP_start_token}和{HYP_end_token}之间；\n
2. 在输出标签的同时，您需要从PREMISE文本中摘抄必要的相关文本，而不要对文本进行修改；\n
3. 请您在保证语义的前提下，尽量只摘抄必要的部分而不要过多的输出原文，同时需要保证您摘抄的部分可以有效支持您对输出标签的分类判断，即支持您对PREMISE文本与HYPOTHESIS文本的关系推断的结果。具体来说，1) 若关系判断为"entailment", 请您从PREMISE文本中摘抄可以充分支持“HYPOTHESIS文本的含义蕴含（entail）于PREMISE文本”这一判断的必要的语句或者段落; 2) 若关系判断为"contradiction"，请您从PREMISE文本中摘抄可以充分支持“HYPOTHESIS文本的含义与PREMISE文本含义相矛盾（contradiction）”这一判断的必要的语句或者段落；3) 若关系判断为"neutral"，请您从PREMISE文本中摘抄可以充分支持“PREMISE文本与HYPOTHESIS文本关系为 neural”这一判断的必要的语句或者段落，同时在判断为"neural"关系时，请您尽量使摘抄的语句或段落的结构或语义与HYPOTHESIS文本的结构或语义具有一定的相似性。\n
在输出时，请注意您的输出结果需要满足以下格式：\n
1) 在输出分类标签Label时，每次输出的标签Label开头使用{NLI_start_token}标注，结尾使用{NLI_end_token}标注。\n 
2) 请保证输出的标签Label为合法Label，即来自上述三种标签（即(1) entailment; (2) contradiction; (3) neutral）之一 \n
3) 在输出从PREMISE中摘抄的文本时，对于摘抄的文本，若该文本与输出的摘抄前后文在原文中不连续则定义为此文本为1条摘要文本。对于每1条摘要文本，请您在它的开头和结尾分别使用{EX_COMP_start_token}和{EX_COMP_end_token}进行标注。
4) 在输出从PREMISE中摘抄的文本时,在保证摘抄部分可以有效支持您的判断的前提下，尽量只摘抄必要的部分而不要过多的输出原文。\n
""",
                              "entailment": f"您好，您的任务是分析和理解两个输入的自然语言文本即(1)PREMISE文本和(2)HYPOTHESIS文本的关系,进而实现NLI推断。\n在本任务中，PREMISE文本与HYPOTHESIS文本的关系有且仅有以下3种：(1) entailment: HYPOTHESIS文本的含义或语义蕴含于/包含于PREMISE文本之中，即，通过PREMISE文本可以推断出HYPOTHESIS文本的含义或语义; (2) contradiction: HYPOTHESIS文本的含义或语义与PREMISE文本的含义或语义相矛盾; (3) neutral: 中立的情况，除了entailment和contradiction以外的情况, 即无法推断出HYPOTHESIS相对于PREMISE文本有entailment或contradiction的关系。请注意，判断为neutral关系时需要同时满足以下两个条件: 1) HYPOTHESIS文本的含义或语义与PREMISE文本的含义或语义没有出现矛盾；2)  HYPOTHESIS文本的含义或语义没有蕴含于（也没有包含于）PREMISE文本的含义或语义之中，通过PREMISE文本推断不出HYPOTHESIS文本的含义。我们已知，本次输入的PREMISE文本与HYPOTHESIS文本的关系为entailment，请您从PREMISE文本中抽取可以充分支持“HYPOTHESIS文本的含义蕴含（entail）于PREMISE文本”这一判断的必要的语句或者段落。\n请注意：1. 输入的PREMISE文本的开头和结尾分别由{PRE_start_token}和{PRE_end_token}标注，需要理解的PREMISE文本在{PRE_start_token}和{PRE_end_token}之间；类似地，HYPOTHESIS文本的开头和结尾分别由{HYP_start_token}和{HYP_end_token}标注，需要理解的HYPOTHESIS文本在{HYP_start_token}和{HYP_end_token}之间；\n2. 您需要从PREMISE文本中摘抄必要的相关文本，而不要对文本进行修改；\n3. 请您在保证语义的前提下，尽量只摘抄必要的部分而不要过多的输出原文，同时需要保证您摘抄的部分可以有效支持“本次输入的PREMISE文本与HYPOTHESIS文本的关系为entailment”这一判断。\n在输出时，请注意您的输出结果需要满足以下格式：1）对于摘抄的文本，若该文本与输出的摘抄前后文在原文中不连续则定义为1条摘要文本。对于每1条摘要文本，请您在它的开头和结尾分别使用{EX_COMP_start_token}和{EX_COMP_end_token}进行标注。",
                              "neural": f"您好，您的任务是分析和理解两个输入的自然语言文本即(1)PREMISE文本和(2)HYPOTHESIS文本的关系,进而实现NLI推断。\n在本任务中，PREMISE文本与HYPOTHESIS文本的关系有且仅有以下3种：(1) entailment: HYPOTHESIS文本的含义或语义蕴含于/包含于PREMISE文本之中，即，通过PREMISE文本可以推断出HYPOTHESIS文本的含义或语义; (2) contradiction: HYPOTHESIS文本的含义或语义与PREMISE文本的含义或语义相矛盾; (3) neutral: 中立的情况，除了entailment和contradiction以外的情况, 即无法推断出HYPOTHESIS相对于PREMISE文本有entailment或contradiction的关系。请注意，判断为neutral关系时需要同时满足以下两个条件: 1) HYPOTHESIS文本的含义或语义与PREMISE文本的含义或语义没有出现矛盾；2)  HYPOTHESIS文本的含义或语义没有蕴含于（也没有包含于）PREMISE文本的含义或语义之中，通过PREMISE文本推断不出HYPOTHESIS文本的含义。我们已知，本次输入的PREMISE文本与HYPOTHESIS文本的关系为neutral，请您从PREMISE文本中抽取可以充分支持“PREMISE文本与HYPOTHESIS文本关系为 neural”这一判断的必要的语句或者段落。\n请注意：1. 输入的PREMISE文本的开头和结尾分别由{PRE_start_token}和{PRE_end_token}标注，需要理解的PREMISE文本在{PRE_start_token}和{PRE_end_token}之间；类似地，HYPOTHESIS文本的开头和结尾分别由{HYP_start_token}和{HYP_end_token}标注，需要理解的HYPOTHESIS文本在{HYP_start_token}和{HYP_end_token}之间；\n2. 您需要从PREMISE文本中摘抄必要的相关文本，而不要对文本进行修改；\n3. 请您在保证语义的前提下，尽量只摘抄必要的部分而不要过多的输出原文，同时需要保证您摘抄的部分可以充分支持“PREMISE文本与HYPOTHESIS文本关系为 neural”这一判断。\n在输出时，请注意您的输出结果需要满足以下格式：1）对于摘抄的文本，若该文本与输出的摘抄前后文在原文中不连续则定义为1条摘要文本。对于每1条摘要文本，请您在它的开头和结尾分别使用{EX_COMP_start_token}和{EX_COMP_end_token}。",
                              "ENG": f"""
Hello, your task is to analyze and understand the relationship between two input natural language texts, namely (1) PREMISE text and (2) HYPOTHESIS text, and then perform NLI inference. Specifically, you need to implement category classification and output the relationship label Label, and extract the corresponding supporting text Extracted Components from the PREMISE text.\n
In this task, the legal relationship between PREMISE text and HYPOTHESIS text, that is, the candidate relationship labels Label, are only the following 3 types: (1) entailment; (2) contradiction; (3) neutral.\n
Next, we separately explain the conditions for judging the relationship between the PREMISE text and the HYPOTHESIS text as one of these three labels:\n
(1) entailment: The meaning or semantics of the HYPOTHESIS text is implied by or included in the PREMISE text, that is, the meaning or semantics of the HYPOTHESIS text can be inferred from the PREMISE text; \n
(2) contradiction: The meaning or semantics of the HYPOTHESIS text is contradictory to the meaning or semantics of the PREMISE text; \n
(3) neutral: A neutral situation, other than entailment and contradiction, that is, it is not possible to infer that the HYPOTHESIS text has a relationship of entailment or contradiction relative to the PREMISE text. Please note that when judging as a neutral relationship, the following two conditions need to be satisfied simultaneously: 1) There is no contradiction between the meaning or semantics of the HYPOTHESIS text and the meaning or semantics of the PREMISE text; 2) The meaning or semantics of the HYPOTHESIS text is not implied in (or included in) the meaning or semantics of the PREMISE text, and the meaning of the HYPOTHESIS text cannot be inferred from the PREMISE text.\n
After you classify the relationship between the input PREMISE text and HYPOTHESIS text and give a label, please extract the necessary sentences or paragraphs from the PREMISE text that can fully support your judgment of "the relationship between the PREMISE text and the HYPOTHESIS text".\n
When understanding the text and completing the task, please note: 1. The beginning and the end of the input PREMISE text are marked by {PRE_start_token} and {PRE_end_token} respectively, and the PREMISE text to be understood is between {PRE_start_token} and {PRE_end_token}; similarly, the beginning and the end of the HYPOTHESIS text are marked by {HYP_start_token} and {HYP_end_token} respectively, and the HYPOTHESIS text to be understood is between {HYP_start_token} and {HYP_end_token};\n
2. When outputting labels, you need to extract the necessary relevant text from the PREMISE text without modifying the text;\n
3. On the premise of ensuring semantics, try to only extract the necessary parts without overly outputting the original text, and ensure that the parts you extract can effectively support your judgment of the output label, that is, support your inference result of the relationship between the PREMISE text and the HYPOTHESIS text. Specifically, 1) If the relationship judgment is "entailment", please extract the necessary sentences or paragraphs from the PREMISE text that can fully support the judgment that "the meaning of the HYPOTHESIS text is contained (entail) within the PREMISE text"; 2) If the relationship judgment is "contradiction", please extract the necessary sentences or paragraphs from the PREMISE text that can fully support the judgment that "the meaning of the HYPOTHESIS text is contradictory (contradiction) to the meaning of the PREMISE text"; 3) If the relationship judgment is "neutral", please extract the necessary sentences or paragraphs from the PREMISE text that can fully support the judgment that "the relationship between the PREMISE text and the HYPOTHESIS text is neural", and when judging the "neural" relationship, please try to make the structure or semantics of the extracted sentence or paragraph similar to the structure or semantics of the HYPOTHESIS text.\n
When outputting, please ensure that your output results meet the following format:\n
1) When outputting the classification label Label, use {NLI_start_token} to mark the beginning of each label Label output and {NLI_end_token} to mark the end.\n
2) Please ensure that the output label Label is a legal Label, that is, it comes from one of the three aforementioned labels (namely (1) entailment; (2) contradiction; (3) neutral).\n
3) When outputting text extracted from the PREMISE, for the extracted text, if the text and the context surrounding the extracted output are discontinuous in the original text, then this text is defined as one summary text. For each summary text, please use {EX_COMP_start_token} at the beginning and {EX_COMP_end_token} at the end to mark it. The specific output format example is shown in the following Demonstrations:\n
4) When outputting text extracted from the PREMISE, on the premise that the extracted part can effectively support your judgment, try to extract only the necessary parts without excessively outputting the original text.\n
""",

                              "entailment_ENG": f"""
Hello, your task is to analyze and understand the relationship between two input natural language texts, namely (1) PREMISE text and (2) HYPOTHESIS text, and then perform NLI inference.\nIn this task, the relationship between PREMISE text and HYPOTHESIS text has only the following 3 kinds: (1) entailment: The meaning or semantics of the HYPOTHESIS text is implied by or contained within the PREMISE text, that is, the meaning or semantics of the HYPOTHESIS text can be inferred from the PREMISE text; (2) contradiction: The meaning or semantics of HYPOTHESIS text is contradictory to the meaning or semantics of the PREMISE text; (3) neutral: The neutral case, other than entailment and contradiction, that is, it is not possible to infer that HYPOTHESIS has an entailment or contradiction relationship with respect to the PREMISE text. Please note that when judging as a neutral relationship, the following two conditions must be satisfied: 1) There is no contradiction between the meaning or semantics of HYPOTHESIS text and PREMISE text; 2) The meaning or semantics of HYPOTHESIS text is not implied by (or contained within) the meaning or semantics of PREMISE text, and the meaning of HYPOTHESIS text cannot be deduced from the PREMISE text. We know that the relationship between the input PREMISE text and HYPOTHESIS text is entailment, please extract the necessary sentences or paragraphs from the PREMISE text that can fully support the judgment that 'The meaning of HYPOTHESIS text is entailed (implied) by the PREMISE text'.\nPlease note: 1. The beginning and end of the input PREMISE text are marked by {PRE_start_token} and {PRE_end_token}, and the PREMISE text to be understood is between {PRE_start_token} and {PRE_end_token}; similarly, the beginning and end of the HYPOTHESIS text are marked by {HYP_start_token} and {HYP_end_token}, and the HYPOTHESIS text to be understood is between {HYP_start_token} and {HYP_end_token};\n2. You need to extract the necessary related text from the PREMISE text without modifying the text;\n3. On the premise of ensuring semantics, try to only extract the necessary parts without overly outputting the original text, and ensure that the part you extract can effectively support the judgment that 'the relationship between the input PREMISE text and HYPOTHESIS text is entailment'.\nWhen outputting, please ensure that your output results meet the following format: 1) For the extracted text, if the text and the context surrounding the extracted output are discontinuous in the original text, then define this text as one summary text. For each summary text, please use {EX_COMP_start_token} at the beginning and {EX_COMP_end_token} at the end to mark it. The specific output format example is shown in the following Demonstrations:\n\nDemonstrations:\nDemonstration 1:\nInput:\nPREMISE text: {PRE_start_token}The specific user input question you need to answer is: ... [translation continues with the rest of the excerpt] ... Please first go to the work item configuration page to enable the task type”, it's hard work to please open it for me{EX_COMP_end_token}\n{EX_COMP_start_token}Creating a subitem -> Adding a subtask error "No available subtask type, please first go to the work item configuration page to enable a task type". User: The creation of demand prompts "Please go to the work item configuration page to enable a demand type".{EX_COMP_end_token}\n{EX_COMP_start_token} Reference title: How to Modify Work Item Types. Reference content: Question: When creating a demand work item type, it is not possible to select other demand types. Answer: Backend settings need to be implemented to enable other demand types. Answer: Hello, when you encounter this prompt, please contact the project administrator to enable the needed demand types in the project 'Settings' - 'Service Management' - 'Demand Configuration' - 'Expand Disabled Types', then you can choose other demands in work item types.{EX_COMP_end_token}\n
""",

                              "neural_ENG": f"""
Hello, your task is to analyze and understand the relationship between two input natural language texts, namely (1) the PREMISE text and (2) the HYPOTHESIS text, and then to perform NLI inference.\nIn this task, the relationship between the PREMISE text and the HYPOTHESIS text is exclusively one of the following three types: (1) entailment: The meaning or semantics of the HYPOTHESIS text is entailed by or included within the PREMISE text, that is, the meaning or semantics of the HYPOTHESIS text can be inferred from the PREMISE text; (2) contradiction: The meaning or semantics of the HYPOTHESIS text contradicts with the meaning or semantics of the PREMISE text; (3) neutral: A neutral situation, other than entailment and contradiction, meaning it cannot be inferred that the HYPOTHESIS has a relationship of entailment or contradiction with the PREMISE text. Please note, to judge a relationship as neutral, the following two conditions must be simultaneously met: 1) The meaning or semantics of the HYPOTHESIS text does not contradict with the meaning or semantics of the PREMISE text; 2) The meaning or semantics of the HYPOTHESIS text is not contained within (nor entailed by) the meaning or semantics of the PREMISE text, thus the HYPOTHESIS text’s meaning cannot be inferred from the PREMISE text. We already know the relationship between the input PREMISE text and HYPOTHESIS text is neutral, please extract from the PREMISE text the necessary sentences or paragraphs that can fully support the judgement that 'the relationship between the PREMISE text and the HYPOTHESIS text is neural'.\nPlease note: 1. The beginning and end of the input PREMISE text are marked by {PRE_start_token} and {PRE_end_token}, with the PREMISE text to be understood lying between {PRE_start_token} and {PRE_end_token}; similarly, the beginning and end of the HYPOTHESIS text are marked by {HYP_start_token} and {HYP_end_token}, with the HYPOTHESIS text to be understood lying between {HYP_start_token} and {HYP_end_token};\n2. You need to excerpt the necessary related text from the PREMISE text, without altering the text;\n3. Please excerpt only the necessary parts without overly outputting the original text while ensuring semantics, and ensure that your excerpted parts can fully support the judgement that 'the relationship between the PREMISE text and HYPOTHESIS text is neural'.\nWhen outputting, please ensure your output meets the following format: 1) For the excerpted text, if it and the text excerpted before and after it in the original are not continuous, it's defined as 1 abstract text. For each abstract text, please use {EX_COMP_start_token} and {EX_COMP_end_token} at its beginning and end, respectively. 
"""
                              }


def generate_prompt_for_extract_relation(premise_sentences, hyp_sentences, input_prompt, relationship="entailment_ENG",
                                         template="[PROMPT] [INSTRUCTION]",
                                         pre_start_token=PRE_start_token,
                                         pre_end_token=PRE_end_token,
                                         hyp_start_token=HYP_start_token,
                                         hyp_end_token=HYP_end_token):
    # gen_prompt = re.sub("\[PROMPT\]", input_prompt, template)
    instruction = f'具体来说，本次输入的PREMISE文本为：{pre_start_token}{premise_sentences}{pre_end_token}; 本次输入的HYPOTHESIS文本为：{hyp_start_token}{hyp_sentences}{hyp_end_token} \n 本次输入的PREMISE文本与HYPOTHESIS文本的为{relationship}, 请您输出从PREMISE文本中摘抄的必要文本以支持\"PREMISE文本与HYPOTHESIS文本的为{relationship}\"这一判断，输出如下:'
    # gen_prompt = re.sub("\[INSTRUCTION\]", instruction, gen_prompt)
    gen_prompt = f"{input_prompt}\n{instruction}"
    return gen_prompt


HIST_START_TOKEN = "[HIS]"
HIST_END_TOKEN = "[/HIS]"

TRUTH_SCORE_START = "[TRUTH]"
TRUTH_SCORE_END = "[/TRUTH]"

FACT_SCORE_START = "[FACT]"
FACT_SCORE_END = "[/FACT]"

FAITH_SCORE_START = "[CONSIST]"
FAITH_SCORE_END = "[/CONSIST]"

OPT_START_TOKEN = "[OPT]"
OPT_END_TOKEN = "[/OPT]"

TOP_START = "**"
TOP_END = "**"

import pandas as pd

PART_START = "[PART]"
PART_END = "[/PART]"

STEP_START = "[STEP]"
STEP_END = "[/STEP]"

HIST_START_TOKEN = "[HIS]"
HIST_END_TOKEN = "[/HIS]"

TRUTH_SCORE_START = "[TRUTH]"
TRUTH_SCORE_END = "[/TRUTH]"

FACT_SCORE_START = "[FACT]"
FACT_SCORE_END = "[/FACT]"

FAITH_SCORE_START = "[FAITHFUL]"
FAITH_SCORE_END = "[/FAITHFUL]"

OPT_START_TOKEN = "[OPT]"
OPT_END_TOKEN = "[/OPT]"

PART_START = "[PART]"
PART_END = "[/PART]"

STEP_START = "[STEP]"
STEP_END = "[/STEP]"

TOP_START = "**"
TOP_END = "**"

EVO_META_PROMPT_CHN_SUPER = {"gpt": f"""您好，您的任务是观察给定的Documents，Question，Answer，并参考提供的History中的reasoning process以及对这些reasoning process的量化评分 (score)，生成您认为的优化后的即可以获得更高分数的reasoning process。具体来说，每一个生成的reasoning process即为您所认为的大模型在根据Documents回答Question时，当大模型的输出可以更好地逼近当前提供的目标Answer时，大模型所需要使用或遵循的思维方式 (thought) 或逻辑推理过程 (Reasoning Process)。
请注意，在生成优化的reasoning process时您需要遵循以下的要求：
1. 请您 think step by step地来生成reasoning process。在这一过程中体现您认为的大模型的思维（thought）方式或思考逻辑。
2. 请您注意，您所实现的reasoning process需要尽量是对History提供的先前生成的reasoning process进行优化后的结果,且不能重复History中的reasoning process。具体来说，我们对于History中提供的每一个reasoning process同时进行了3个维度的量化评分 (score):
(1) truthfulity score: 范围在0~100之间，该分数代表了大模型使用相对应的reasoning process来引导大模型根据提供的Documents回答Question时，其生成的回答结果与提供的目标Answer之间的相似度，分数越高越好。该分数越高则代表使用相应reasoing process时可以越真实的（truthful）引导大模型生成与目标Answer相似的回答。具体来说，该分数越高，相应reasoning prcocess引导大模型生成的结果的文本含义或语义蕴含（entail）于目标Answer的程度越高，且生成结果与目标Answer的语义越接近。
(2) faithfulness score: 范围在0~100之间，对于相应的reasoning process的每一步(each step) 都会给出faithfulness score，越高越好。每一步（each step）的该分数代表了大模型在真实的（truthful）地生成与目标Answer相似的回答时，对该分数的所对应的reasoning prcocess的相应step的忠诚度（faithfulity）。具体来说，该分数越高代表了相应reasoning process的相应step对于逼近目标Answer这一目标的贡献度越高。即，该分数越高代表着大模型在回答问题时忠实地（faithfuly）采用相应step的推理逻辑的程度越高（也就是说相应step越忠实地反映了大模型的思维方式），且因为采用该step的逻辑而带来的逼近目标Answer的收益越大。
(3) factuality score: 范围在0~100之间，对于相应的reasoning process的每一步(each step) 都会给出factuality score，越高越好。每一步（each step）的该分数代表了相应的reasoning prcocess的每一步（each step）所表述内容的事实性（factual）程度。具体来说，该分数越高代表了相应reasoning process的相应step的推理依据来自于对提供的Documents或Question的引用或理解的程度越高，相应地，该step的推理依据来自于大模型自身的猜测概率越低且推理依据脱离提供的Document与Question的概率越低。
综上所述，请您在生成优化的reasoning process时尽量保证该reasoning process整体可以获得更高的truthfulity score。在这一前提下，请您生成的优化的reasoning process尽量提高每一step的faithfulness score和factuality score。同时请注意，您生成的step的数量和标题不需要和History中的reasoning process相同。
3.请使用汉语来输出优化的reasoning process
4.请您注意，生成reasoning process的目的是引导大模型使其提供的答案逼近目标Answer，无论给定的答案是错误的还是正确的。
5.请您确保不要回答问题Question，也不要给目标Answer提供任何优化建议。
6. 在本次输入的History中，每一个提供的先前生成的reasoning process的格式如下：
（1）每一个reasoning process实例包括具体文本内容和相应打分，开头由{HIST_START_TOKEN}标注，结尾由{HIST_END_TOKEN}标注
（2）每一个reasoning process具体文本内容的开头由{PART_START}标注，结尾由{PART_END}标注，后面紧接着会给出整体的truthfulity score开头由{TRUTH_SCORE_START}标注，结尾由{TRUTH_SCORE_END}标注
（3）每一个reasoning process具体文本内容分为多个step，每个step的具体文本内容开头由{STEP_START}标注，结尾由{STEP_END}标注，每个step文本会给出标题与标号（从1开始）。每个step文本内容结束后，紧接着会给出该step的faithfulness score和factuality score，faithtulity score的开头由{FAITH_SCORE_START}标注，结尾由{FAITH_SCORE_END}标注；factuality score的开头由{FACT_SCORE_START}标注，结尾由{FACT_SCORE_END}标注。

在生成优化后的reasoning process时，请注意您需要同时输出优化后的reasoning process以及您所估计的truthfulity score，faithfulness score和factuality score。具体格式需要参考History中的reasoning process，具体如下：
（1）生成的优化后的reasoning process实例包括具体文本内容和相应打分开头由{OPT_START_TOKEN}标注，结尾由{OPT_END_TOKEN}标注
（2）每一个reasoning process具体文本内容的开头由{PART_START}标注，结尾由{PART_END}标注，后面紧接着给出您所估计的整体的truthfulity score开头由{TRUTH_SCORE_START}标注，结尾由{TRUTH_SCORE_END}标注
（3）每一个reasoning process具体文本内容分为多个step，每个step的具体文本内容开头由{STEP_START}标注，结尾由{STEP_END}标注，每个step文本需要给出标题与标号。step的标号从1开始，每个标题的开头由\"{TOP_START}\"标注，结尾由\"{TOP_END}\"标注。每个step的文本内容结束后，紧接着需要给出您所估计的该step的faithfulness score和factuality score，faithtulity score的开头由{FAITH_SCORE_START}标注，结尾由{FAITH_SCORE_END}标注；factuality score的开头由{FACT_SCORE_START}标注，结尾由{FACT_SCORE_END}标注。
（4）生成的优化后的reasoning process的具体文本内容请以汉语为主体输出。
（5）请注意，您生成的step的数量和标题不需要和History中的reasoning process相同。
""",
                             "qwen": f"""您好，您的任务是观察给定的Documents，Question，Answer，并参考提供的History中的reasoning process以及对这些reasoning process的量化评分 (score)，生成您认为的优化后的即可以获得更高分数的reasoning process。具体来说，每一个生成的reasoning process即为您所认为的当您在根据Documents回答Question时，您的输出可以更好地逼近当前提供的目标Answer时，您所需要使用或遵循的思维方式 (thought) 或逻辑推理过程 (Reasoning Process)。
请注意，在生成优化的reasoning process时您需要遵循以下的要求：
1. 请您 think step by step地来生成reasoning process。在这一过程中体现您的思维（thought）方式或思考逻辑。
2. 请您注意，您所输出的reasoning process需要尽量是对History提供的先前生成的reasoning process进行优化后的结果，且不能重复History中的reasoning process。具体来说，我们对于History中提供的每一个reasoning process同时进行了3个维度的量化评分 (score):
(1) truthfulity score: 范围在0~100之间，该分数代表了您使用相对应的reasoning process来引导您根据提供的Documents回答Question时，其生成的回答结果与提供的目标Answer之间的相似度，分数越高越好。该分数越高则代表使用相应reasoing process时可以越真实的（truthful）引导您生成与目标Answer相似的回答。具体来说，该分数越高，相应reasoning prcocess引导您生成的结果的文本含义或语义蕴含（entail）于目标Answer的程度越高，且生成结果与目标Answer的语义越接近。
(2) faithfulness score: 范围在0~100之间，对于相应的reasoning process的每一步(each step) 都会给出faithfulness score，越高越好。每一步（each step）的该分数代表了您在真实的（truthful）地生成与目标Answer相似的回答时，对该分数的所对应的reasoning prcocess的相应step的忠诚度（faithfulity）。具体来说，该分数越高代表了相应reasoning process的相应step对于逼近目标Answer这一目标的贡献度越高，即，该分数越高代表着您在回答问题时忠实地（faithfuly）采用相应step的推理逻辑的程度越高（也就是说越忠实的反映了您的思维方式），且因为采用该step的逻辑而带来的逼近目标Answer的收益越大。
(3) factuality score: 范围在0~100之间，对于相应的reasoning process的每一步(each step) 都会给出factuality score，越高越好。每一步（each step）的该分数代表了相应的reasoning prcocess的每一步（each step）所表述内容的事实性（factual）程度。具体来说，该分数越高代表了相应reasoning process的相应step的推理依据来自于对提供的Documents或Question的引用或理解的程度越高，相应地，该分数越高，则该step的推理依据来自于您自身的猜测概率越低且推理依据脱离提供的Document与Question的概率越低。
综上所述，请您在生成优化的reasoning process时尽量保证该reasoning process整体可以获得更高的truthfulity score。在这一前提下，请您生成的优化的reasoning process尽量提高每一step的faithfulness score和factuality score。同时请注意，您生成的step的数量和标题不需要和History中的reasoning process相同。
3.请使用汉语来输出优化的reasoning process
4.请您注意，生成reasoning process的目的是引导大模型使其提供的答案逼近目标Answer，无论给定的答案是错误的还是正确的。
5.请您确保不要回答问题Question，也不要给目标Answer提供任何优化建议。
6. 在本次输入的History中，每一个提供的先前生成的reasoning process的格式如下：
（1）每一个reasoning process实例包括具体文本内容和相应打分，开头由{HIST_START_TOKEN}标注，结尾由{HIST_END_TOKEN}标注
（2）每一个reasoning process具体文本内容的开头由{PART_START}标注，结尾由{PART_END}标注，后面紧接着会给出整体的truthfulity score开头由{TRUTH_SCORE_START}标注，结尾由{TRUTH_SCORE_END}标注
（3）每一个reasoning process具体文本内容分为多个step，每个step的具体文本内容开头由{STEP_START}标注，结尾由{STEP_END}标注，每个step文本会给出标题与标号（从1开始）。每个step文本内容结束后，紧接着会给出该step的faithfulness score和factuality score，faithtulity score的开头由{FAITH_SCORE_START}标注，结尾由{FAITH_SCORE_END}标注；factuality score的开头由{FACT_SCORE_START}标注，结尾由{FACT_SCORE_END}标注。

在生成优化后的reasoning process时，请注意您需要同时输出优化后的reasoning process以及您所估计的truthfulity score，faithfulness score和factuality score。具体格式需要参考History中的reasoning process，具体如下：
（1）生成的优化后的reasoning process实例包括具体文本内容和相应打分开头由{OPT_START_TOKEN}标注，结尾由{OPT_END_TOKEN}标注
（2）每一个reasoning process具体文本内容的开头由{PART_START}标注，结尾由{PART_END}标注，后面紧接着给出您所估计的整体的truthfulity score开头由{TRUTH_SCORE_START}标注，结尾由{TRUTH_SCORE_END}标注
（3）每一个reasoning process具体文本内容分为多个step，每个step的具体文本内容开头由{STEP_START}标注，结尾由{STEP_END}标注，每个step文本需要给出标题与标号。step的标号从1开始，每个标题的开头由\"{TOP_START}\"标注，结尾由\"{TOP_END}\"标注。每个step的文本内容结束后，紧接着需要给出您所估计的该step的faithfulness score和factuality score，faithtulity score的开头由{FAITH_SCORE_START}标注，结尾由{FAITH_SCORE_END}标注；factuality score的开头由{FACT_SCORE_START}标注，结尾由{FACT_SCORE_END}标注。
（4）生成的优化后的reasoning process的具体文本内容请以汉语为主体输出。
（5）请注意，您生成的step的数量和标题不需要和History中的reasoning process相同。
""",
                             }

EVO_META_PROMPT_ENG_SUPER = {"gpt": f"""
Hello, your task is to observe the given Documents, Question, and Answer, and refer to the reasoning process provided in the History and the quantitative scoring (score) of these reasoning processes to generate an optimized reasoning process that you think could achieve a higher score. Specifically, each generated reasoning process counts as what you consider the large model needs to use or follow in terms of thought or logical reasoning process when the output of the large model can better approximate the target Answer provided based on the Documents to answer the Question.
Please note that when generating the optimized reasoning process, you need to follow the following requirements:
1. Please generate the reasoning process step by step. This process should reflect the thinking method or logic that you consider for the large model.
2. Note that the reasoning process you realize should be an optimized result of the reasoning process provided in History, and should not repeat the reasoning process in History. Specifically, each reasoning process provided in History has been quantitatively scored in 3 dimensions:
   (1) truthfulness score: Ranging between 0~100, this score represents the similarity between the answer generated by the large model using the corresponding reasoning process to guide the model to answer the Question based on the provided Documents and the target Answer provided - the higher the score, the better. The higher the score, the more truly (truthful) the corresponding reasoning process can guide the large model to generate answers similar to the target Answer. Specifically, the higher the score, the more the corresponding reasoning process guides the model's output to semantically entail the target Answer and be closer to it in meaning.
   (2) faithfulness score: Also ranging between 0~100, a faithfulness score is given for each step of the corresponding reasoning process - the higher, the better. Each step's score represents the fidelity (faithfulness) of the large model in truly generating an answer similar to the target Answer for that score's corresponding step of the reasoning process. Specifically, the higher the score, the more the corresponding step of the reasoning process contributes to the goal of approximating the target Answer. That is, the higher the score, the more faithfully the large model adopts the reasoning logic of the corresponding step in answering the question (meaning the step more faithfully reflects the model's thinking), and the greater is the benefit of approaching the target Answer due to adopting that step's logic.
   (3) factuality score: Also ranging between 0~100, a factuality score is given for each step of the corresponding reasoning process - the higher, the better. Each step's score represents the factual degree of the content expressed in each step of the reasoning process. Specifically, the higher the score, the more the reasoning basis of the corresponding step of the reasoning process comes from the reference or understanding of the provided Documents or Question, correspondingly, the lower the probability that the reasoning basis comes from the model's own guesses and the lower the probability that the reasoning basis strays from the provided Document and Question.
In summary, when generating the optimized reasoning process, please ensure that the reasoning process as a whole can achieve a higher truthfulness score. Under this premise, please try to improve the faithfulness score and factuality score for each step of your generated optimized reasoning process. Also, note that the number of steps and titles in your generated process do not need to be the same as those in History.
3. Please output the optimized reasoning process in English.
4. Please note that the purpose of generating the reasoning process is to guide the large model so that its provided answer approximates the target Answer, regardless of whether the given answer is wrong or right.
5. Please ensure not to answer the Question or provide any optimization suggestions for the target Answer.
6. In the History provided in this input, each previously generated reasoning process is formatted as follows:
   (1) Each reasoning process instance includes specific text content and respective scoring, starting with {HIST_START_TOKEN} and ending with {HIST_END_TOKEN}
   (2) The beginning of each reasoning process's specific text content is marked with {PART_START} and ends with {PART_END}, followed by the overall truthfulness score starting with {TRUTH_SCORE_START} and ending with {TRUTH_SCORE_END}
   (3) Each reasoning process's specific text content is divided into multiple steps, each with its specific text content starting with {STEP_START} and ending with {STEP_END}, each step text provides a title and number (starting from 1). After the end of each step text, the faithfulness score and factuality score estimated for that step are given; the start of the faithfulness score is marked with {FAITH_SCORE_START} and ends with {FAITH_SCORE_END}; the start of the factuality score is marked with {FACT_SCORE_START} and ends with {FACT_SCORE_END}.
When generating the optimized reasoning process, please note that you need to output both the optimized reasoning process and your estimated scores for truthfulness, faithfulness, and factuality. The specific format should refer to the reasoning process in History, as follows:
(1) The instance of the optimized reasoning process generated includes specific text content and corresponding scores, marked at the beginning by {OPT_START_TOKEN} and at the end by {OPT_END_TOKEN}
(2) The beginning of each specific text content of a reasoning process is marked by {PART_START}, and the end is marked by {PART_END}, followed immediately by your estimated overall truthfulness score, starting with {TRUTH_SCORE_START} and ending with {TRUTH_SCORE_END}
(3) The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}. Each step's text needs to provide a title and number. The numbering of steps starts at 1, with each title's start denoted by {TOP_START} and end by {TOP_END}. After the text content of each step, you need to immediately provide your estimated faithfulness score and factuality score for that step. The beginning of the faithfulness score is marked by {FAITH_SCORE_START} and the end by {FAITH_SCORE_END}; the beginning of the factuality score is marked by {FACT_SCORE_START} and the end by {FACT_SCORE_END}.
(4) Please output the specific text content of the optimized reasoning process in English.
(5) Please note that the number and titles of steps you generate do not need to match those in the reasoning process in History.
""",

                             "qwen": f"""Hello, your task is to observe the given Documents, Question, and Answer, and refer to the reasoning process in the provided History and the quantitative scoring (score) of these reasoning processes to generate an optimized reasoning process that you believe can achieve higher scores. Specifically, each generated reasoning process is what you think you need to use or follow in terms of thinking (thought) or logical reasoning process (Reasoning Process) when you approach the target Answer better as you answer the Question based on the Documents.
Please note the following requirements when generating the optimized reasoning process:
1. Please think step by step to generate the reasoning process. Reflect your thinking (thought) method or thinking logic during this process.
2. Please note that the reasoning process you output should be an optimized result of the reasoning process previously provided in History, and should not repeat the reasoning process in History. Specifically, we have quantitatively scored each reasoning process provided in History in three dimensions:
   (1) truthfulness score: ranging from 0 to 100, this score represents the similarity between the answer generated by guiding you with the corresponding reasoning process when answering the Question based on the provided Documents and the provided target Answer, the higher the better. The higher the score, the more truthfully (truthful) it guides you to generate an answer similar to the target Answer. Specifically, the higher the score, the higher the degree to which the text meaning or semantic implication (entail) of the result generated by the corresponding reasoning process is at the target Answer, and the closer the semantics of the generated result are to the target Answer.
   (2) faithfulness score: ranging from 0 to 100, a faithfulness score is given for each step of the corresponding reasoning process, the higher the better. This score for each step (each step) represents the degree of fidelity (faithfulness) of the corresponding reasoning process step when you truthfully (truthful) generate an answer similar to the target Answer. Specifically, the higher the score, the greater the contribution of the corresponding reasoning process step to the goal of approaching the target Answer, that is, the higher the score represents the higher degree of fidelity (faithfuly) in employing the reasoning logic of the corresponding step when answering the question, and the greater the benefit of approaching the target Answer due to using the logic of that step.
   (3) factuality score: ranging from 0 to 100, a factuality score is given for each step of the corresponding reasoning process, the higher the better. This score for each step (each step) represents the degree of factuality of the content expressed in each step of the corresponding reasoning process. Specifically, the higher the score, the more the reasoning basis of the corresponding reasoning process step comes from the degree of reference or understanding of the provided Documents or Question, accordingly, the higher the score, the lower the probability that the reasoning basis of the step comes from your own guess and the lower the probability of the reasoning basis deviating from the provided Document and Question.
In summary, please ensure that the optimized reasoning process as a whole can achieve a higher overall truthfulness score when generating the optimized reasoning process. Under this premise, please try to improve the faithfulness score and factuality score of each step of your optimized reasoning process. Also, please note that the number and titles of steps you generate do not need to match those in the reasoning process in History.
3. Please output the optimized reasoning process in English.
4. Please note that the purpose of generating a reasoning process is to guide the large model to provide an answer that approximates the target Answer, regardless of whether the given answer is wrong or right.
5. Please ensure not to answer the Question, nor to provide any optimization suggestions for the target Answer.
6. In the input History for this session, the format of each previously generated reasoning process provided is as follows:
(1) Each reasoning process instance includes specific text content and corresponding scores, marked at the beginning by {HIST_START_TOKEN} and at the end by {HIST_END_TOKEN}
(2) The beginning of each specific text content of a reasoning process is marked by {PART_START}, and the end is marked by {PART_END}, immediately followed by the overall truthfulness score starting with {TRUTH_SCORE_START} and ending with {TRUTH_SCORE_END}
(3) The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, each step text will provide a title and number (starting from 1). After the text content of each step, the faithfulness score and factuality score for that step are given, with the beginning of the faithfulness score marked by {FAITH_SCORE_START} and the end by {FAITH_SCORE_END}; the factuality score starts with {FACT_SCORE_START} and ends with {FACT_SCORE_END}.
When generating the optimized reasoning process, please note that you need to output both the optimized reasoning process and your estimated truthfulness score, faithfulness score, and factuality score. The specific format should refer to the reasoning process in History, as detailed below:
(1) The instance of the optimized reasoning process generated includes specific text content and corresponding scores, starting with {OPT_START_TOKEN} and ending with {OPT_END_TOKEN}
(2) The beginning of each specific text content of a reasoning process is marked by {PART_START}, and the end is marked by {PART_END}, immediately followed by your estimated overall truthfulness score, starting with {TRUTH_SCORE_START} and ending with {TRUTH_SCORE_END}
(3) The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, each step text needs to provide a title and number. The numbering of steps starts at 1, with the beginning of each title marked by "{TOP_START}" and the end by "{TOP_END}". After the text content of each step, you need to immediately provide your estimated faithfulness score and factuality score for that step, with the beginning of the faithfulness score marked by {FAITH_SCORE_START} and ending by {FAITH_SCORE_END}; the factuality score starts with {FACT_SCORE_START} and ends with {FACT_SCORE_END}.
(4) Please output the specific text content of the optimized reasoning process primarily in English.
(5) Please note, the number and titles of steps you generate do not need to match those in the reasoning process in History.
"""
                             }

EVO_META_PROMPT_NO_WEIGHT_CHN_SUPER = {"gpt": f"""您好，您的任务是观察给定的Documents，Question，Answer，并参考提供的History中的reasoning process以及对这些reasoning process的量化评分 (score)，生成您认为的优化后的即可以获得更高分数的reasoning process。具体来说，每一个生成的reasoning process即为您所认为的大模型在根据Documents回答Question时，当大模型的输出可以更好地逼近当前提供的目标Answer时，大模型所需要使用或遵循的思维方式 (thought) 或逻辑推理过程 (Reasoning Process)。
请注意，在生成优化的reasoning process时您需要遵循以下的要求：
1. 请您 think step by step地来生成reasoning process。在这一过程中体现您认为的大模型的思维（thought）方式或思考逻辑。
2. 请您注意，您所实现的reasoning process需要尽量是对History提供的先前生成的reasoning process进行优化后的结果,且不能重复History中的reasoning process。具体来说，我们对于History中提供的每一个reasoning process同时进行了3个维度的量化评分 (score):
(1) truthfulity score: 范围在0~100之间，该分数代表了大模型使用相对应的reasoning process来引导大模型根据提供的Documents回答Question时，其生成的回答结果与提供的目标Answer之间的相似度，分数越高越好。该分数越高则代表使用相应reasoing process时可以越真实的（truthful）引导大模型生成与目标Answer相似的回答。具体来说，该分数越高，相应reasoning prcocess引导大模型生成的结果的文本含义或语义蕴含（entail）于目标Answer的程度越高，且生成结果与目标Answer的语义越接近。
(2) faithfulness score: 范围在0~100之间，对于相应的reasoning process的每一步(each step) 都会给出faithfulness score，越高越好。每一步（each step）的该分数代表了大模型在生成对Question的回答时，对该分数的所对应的reasoning prcocess的相应step的忠诚度（faithfulity）。具体来说，该分数越高代表着大模型在回答问题时忠实地（faithfuly）采用相应step的推理逻辑的程度越高（也就是说该step越忠实地反映了大模型的思维方式）。
(3) factuality score: 范围在0~100之间，对于相应的reasoning process的每一步(each step) 都会给出factuality score，越高越好。每一步（each step）的该分数代表了相应的reasoning prcocess的每一步（each step）所表述内容的事实性（factual）程度。具体来说，该分数越高代表了相应reasoning process的相应step的推理依据来自于对提供的Documents或Question的引用或理解的程度越高，相应地，该step的推理依据来自于大模型自身的猜测概率越低且推理依据脱离提供的Document与Question的概率越低。
综上所述，请您在生成优化的reasoning process时尽量保证该reasoning process整体可以获得更高的truthfulity score。在这一前提下，请您生成的优化的reasoning process尽量提高每一step的faithfulness score和factuality score。同时请注意，您生成的step的数量和标题不需要和History中的reasoning process相同。
3.请使用汉语来输出优化的reasoning process
4.请您注意，生成reasoning process的目的是引导大模型使其提供的答案逼近目标Answer，无论给定的答案是错误的还是正确的。
5.请您确保不要回答问题Question，也不要给目标Answer提供任何优化建议。
6. 在本次输入的History中，每一个提供的先前生成的reasoning process的格式如下：
（1）每一个reasoning process实例包括具体文本内容和相应打分，开头由{HIST_START_TOKEN}标注，结尾由{HIST_END_TOKEN}标注
（2）每一个reasoning process具体文本内容的开头由{PART_START}标注，结尾由{PART_END}标注，后面紧接着会给出整体的truthfulity score开头由{TRUTH_SCORE_START}标注，结尾由{TRUTH_SCORE_END}标注
（3）每一个reasoning process具体文本内容分为多个step，每个step的具体文本内容开头由{STEP_START}标注，结尾由{STEP_END}标注，每个step文本会给出标题与标号（从1开始）。每个step文本内容结束后，紧接着会给出该step的faithfulness score和factuality score，faithtulity score的开头由{FAITH_SCORE_START}标注，结尾由{FAITH_SCORE_END}标注；factuality score的开头由{FACT_SCORE_START}标注，结尾由{FACT_SCORE_END}标注。

在生成优化后的reasoning process时，请注意您需要同时输出优化后的reasoning process以及您所估计的truthfulity score，faithfulness score和factuality score。具体格式需要参考History中的reasoning process，具体如下：
（1）生成的优化后的reasoning process实例包括具体文本内容和相应打分开头由{OPT_START_TOKEN}标注，结尾由{OPT_END_TOKEN}标注
（2）每一个reasoning process具体文本内容的开头由{PART_START}标注，结尾由{PART_END}标注，后面紧接着给出您所估计的整体的truthfulity score开头由{TRUTH_SCORE_START}标注，结尾由{TRUTH_SCORE_END}标注
（3）每一个reasoning process具体文本内容分为多个step，每个step的具体文本内容开头由{STEP_START}标注，结尾由{STEP_END}标注，每个step文本需要给出标题与标号。step的标号从1开始，每个标题的开头由\"{TOP_START}\"标注，结尾由\"{TOP_END}\"标注。每个step的文本内容结束后，紧接着需要给出您所估计的该step的faithfulness score和factuality score，faithtulity score的开头由{FAITH_SCORE_START}标注，结尾由{FAITH_SCORE_END}标注；factuality score的开头由{FACT_SCORE_START}标注，结尾由{FACT_SCORE_END}标注。
（4）生成的优化后的reasoning process的具体文本内容请以汉语为主体输出。
（5）请注意，您生成的step的数量和标题不需要和History中的reasoning process相同。
""",
                                       "qwen": f"""您好，您的任务是观察给定的Documents，Question，Answer，并参考提供的History中的reasoning process以及对这些reasoning process的量化评分 (score)，生成您认为的优化后的即可以获得更高分数的reasoning process。具体来说，每一个生成的reasoning process即为您所认为的当您在根据Documents回答Question时，您的输出可以更好地逼近当前提供的目标Answer时，您所需要使用或遵循的思维方式 (thought) 或逻辑推理过程 (Reasoning Process)。
请注意，在生成优化的reasoning process时您需要遵循以下的要求：
1. 请您 think step by step地来生成reasoning process。在这一过程中体现您的思维（thought）方式或思考逻辑。
2. 请您注意，您所输出的reasoning process需要尽量是对History提供的先前生成的reasoning process进行优化后的结果，且不能重复History中的reasoning process。具体来说，我们对于History中提供的每一个reasoning process同时进行了3个维度的量化评分 (score):
(1) truthfulity score: 范围在0~100之间，该分数代表了您使用相对应的reasoning process来引导您根据提供的Documents回答Question时，其生成的回答结果与提供的目标Answer之间的相似度，分数越高越好。该分数越高则代表使用相应reasoing process时可以越真实的（truthful）引导您生成与目标Answer相似的回答。具体来说，该分数越高，相应reasoning prcocess引导您生成的结果的文本含义或语义蕴含（entail）于目标Answer的程度越高，且生成结果与目标Answer的语义越接近。
(2) faithfulness score: 范围在0~100之间，对于相应的reasoning process的每一步(each step) 都会给出faithfulness score，越高越好。每一步（each step）的该分数代表了您在生成对Question的回答时，对该分数的所对应的reasoning prcocess的相应step的忠诚度（faithfulity）。具体来说，该分数越高代表着您在回答问题时忠实地（faithfuly）采用相应step的推理逻辑的程度越高（也就是说该step越忠实地反映了您的思维方式）。
(3) factuality score: 范围在0~100之间，对于相应的reasoning process的每一步(each step) 都会给出factuality score，越高越好。每一步（each step）的该分数代表了相应的reasoning prcocess的每一步（each step）所表述内容的事实性（factual）程度。具体来说，该分数越高代表了相应reasoning process的相应step的推理依据来自于对提供的Documents或Question的引用或理解的程度越高，相应地，该分数越高，则该step的推理依据来自于您自身的猜测概率越低且推理依据脱离提供的Document与Question的概率越低。
综上所述，请您在生成优化的reasoning process时尽量保证该reasoning process整体可以获得更高的truthfulity score。在这一前提下，请您生成的优化的reasoning process尽量提高每一step的faithfulness score和factuality score。同时请注意，您生成的step的数量和标题不需要和History中的reasoning process相同。
3.请使用汉语来输出优化的reasoning process
4.请您注意，生成reasoning process的目的是引导大模型使其提供的答案逼近目标Answer，无论给定的答案是错误的还是正确的。
5.请您确保不要回答问题Question，也不要给目标Answer提供任何优化建议。
6. 在本次输入的History中，每一个提供的先前生成的reasoning process的格式如下：
（1）每一个reasoning process实例包括具体文本内容和相应打分，开头由{HIST_START_TOKEN}标注，结尾由{HIST_END_TOKEN}标注
（2）每一个reasoning process具体文本内容的开头由{PART_START}标注，结尾由{PART_END}标注，后面紧接着会给出整体的truthfulity score开头由{TRUTH_SCORE_START}标注，结尾由{TRUTH_SCORE_END}标注
（3）每一个reasoning process具体文本内容分为多个step，每个step的具体文本内容开头由{STEP_START}标注，结尾由{STEP_END}标注，每个step文本会给出标题与标号（从1开始）。每个step文本内容结束后，紧接着会给出该step的faithfulness score和factuality score，faithtulity score的开头由{FAITH_SCORE_START}标注，结尾由{FAITH_SCORE_END}标注；factuality score的开头由{FACT_SCORE_START}标注，结尾由{FACT_SCORE_END}标注。

在生成优化后的reasoning process时，请注意您需要同时输出优化后的reasoning process以及您所估计的truthfulity score，faithfulness score和factuality score。具体格式需要参考History中的reasoning process，具体如下：
（1）生成的优化后的reasoning process实例包括具体文本内容和相应打分开头由{OPT_START_TOKEN}标注，结尾由{OPT_END_TOKEN}标注
（2）每一个reasoning process具体文本内容的开头由{PART_START}标注，结尾由{PART_END}标注，后面紧接着给出您所估计的整体的truthfulity score开头由{TRUTH_SCORE_START}标注，结尾由{TRUTH_SCORE_END}标注
（3）每一个reasoning process具体文本内容分为多个step，每个step的具体文本内容开头由{STEP_START}标注，结尾由{STEP_END}标注，每个step文本需要给出标题与标号。step的标号从1开始，每个标题的开头由\"{TOP_START}\"标注，结尾由\"{TOP_END}\"标注。每个step的文本内容结束后，紧接着需要给出您所估计的该step的faithfulness score和factuality score，faithtulity score的开头由{FAITH_SCORE_START}标注，结尾由{FAITH_SCORE_END}标注；factuality score的开头由{FACT_SCORE_START}标注，结尾由{FACT_SCORE_END}标注。
（4）生成的优化后的reasoning process的具体文本内容请以汉语为主体输出。
（5）请注意，您生成的step的数量和标题不需要和History中的reasoning process相同。
""",
                                       }

EVO_META_PROMPT_NO_WEIGHT_ENG_SUPER = {
    "gpt": f"""
Hello, your task is to observe the given Documents, Question, and refer to the reasoning process and their quantitative scores (score) provided in History, to generate what you consider an optimized reasoning process that could result in higher scores. Specifically, each generated reasoning process should represent what you consider to be the large model's way of thinking (thought) or logical reasoning process (Reasoning Process) that it uses or follows when answering the Question based on the Documents.
Please follow these requirements when generating the optimized reasoning process:
1. Think step by step when generating the reasoning process. This process should reflect your understanding of the LLMs's way of thinking (thought) or logic.
2. The reasoning process you devise should be, as much as possible, an improvement on the ones previously generated in History, without repeating them. Specifically, each reasoning process in History has been quantitatively scored in three dimensions:
   (1) truthfulness score: Ranging from 0 to 100, this score indicates the similarity between the answer produced by the LLMs using the corresponding reasoning process and the standard correct answer when answering the Question based on the provided Documents—the higher, the better. The higher this score, the more truthfully (truthful) the corresponding reasoning process can guide the LLMs to generate answers similar to the standard correct answer. In other words, the higher the score, the greater the extent to which the text meaning or semantic implication (entailment) of the result generated by the reasoning process is in accordance with the standard correct answer, and the closer the semantics and structure of the generated result are to the standard correct answer.
   (2) faithfulness score: Also ranging from 0 to 100, each step (each step) of the corresponding reasoning process is given a faithfulness score, with higher scores indicating better performance. This score for each step signifies how faithfully the reasoning process step corresponds with the LLMs's answer that closely matches the standard correct answer. The higher this score, the greater the contribution of the respective step of the reasoning process towards approximating the standard correct answer. The higher the score, the more faithfully (faithfuly) the LLMs employs the reasoning logic of the step when answering the question (meaning the step more faithfully reflects the LLMs's way of thinking), and the greater the benefit of approaching the standard correct answer due to the use of the step's logic.
   (3) factuality score: Also ranging from 0 to 100, a factuality score is assigned to each step (each step) of the corresponding reasoning process, with higher scores again indicating better performance. This score for each step represents the degree of factualness of the content described in each step of the respective reasoning process. Higher scores indicate that the reasoning basis of the respective step of the reasoning process derives more from the reference to or understanding of the provided Documents or Question, and correspondingly, the less it relies on the LLMs's own assumptions or deviates from the Documents and Question provided.
In summary, please ensure that the overall optimized reasoning process you generate can achieve a higher overall truthfulness score. With this in mind, aim to improve the faithfulness score and factuality score for each step of the optimized reasoning process. Also, note that the number and titles of the steps you generate do not need to match those in the reasoning process in History.
3. Please output the optimized reasoning process in English.
4. Note that the purpose of generating a reasoning process is to explain the thinking or reasoning method of the LLMs when generating answers. The way of thinking or reasoning explained through the reasoning process can guide the LLMs to refer to the Documents as correctly as possible when answering the Question, to obtain the correct Answer as accurately as possible.
5. Ensure not to directly answer the Question in the generated reasoning process.
6. In the current input History, the format for each previously generated reasoning process provided is as follows:
(1) Each reasoning process instance includes specific text content and corresponding scores, marked at the beginning by {HIST_START_TOKEN} and at the end by {HIST_END_TOKEN}.
(2) The beginning of each specific text content of a reasoning process is marked by {PART_START}, and the end is marked by {PART_END}, immediately followed by the overall truthfulness score starting with {TRUTH_SCORE_START} and ending with {TRUTH_SCORE_END}.
(3) Each reasoning process's specific text content is divided into multiple steps; each step's specific text content starts with {STEP_START} and ends with {STEP_END}, providing a title and a number for each step (starting with 1). After the text content of each step, the faithfulness score and factuality score for that step are given, marked at the beginning with {FAITH_SCORE_START} and at the end with {FAITH_SCORE_END} for the faithfulness score; the factuality score starts with {FACT_SCORE_START} and ends with {FACT_SCORE_END}.
When generating the optimized reasoning process, please note that you need to output both the optimized reasoning process and your estimated truthfulness score, faithfulness score, and factuality score. The specific format should refer to the reasoning process in History, as follows:
(1) The instance of the optimized reasoning process generated includes specific text content and corresponding scores, starting with {OPT_START_TOKEN} and ending with {OPT_END_TOKEN}.
(2) The beginning of each specific text content of a reasoning process is marked by {PART_START}, and the end is marked by {PART_END}, immediately followed by your estimated overall truthfulness score starting with {TRUTH_SCORE_START} and ending with {TRUTH_SCORE_END}.
(3) The specific text content of each reasoning process is divided into multiple steps; each step's specific text content starts with {STEP_START} and ends with {STEP_END}, requiring a title and a number for each step. The numbering of steps starts at 1, with each title marked at the beginning with "{TOP_START}" and at the end with "{TOP_END}". Immediately after the text content of each step, you need to provide your estimated faithfulness score and factuality score for that step, starting with {FAITH_SCORE_START} for the faithfulness score and ending with {FAITH_SCORE_END}; the factuality score starting with {FACT_SCORE_START} and ending with {FACT_SCORE_END}.
(4) The specific text content of the optimized reasoning process should primarily be output in English.
(5) Please note that the number and titles of the steps you generate do not need to match those in the reasoning process in History.
    """,
    "qwen": f"""
Hello, your task is to observe the given Documents, Question, Answer, and refer to the reasoning process and their quantitative scores (score) provided in History to generate what you consider an optimized reasoning process that could result in higher scores. Specifically, each generated reasoning process should represent the way of thinking (thought) or logical reasoning process (Reasoning Process) that you would need to follow to better approximate the given target Answer when you answer the Question based on the Documents.
Please follow these requirements when generating the optimized reasoning process:
1. Think step by step to generate the reasoning process. This process should reflect your thinking method or logic.
2. Your outputted reasoning process should be an optimized result based on reasoning processes previously provided in History, without repeating them. Specifically, we have scored each reasoning process in History on three quantitative dimensions:
   (1) truthfulness score: Ranging between 0 to 100, this score represents the similarity between the answer generated by using the corresponding reasoning process to answer the Question based on the Documents and the target Answer provided, with higher scores being better. The higher the score, the more truthfully the corresponding reasoning process can guide you to generate an answer similar to the target Answer. That is, the higher the score, the greater the extent to which the resulting text meaning or semantic implication (entailment) is in accordance with the target Answer.
   (2) faithfulness score: Also ranging between 0 to 100, a faithfulness score is given to each step of the corresponding reasoning process, with higher scores being better. This score for each step reflects the level of faithfulness to the reasoning logic of that step when generating an answer to the Question. A higher score indicates a higher degree of faithfulness in employing the reasoning logic of the respective step.
   (3) factuality score: This is a score given to each step of the reasoning process, ranging between 0 to 100. A higher score indicates a higher degree of factuality of the content expressed in each step. A higher score suggests that the reasoning basis of the respective step comes more from the documents or the understanding of the Question, meaning that the reasoning is based less on guesswork and more on the information provided.
In summary, while generating the optimized reasoning process, ensure that the overall process can achieve a higher truthfulness score. Under this premise, aim to improve the faithfulness score and factuality score for each step of the optimized reasoning process. Please note that the number and titles of steps you generate do not need to match those in the reasoning process in History.
3. Please use English to output the optimized reasoning process.
4. Keep in mind that the purpose of generating the reasoning process is to guide the large model to produce an answer that approximates the target Answer, regardless of whether the given answer is correct or mistaken.
5. Ensure not to directly answer the Question or offer any optimization suggestions for the target Answer.
6. In this session's input History, the format for each previously generated reasoning process is as follows:
(1) Each reasoning process instance includes specific text content and corresponding scoring, marked at the beginning by {HIST_START_TOKEN} and at the end by {HIST_END_TOKEN}.
(2) The beginning of the specific text content of each reasoning process is marked by {PART_START}, and the end is marked by {PART_END}, closely followed by the overall truthfulness score which starts with {TRUTH_SCORE_START} and ends with {TRUTH_SCORE_END}.
(3) Each reasoning process’s specific text content is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, providing a title and number for each step (starting with 1). After the text content of each step, a faithfulness score and a factuality score are given for that step, with faithfulness scoring starting with {FAITH_SCORE_START} and ending with {FAITH_SCORE_END}, and factuality scoring starting with {FACT_SCORE_START} and ending with {FACT_SCORE_END}.
When generating the optimized reasoning process, please note that you need to output both the optimized reasoning process and your estimated truthfulness score, faithfulness score, and factuality score. The specific format should refer to the reasoning processes in History, as detailed below:
(1) Instances of optimized reasoning processes that are generated should include specific text content and corresponding scores, starting with {OPT_START_TOKEN} and ending with {OPT_END_TOKEN}.
(2) The beginning of the specific text content of each reasoning process should start with {PART_START} and end with {PART_END}, followed closely by your estimated overall truthfulness score, beginning with {TRUTH_SCORE_START} and ending with {TRUTH_SCORE_END}.
(3) The specific text content of each reasoning process should be divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, and each step text should provide a title and number. Steps are numbered starting with 1, with the beginning of each title marked with "{TOP_START}" and the end with "{TOP_END}". After the text of each step, you need to immediately provide your estimated faithfulness score and factuality score for that step, with faithfulness scoring starting with {FAITH_SCORE_START} and ending with {FAITH_SCORE_END}, and factuality scoring beginning with {FACT_SCORE_START} and ending with {FACT_SCORE_END}.
(4) Please output the specific text content of the optimized reasoning process primarily in English.
(5) Please note that the number and titles of the steps you generate do not need to match those in the reasoning process in History.
"""
}

EVO_META_PROMPT_CHN_UNSUPER = {"gpt": f"""您好，您的任务是观察给定的Documents，Question，并参考提供的History中的reasoning process以及对这些reasoning process的量化评分 (score)，生成您认为的优化后的即可以获得更高分数的reasoning process。具体来说，每一个生成的reasoning process即为您所认为的大模型在根据Documents回答Question时，大模型所使用或遵循的思维方式 (thought) 或逻辑推理过程 (Reasoning Process)。
请注意，在生成优化的reasoning process时您需要遵循以下的要求：
1. 请您 think step by step地来生成reasoning process。在这一过程中体现您认为的大模型的思维（thought）方式或思考逻辑。
2. 请您注意，您所实现的reasoning process需要尽量是对History提供的先前生成的reasoning process进行优化后的结果,且不能重复History中的reasoning process。具体来说，我们对于History中提供的每一个reasoning process同时进行了3个维度的量化评分 (score):
(1) truthfulity score: 范围在0~100之间，该分数代表了大模型使用相对应的reasoning process来引导大模型根据提供的Documents回答Question时，其生成的回答结果与标准正确答案Answer之间的相似度，分数越高越好。该分数越高则代表使用相应reasoing process时可以越真实的（truthful）引导大模型生成与标准正确答案Answer相似的回答。具体来说，该分数越高，相应reasoning prcocess引导大模型生成的结果的文本含义或语义蕴含（entail）于标准正确答案Answer的程度越高，且生成结果与标准正确答案Answer的语义与结构越接近。
(2) faithfulness score: 范围在0~100之间，对于相应的reasoning process的每一步(each step) 都会给出faithfulness score，越高越好。每一步（each step）的该分数代表了大模型在真实的（truthful）地生成与标准正确答案Answer相似的回答时，对该分数的所对应的reasoning prcocess的相应step的忠诚度（faithfulity）。具体来说，该分数越高代表了相应reasoning process的相应step对于逼近标准正确答案Answer这一目标的贡献度越高。即，该分数越高代表着大模型在回答问题时忠实地（faithfuly）采用相应step的推理逻辑的程度越高（也就是说相应step越忠实地反映了大模型的思维方式），且因为采用该step的逻辑而带来的逼近标准正确答案Answer的收益越大。
(3) factuality score: 范围在0~100之间，对于相应的reasoning process的每一步(each step) 都会给出factuality score，越高越好。每一步（each step）的该分数代表了相应的reasoning prcocess的每一步（each step）所表述内容的事实性（factual）程度。具体来说，该分数越高代表了相应reasoning process的相应step的推理依据来自于对提供的Documents或Question的引用或理解的程度越高，相应地，该step的推理依据来自于大模型自身的猜测概率越低且推理依据脱离提供的Document与Question的概率越低。
综上所述，请您在生成优化的reasoning process时尽量保证该reasoning process整体可以获得更高的truthfulity score。在这一前提下，请您生成的优化的reasoning process尽量提高每一step的faithfulness score和factuality score。同时请注意，您生成的step的数量和标题不需要和History中的reasoning process相同。
3.请使用汉语来输出优化的reasoning process
4.请您注意，生成reasoning process的目的是解释大模型生成答案时的思维方式或思考方式，通过reasoning process解释的思维或思考方式，可以引导大模型使其可以尽量正确地参考Documents来回答Question，以尽量正确地获得正确Answer。
5.请您确保不要在生成的reasoning process中直接回答问题Question。
6. 在本次输入的History中，每一个提供的先前生成的reasoning process的格式如下：
（1）每一个reasoning process实例包括具体文本内容和相应打分，开头由{HIST_START_TOKEN}标注，结尾由{HIST_END_TOKEN}标注
（2）每一个reasoning process具体文本内容的开头由{PART_START}标注，结尾由{PART_END}标注，后面紧接着会给出整体的truthfulity score开头由{TRUTH_SCORE_START}标注，结尾由{TRUTH_SCORE_END}标注
（3）每一个reasoning process具体文本内容分为多个step，每个step的具体文本内容开头由{STEP_START}标注，结尾由{STEP_END}标注，每个step文本会给出标题与标号（从1开始）。每个step文本内容结束后，紧接着会给出该step的faithfulness score和factuality score，faithtulity score的开头由{FAITH_SCORE_START}标注，结尾由{FAITH_SCORE_END}标注；factuality score的开头由{FACT_SCORE_START}标注，结尾由{FACT_SCORE_END}标注。

在生成优化后的reasoning process时，请注意您需要同时输出优化后的reasoning process以及您所估计的truthfulity score，faithfulness score和factuality score。具体格式需要参考History中的reasoning process，具体如下：
（1）生成的优化后的reasoning process实例包括具体文本内容和相应打分开头由{OPT_START_TOKEN}标注，结尾由{OPT_END_TOKEN}标注
（2）每一个reasoning process具体文本内容的开头由{PART_START}标注，结尾由{PART_END}标注，后面紧接着给出您所估计的整体的truthfulity score开头由{TRUTH_SCORE_START}标注，结尾由{TRUTH_SCORE_END}标注
（3）每一个reasoning process具体文本内容分为多个step，每个step的具体文本内容开头由{STEP_START}标注，结尾由{STEP_END}标注，每个step文本需要给出标题与标号。step的标号从1开始，每个标题的开头由\"{TOP_START}\"标注，结尾由\"{TOP_END}\"标注。每个step的文本内容结束后，紧接着需要给出您所估计的该step的faithfulness score和factuality score，faithtulity score的开头由{FAITH_SCORE_START}标注，结尾由{FAITH_SCORE_END}标注；factuality score的开头由{FACT_SCORE_START}标注，结尾由{FACT_SCORE_END}标注。
（4）生成的优化后的reasoning process的具体文本内容请以汉语为主体输出。
（5）请注意，您生成的step的数量和标题不需要和History中的reasoning process相同。
""",
"qwen": f"""您好，您的任务是观察给定的Documents，Question，并参考提供的History中的reasoning process以及对这些reasoning process的量化评分 (score)，生成您认为的优化后的即可以获得更高分数的reasoning process。具体来说，每一个生成的reasoning process即为您所认为的大模型在根据Documents回答Question时，大模型所使用或遵循的思维方式 (thought) 或逻辑推理过程 (Reasoning Process)。
请注意，在生成优化的reasoning process时您需要遵循以下的要求：
1. 请您 think step by step地来生成reasoning process。在这一过程中体现您的思维（thought）方式或思考逻辑。
2. 请您注意，您所输出的reasoning process需要尽量是对History提供的先前生成的reasoning process进行优化后的结果，且不能重复History中的reasoning process。具体来说，我们对于History中提供的每一个reasoning process同时进行了3个维度的量化评分 (score):
(1) truthfulity score: 范围在0~100之间，该分数代表了您使用相对应的reasoning process来引导您根据提供的Documents回答Question时，其生成的回答结果与标准正确答案Answer之间的相似度，分数越高越好。该分数越高则代表使用相应reasoing process时可以越真实的（truthful）引导您生成与标准正确答案Answer相似的回答。具体来说，该分数越高，相应reasoning prcocess引导您生成的结果的文本含义或语义蕴含（entail）于标准正确答案Answer的程度越高，且生成结果与标准正确答案Answer的语义与结构越接近。
(2) faithfulness score: 范围在0~100之间，对于相应的reasoning process的每一步(each step) 都会给出faithfulness score，越高越好。每一步（each step）的该分数代表了您在真实的（truthful）地生成与标准正确答案Answer相似的回答时，对该分数的所对应的reasoning prcocess的相应step的忠诚度（faithfulity）。具体来说，该分数越高代表了相应reasoning process的相应step对于逼近标准正确答案Answer这一目标的贡献度越高。即，该分数越高代表着您在回答问题时忠实地（faithfuly）采用相应step的推理逻辑的程度越高（也就是说相应step越忠实地反映了您的思维方式），且因为采用该step的逻辑而带来的逼近标准正确答案Answer的收益越大。
(3) factuality score: 范围在0~100之间，对于相应的reasoning process的每一步(each step) 都会给出factuality score，越高越好。每一步（each step）的该分数代表了相应的reasoning prcocess的每一步（each step）所表述内容的事实性（factual）程度。具体来说，该分数越高代表了相应reasoning process的相应step的推理依据来自于对提供的Documents或Question的引用或理解的程度越高，相应地，该分数越高，则该step的推理依据来自于您自身的猜测概率越低且推理依据脱离提供的Document与Question的概率越低。
综上所述，请您在生成优化的reasoning process时尽量保证该reasoning process整体可以获得更高的truthfulity score。在这一前提下，请您生成的优化的reasoning process尽量提高每一step的faithfulness score和factuality score。同时请注意，您生成的step的数量和标题不需要和History中的reasoning process相同。
3.请使用汉语来输出优化的reasoning process
4.请您注意，生成reasoning process的目的是解释大模型生成答案时的思维方式或思考方式，通过reasoning process解释的思维或思考方式，可以引导大模型使其可以尽量正确地参考Documents来回答Question，以尽量正确地获得正确Answer。
5.请您确保不要在生成的reasoning process中直接回答问题Question。
6. 在本次输入的History中，每一个提供的先前生成的reasoning process的格式如下：
（1）每一个reasoning process实例包括具体文本内容和相应打分，开头由{HIST_START_TOKEN}标注，结尾由{HIST_END_TOKEN}标注
（2）每一个reasoning process具体文本内容的开头由{PART_START}标注，结尾由{PART_END}标注，后面紧接着会给出整体的truthfulity score开头由{TRUTH_SCORE_START}标注，结尾由{TRUTH_SCORE_END}标注
（3）每一个reasoning process具体文本内容分为多个step，每个step的具体文本内容开头由{STEP_START}标注，结尾由{STEP_END}标注，每个step文本会给出标题与标号（从1开始）。每个step文本内容结束后，紧接着会给出该step的faithfulness score和factuality score，faithtulity score的开头由{FAITH_SCORE_START}标注，结尾由{FAITH_SCORE_END}标注；factuality score的开头由{FACT_SCORE_START}标注，结尾由{FACT_SCORE_END}标注。

在生成优化后的reasoning process时，请注意您需要同时输出优化后的reasoning process以及您所估计的truthfulity score，faithfulness score和factuality score。具体格式需要参考History中的reasoning process，具体如下：
（1）生成的优化后的reasoning process实例包括具体文本内容和相应打分开头由{OPT_START_TOKEN}标注，结尾由{OPT_END_TOKEN}标注
（2）每一个reasoning process具体文本内容的开头由{PART_START}标注，结尾由{PART_END}标注，后面紧接着给出您所估计的整体的truthfulity score开头由{TRUTH_SCORE_START}标注，结尾由{TRUTH_SCORE_END}标注
（3）每一个reasoning process具体文本内容分为多个step，每个step的具体文本内容开头由{STEP_START}标注，结尾由{STEP_END}标注，每个step文本需要给出标题与标号。step的标号从1开始，每个标题的开头由\"{TOP_START}\"标注，结尾由\"{TOP_END}\"标注。每个step的文本内容结束后，紧接着需要给出您所估计的该step的faithfulness score和factuality score，faithtulity score的开头由{FAITH_SCORE_START}标注，结尾由{FAITH_SCORE_END}标注；factuality score的开头由{FACT_SCORE_START}标注，结尾由{FACT_SCORE_END}标注。
（4）生成的优化后的reasoning process的具体文本内容请以汉语为主体输出。
（5）请注意，您生成的step的数量和标题不需要和History中的reasoning process相同。
""",
                               }

EVO_META_PROMPT_ENG_UNSUPER = {
    "gpt": f"""
Hello, your task is to observe the provided Documents, Question, and Answer, and refer to the reasoning processes and their quantitative scores given in History to generate what you consider an optimized reasoning process that could result in higher scores. Specifically, each generated reasoning process is your conception of the thought process or logical reasoning (Reasoning Process) that the LLMs follows to better approximate the provided target Answer when answering the Question based on the Documents.
Please adhere to the following requirements when generating the optimized reasoning process:
1. Think step by step when creating the reasoning process. This process should mirror what you consider the LLMs's way of thinking (thought) or the logic it follows.
2. The reasoning process you come up with should aim to be an improvement on the reasoning processes previously provided in History, without reiterating them. Specifically, we have rated each reasoning process in History across three quantitative metrics (scores):
   (1) Truthfulness score: A scale from 0 to 100, this score reflects the similarity between the answer generated by the LLMs using the respective reasoning process and the standard correct Answer. A higher score suggests a more truthful alignment of the reasoning process with the correct Answer. In essence, the higher the score, the closer the text meaning or semantic entailment of the result produced by the reasoning process aligns with the correct Answer.
   (2) Faithfulness score: Ranging from 0 to 100 for each step of the corresponding reasoning process, with higher scores being preferable. The score for each step measures how faithfully the LLMs adopts the reasoning logic of that particular step when generating an answer to the Question. This means the higher the faithfulness score, the more closely the corresonding step reflects the LLMs's thinking style.
   (3) Factuality score: Ranging from 0 to 100 for each reasoning process step, with higher scores being ideal. This score for each step represents the degree of factual correctness of the content described in that step. Higher scores indicate that the reasoning used by the LLMs is more grounded in the provided Documents or Question, and therefore less reliant on speculation by the LLMs itself.
With these points in mind, ensure that the overall reasoning process you generate can achieve a higher truthfulness score. On this basis, strive to also improve the faithfulness score and factuality score for each step of your optimized reasoning process. Additionally, be aware that the number and titles of steps you create do not have to match those found in the reasoning process in History.
3. Please use English to output the optimized reasoning process.
4. Understand that the purpose of creating a reasoning process is to explain the LLMs's way of thinking or reasoning as it generates answers, which in turn can guide the LLMs to reference the Documents as accurately as possible when addressing the Question and obtaining the correct Answer.
5. Please ensure not to directly respond to the Question within the generated reasoning process.
6. In the History input for this session, the format of each previously generated reasoning process provided is as follows:
(1) Each reasoning process instance includes specific text content and respective scores, marked at the start with {HIST_START_TOKEN} and at the end with {HIST_END_TOKEN}.
(2) The beginning of each specific text content of a reasoning process is marked with {PART_START}, and the end with {PART_END}, followed immediately by the overall truthfulness score starting with {TRUTH_SCORE_START} and ending with {TRUTH_SCORE_END}.
(3) Each reasoning process's specific text content is divided into multiple steps; for each step, the specific text content starts with {STEP_START} and ends with {STEP_END}, and each step gives a title and number (starting from 1). Following the text content of each step, the faithfulness score and factuality score for that step are provided, with the faithfulness score starting with {FAITH_SCORE_START} and ending with {FAITH_SCORE_END}, and the factuality score starting with {FACT_SCORE_START} and ending with {FACT_SCORE_END}.

When generating the optimized reasoning process, please keep in mind that you need to output both the optimized reasoning process and your estimated truthfulness score, faithfulness score, and factuality score. The specific format should reference the reasoning process in History, as detailed below:
(1) The instance of the optimized reasoning process generated includes specific text content and corresponding scores, starting with {OPT_START_TOKEN} and ending with {OPT_END_TOKEN}.
(2) The beginning of each specific text content of a reasoning process is marked with {PART_START}, and the end with {PART_END}, followed immediately by your estimated overall truthfulness score starting with {TRUTH_SCORE_START} and ending with {TRUTH_SCORE_END}.
(3) Each reasoning process's specific text content is divided into multiple steps; for each step, the specific text content starts with {STEP_START} and ends with {STEP_END}, with each step requiring a title and number. The numbering of steps starts at 1, with each title starting with "{TOP_START}" and ending with "{TOP_END}". Following the text content of each step, you need to immediately provide your estimated faithfulness score and factuality score for that step, with the faithfulness score starting with {FAITH_SCORE_START} and ending with {FAITH_SCORE_END}, and the factuality score starting with {FACT_SCORE_START} and ending with {FACT_SCORE_END}.
(4) The specific text content of the optimized reasoning process should be output primarily in English.
(5) Note that the number and titles of steps you generate do not need to match those in the reasoning process in History.
""",

    "qwen": f"""
Hello, your task is to review the given Documents and Question, and to reference the reasoning processes within the provided History as well as their quantitative scores (score), in order to generate an optimized reasoning process that you believe could result in higher scores. Specifically, each reasoning process you create should represent your conception of the large model's thought process or logical reasoning (Reasoning Process) it would use to answer Questions based on the given Documents. 
Keep in mind the following requirements when generating the optimized reasoning process:
1. Generate the reasoning process step by step. This process should reflect your view of the LLMs's way of thinking or logic.
2. The reasoning process you provide should be an optimized version of those previously generated in History, without duplicating them. Specifically, each reasoning process in History has been assessed on three dimensions:
   (1) Truthfulness score: Between 0 to 100, this score indicates the similarity between the answer generated through the reasoning process and the standard correct Answer when answering the Question based on the Documents. The higher the score, the more truthful the reasoning process is at guiding the answerer to a response similar to the standard correct Answer. The higher the score, the greater the textual meaning or semantic entailment of the result generated by the reasoning process aligns with the standard correct Answer.
   (2) Faithfulness score: Also within the range of 0 to 100, each step of the reasoning process is given a faithfulness score, with higher scores being better. Each step's score reflects the degree of faithfulness that step has in faithfully generating a response similar to the standard correct Answer. The higher this score, the greater the contribution of that step of the reasoning process in approximating the standard correct Answer.
   (3) Factuality score: Also from 0 to 100, this score is given to each step of the reasoning process. The higher the score, the greater the degree of factuality of the content expressed in that step of the reasoning process. A higher score suggests a stronger reliance on the provided Documents or Question, and less on one's own guessing.
Taking the above into consideration, the overall reasoning process you create should strive for a higher truthfulness score. In doing so, aim to maximize the faithfulness score and factuality score for each step of the optimized reasoning process. Remember that the number and titles of the steps you create don’t need to match those found in the reasoning process in History.
3. Please output the optimized reasoning process in English.
4. Note that the purpose of creating a reasoning process is to explain the thinking or method used by the large model in generating answers. This explanation should be capable of guiding the large model to refer to the Documents correctly when answering the Question, in order to provide a correct Answer.
5. Make sure not to directly answer the Question in the reasoning process you generate.
6. In the input history provided, each previously generated reasoning process follows the format below: (1) Each reasoning process instance includes specific text content and its corresponding score, marked by {HIST_START_TOKEN} at the beginning and {HIST_END_TOKEN} at the end. (2) The beginning of each reasoning process's specific text content is marked by {PART_START} and ends with {PART_END}, followed immediately by the overall truthfulness score marked at the beginning by {TRUTH_SCORE_START}, and at the end by {TRUTH_SCORE_END}. (3) The specific text content of each reasoning process is composed of multiple steps, each step's specific text content starts with {STEP_START} and ends with {STEP_END}, each step text provides a title and a number (starting from 1). After each step's text content, the step's faithfulness score and factuality score will be provided; the beginning of the faithfulness score is marked by {FAITH_SCORE_START} and ends with {FAITH_SCORE_END}; the beginning of the factuality score is marked by {FACT_SCORE_START} and ends with {FACT_SCORE_END}.

When generating an optimized reasoning process, please note that you need to output both the optimized reasoning process and your estimated truthfulness score, faithfulness score, and factuality score. The specific format should refer to the reasoning process in the History, as follows:
(1) The optimized reasoning process instance includes specific text content and corresponding scores marked at the beginning by {OPT_START_TOKEN} and at the end by {OPT_END_TOKEN}.
(2) The beginning of each reasoning process’s specific text content starts with {PART_START} and ends with {PART_END}, followed immediately by your estimated overall truthfulness score, starting with {TRUTH_SCORE_START} and ending with {TRUTH_SCORE_END}.
(3) The specific text content of each reasoning process is divided into multiple steps, each step's specific text content starts with {STEP_START} and ends with {STEP_END}, each step text needs to provide a title and number. The numbering of steps starts from 1, with each title marked at the beginning by "{TOP_START}" and at the end by "{TOP_END}". After each step's text content, you need to provide your estimated faithfulness score and factuality score; the beginning of the faithfulness score is marked by {FAITH_SCORE_START} and ends with {FAITH_SCORE_END}; the beginning of the factuality score is marked by {FACT_SCORE_START} and ends with {FACT_SCORE_END}.
(4) Please produce the specific text content of the optimized reasoning process primarily in English.
(5) Please note, the number and titles of steps you generate do not need to match those in the history reasoning process.
"""
}

EVO_META_PROMPT_NO_WEIGHT_CHN_UNSUPER = {"gpt": f"""您好，您的任务是观察给定的Documents，Question，并参考提供的History中的reasoning process以及对这些reasoning process的量化评分 (score)，生成您认为的优化后的即可以获得更高分数的reasoning process。具体来说，每一个生成的reasoning process即为您所认为的大模型在根据Documents回答Question时，大模型所使用或遵循的思维方式 (thought) 或逻辑推理过程 (Reasoning Process)。
请注意，在生成优化的reasoning process时您需要遵循以下的要求：
1. 请您 think step by step地来生成reasoning process。在这一过程中体现您认为的大模型的思维（thought）方式或思考逻辑。
2. 请您注意，您所实现的reasoning process需要尽量是对History提供的先前生成的reasoning process进行优化后的结果,且不能重复History中的reasoning process。具体来说，我们对于History中提供的每一个reasoning process同时进行了3个维度的量化评分 (score):
(1) truthfulity score: 范围在0~100之间，该分数代表了大模型使用相对应的reasoning process来引导大模型根据提供的Documents回答Question时，其生成的回答结果与标准正确答案Answer之间的相似度，分数越高越好。该分数越高则代表使用相应reasoing process时可以越真实的（truthful）引导大模型生成与标准正确答案Answer相似的回答。具体来说，该分数越高，相应reasoning prcocess引导大模型生成的结果的文本含义或语义蕴含（entail）于标准正确答案Answer的程度越高，且生成结果与标准正确答案Answer的语义与结构越接近。
(2) faithfulness score: 范围在0~100之间，对于相应的reasoning process的每一步(each step) 都会给出faithfulness score，越高越好。每一步（each step）的该分数代表了大模型在生成对Question的回答时，对该分数的所对应的reasoning prcocess的相应step的忠诚度（faithfulity）。具体来说，该分数越高代表着大模型在回答问题时忠实地（faithfuly）采用相应step的推理逻辑的程度越高（也就是说该step越忠实地反映了大模型的思维方式）。
(3) factuality score: 范围在0~100之间，对于相应的reasoning process的每一步(each step) 都会给出factuality score，越高越好。每一步（each step）的该分数代表了相应的reasoning prcocess的每一步（each step）所表述内容的事实性（factual）程度。具体来说，该分数越高代表了相应reasoning process的相应step的推理依据来自于对提供的Documents或Question的引用或理解的程度越高，相应地，该step的推理依据来自于大模型自身的猜测概率越低且推理依据脱离提供的Document与Question的概率越低。
综上所述，请您在生成优化的reasoning process时尽量保证该reasoning process整体可以获得更高的truthfulity score。在这一前提下，请您生成的优化的reasoning process尽量提高每一step的faithfulness score和factuality score。同时请注意，您生成的step的数量和标题不需要和History中的reasoning process相同。
3.请使用汉语来输出优化的reasoning process
4.请您注意，生成reasoning process的目的是解释大模型生成答案时的思维方式或思考方式，通过reasoning process解释的思维或思考方式，可以引导大模型使其可以尽量正确地参考Documents来回答Question，以尽量正确地获得正确Answer。
5.请您确保不要在生成的reasoning process中直接回答问题Question。
6. 在本次输入的History中，每一个提供的先前生成的reasoning process的格式如下：
（1）每一个reasoning process实例包括具体文本内容和相应打分，开头由{HIST_START_TOKEN}标注，结尾由{HIST_END_TOKEN}标注
（2）每一个reasoning process具体文本内容的开头由{PART_START}标注，结尾由{PART_END}标注，后面紧接着会给出整体的truthfulity score开头由{TRUTH_SCORE_START}标注，结尾由{TRUTH_SCORE_END}标注
（3）每一个reasoning process具体文本内容分为多个step，每个step的具体文本内容开头由{STEP_START}标注，结尾由{STEP_END}标注，每个step文本会给出标题与标号（从1开始）。每个step文本内容结束后，紧接着会给出该step的faithfulness score和factuality score，faithtulity score的开头由{FAITH_SCORE_START}标注，结尾由{FAITH_SCORE_END}标注；factuality score的开头由{FACT_SCORE_START}标注，结尾由{FACT_SCORE_END}标注。

在生成优化后的reasoning process时，请注意您需要同时输出优化后的reasoning process以及您所估计的truthfulity score，faithfulness score和factuality score。具体格式需要参考History中的reasoning process，具体如下：
（1）生成的优化后的reasoning process实例包括具体文本内容和相应打分开头由{OPT_START_TOKEN}标注，结尾由{OPT_END_TOKEN}标注
（2）每一个reasoning process具体文本内容的开头由{PART_START}标注，结尾由{PART_END}标注，后面紧接着给出您所估计的整体的truthfulity score开头由{TRUTH_SCORE_START}标注，结尾由{TRUTH_SCORE_END}标注
（3）每一个reasoning process具体文本内容分为多个step，每个step的具体文本内容开头由{STEP_START}标注，结尾由{STEP_END}标注，每个step文本需要给出标题与标号。step的标号从1开始，每个标题的开头由\"{TOP_START}\"标注，结尾由\"{TOP_END}\"标注。每个step的文本内容结束后，紧接着需要给出您所估计的该step的faithfulness score和factuality score，faithtulity score的开头由{FAITH_SCORE_START}标注，结尾由{FAITH_SCORE_END}标注；factuality score的开头由{FACT_SCORE_START}标注，结尾由{FACT_SCORE_END}标注。
（4）生成的优化后的reasoning process的具体文本内容请以汉语为主体输出。
（5）请注意，您生成的step的数量和标题不需要和History中的reasoning process相同。
""",
                                         "qwen": f"""您好，您的任务是观察给定的Documents，Question，并参考提供的History中的reasoning process以及对这些reasoning process的量化评分 (score)，生成您认为的优化后的即可以获得更高分数的reasoning process。具体来说，每一个生成的reasoning process即为您所认为的大模型在根据Documents回答Question时，大模型所使用或遵循的思维方式 (thought) 或逻辑推理过程 (Reasoning Process)。
请注意，在生成优化的reasoning process时您需要遵循以下的要求：
1. 请您 think step by step地来生成reasoning process。在这一过程中体现您的思维（thought）方式或思考逻辑。
2. 请您注意，您所输出的reasoning process需要尽量是对History提供的先前生成的reasoning process进行优化后的结果，且不能重复History中的reasoning process。具体来说，我们对于History中提供的每一个reasoning process同时进行了3个维度的量化评分 (score):
(1) truthfulity score: 范围在0~100之间，该分数代表了您使用相对应的reasoning process来引导您根据提供的Documents回答Question时，其生成的回答结果与标准正确答案Answer之间的相似度，分数越高越好。该分数越高则代表使用相应reasoing process时可以越真实的（truthful）引导您生成与标准正确答案Answer相似的回答。具体来说，该分数越高，相应reasoning prcocess引导您生成的结果的文本含义或语义蕴含（entail）于标准正确答案Answer的程度越高，且生成结果与标准正确答案Answer的语义与结构越接近。
(2) faithfulness score: 范围在0~100之间，对于相应的reasoning process的每一步(each step) 都会给出faithfulness score，越高越好。每一步（each step）的该分数代表了您在生成对Question的回答时，对该分数的所对应的reasoning prcocess的相应step的忠诚度（faithfulity）。具体来说，该分数越高代表着您在回答问题时忠实地（faithfuly）采用相应step的推理逻辑的程度越高（也就是说该step越忠实地反映了您的思维方式）。
(3) factuality score: 范围在0~100之间，对于相应的reasoning process的每一步(each step) 都会给出factuality score，越高越好。每一步（each step）的该分数代表了相应的reasoning prcocess的每一步（each step）所表述内容的事实性（factual）程度。具体来说，该分数越高代表了相应reasoning process的相应step的推理依据来自于对提供的Documents或Question的引用或理解的程度越高，相应地，该分数越高，则该step的推理依据来自于您自身的猜测概率越低且推理依据脱离提供的Document与Question的概率越低。
综上所述，请您在生成优化的reasoning process时尽量保证该reasoning process整体可以获得更高的truthfulity score。在这一前提下，请您生成的优化的reasoning process尽量提高每一step的faithfulness score和factuality score。同时请注意，您生成的step的数量和标题不需要和History中的reasoning process相同。
3.请使用汉语来输出优化的reasoning process
4.请您注意，生成reasoning process的目的是解释大模型生成答案时的思维方式或思考方式，通过reasoning process解释的思维或思考方式，可以引导大模型使其可以尽量正确地参考Documents来回答Question，以尽量正确地获得正确Answer。
5.请您确保不要在生成的reasoning process中直接回答问题Question。
6. 在本次输入的History中，每一个提供的先前生成的reasoning process的格式如下：
（1）每一个reasoning process实例包括具体文本内容和相应打分，开头由{HIST_START_TOKEN}标注，结尾由{HIST_END_TOKEN}标注
（2）每一个reasoning process具体文本内容的开头由{PART_START}标注，结尾由{PART_END}标注，后面紧接着会给出整体的truthfulity score开头由{TRUTH_SCORE_START}标注，结尾由{TRUTH_SCORE_END}标注
（3）每一个reasoning process具体文本内容分为多个step，每个step的具体文本内容开头由{STEP_START}标注，结尾由{STEP_END}标注，每个step文本会给出标题与标号（从1开始）。每个step文本内容结束后，紧接着会给出该step的faithfulness score和factuality score，faithtulity score的开头由{FAITH_SCORE_START}标注，结尾由{FAITH_SCORE_END}标注；factuality score的开头由{FACT_SCORE_START}标注，结尾由{FACT_SCORE_END}标注。

在生成优化后的reasoning process时，请注意您需要同时输出优化后的reasoning process以及您所估计的truthfulity score，faithfulness score和factuality score。具体格式需要参考History中的reasoning process，具体如下：
（1）生成的优化后的reasoning process实例包括具体文本内容和相应打分开头由{OPT_START_TOKEN}标注，结尾由{OPT_END_TOKEN}标注
（2）每一个reasoning process具体文本内容的开头由{PART_START}标注，结尾由{PART_END}标注，后面紧接着给出您所估计的整体的truthfulity score开头由{TRUTH_SCORE_START}标注，结尾由{TRUTH_SCORE_END}标注
（3）每一个reasoning process具体文本内容分为多个step，每个step的具体文本内容开头由{STEP_START}标注，结尾由{STEP_END}标注，每个step文本需要给出标题与标号。step的标号从1开始，每个标题的开头由\"{TOP_START}\"标注，结尾由\"{TOP_END}\"标注。每个step的文本内容结束后，紧接着需要给出您所估计的该step的faithfulness score和factuality score，faithtulity score的开头由{FAITH_SCORE_START}标注，结尾由{FAITH_SCORE_END}标注；factuality score的开头由{FACT_SCORE_START}标注，结尾由{FACT_SCORE_END}标注。
（4）生成的优化后的reasoning process的具体文本内容请以汉语为主体输出。
（5）请注意，您生成的step的数量和标题不需要和History中的reasoning process相同。
""",
                                         }

EVO_META_PROMPT_NO_WEIGHT_ENG_UNSUPER = {
    "gpt": f"""
Hello, your task is to observe the given Documents and Question, and refer to the reasoning process provided in the History as well as the quantitative scoring (score) of these reasoning processes to generate what you believe to be an optimized reasoning process that can obtain a higher score. Specifically, each generated reasoning process is what you consider to be the thought process or logical reasoning (Reasoning Process) that the large model uses or follows when answering the Question according to the Documents.
Please note that when generating an optimized reasoning process, you must follow the following requirements:

1. Please think step by step to generate the reasoning process. This process should reflect your opinion of the thought pattern or logical thinking of the large model.
2. Please note that your implemented reasoning process should be an optimized result of the reasoning processes provided in the History and must not repeat the reasoning processes in the History. Specifically, for each reasoning process provided in the History, we have simultaneously conducted quantitative scoring in 3 dimensions: 
(1) Truthfulness score: Ranging between 0 to 100, this score represents the similarity between the answer generated by the large model using the corresponding reasoning process to answer the Question based on the provided Documents, and the standard correct answer (Answer). The higher the score, the better. A higher score means that the corresponding reasoning process can more truthfully guide the large model to generate an answer similar to the standard correct Answer. In other words, the higher the score, the more the meaning or semantics of the result generated by the corresponding reasoning process entail the standard correct Answer, and the closer the generated result's semantics and structure to the standard correct Answer. 
(2) Faithfulness score: Ranging between 0 to 100, a faithfulness score will be given for each step of the corresponding reasoning process; the higher, the better. The score for each step indicates the model's faithfulness when generating an answer to the Question concerning the corresponding step of the reasoning process. In essence, the higher the score, the more faithfully the model adopts the reasoning logic of the corresponding step when answering the question—that is, the step more faithfully reflects the actual reasoning process of the model. 
(3) Factuality score: Ranging between 0 to 100, a factuality score will be given for each step of the corresponding reasoning process; the higher, the better. The score for each step indicates the factual degree of the content expressed in the corresponding step of the reasoning process. Specifically, the higher the score, the more the reasoning basis of the corresponding step of the reasoning process is derived from referencing or understanding the provided Documents or Questions, correspondingly, the lower the probability that the reasoning is based on the model's own guesswork and the lower the probability that the reasoning deviates from the provided Document and Question. Summing up, when generating an optimized reasoning process, please ensure that the overall reasoning process can achieve a higher truthfulness score. Under this premise, please try to increase the faithfulness score and factuality score for each step of the optimized reasoning process you generate. Also, note that the number of steps you generate and their titles do not need to match those in the history reasoning process.
3. Please output the optimized reasoning process in English.
4. Please note that the purpose of generating the reasoning process is to explain the thought pattern or thinking process of the large model in generating answers. The thought or thinking process explained by the reasoning process can guide the large model to correctly refer to the Documents to answer the Question, in order to correctly obtain the correct Answer.
5. Please ensure not to directly answer the Question in the reasoning process you generate.
6. In the history input for this session, each previously generated reasoning process is formatted as follows: (1) Each reasoning process instance includes specific text content and corresponding scores, marked at the beginning by {HIST_START_TOKEN} and at the end by {HIST_END_TOKEN}. (2) The beginning of each reasoning process's specific text content is marked by {PART_START} and ends with {PART_END}, followed immediately by the overall truthfulness score marked at the beginning by {TRUTH_SCORE_START} and at the end by {TRUTH_SCORE_END}. (3) The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}. Each step text provides a title and a number (starting from 1). After the end of each step's text content, the step's faithfulness score and factuality score will be given, marked at the beginning by {FAITH_SCORE_START} and at the end by {FAITH_SCORE_END}; and the factuality score marked at the beginning by {FACT_SCORE_START} and at the end by {FACT_SCORE_END}.

When generating an optimized reasoning process, please note that you need to output both the optimized reasoning process and your estimated truthfulness score, faithfulness score, and factuality score. You should refer to the specific format of the reasoning process in the History as follows:
(1) The optimized reasoning process instance includes specific text content and corresponding scores, marked at the beginning by {OPT_START_TOKEN} and at the end by {OPT_END_TOKEN}.
(2) The beginning of each reasoning process's specific text content starts with {PART_START} and ends with {PART_END}, followed immediately by your estimated overall truthfulness score, starting with {TRUTH_SCORE_START} and ending with {TRUTH_SCORE_END}.
(3) The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}. Each step text needs to provide a title and a number. The numbering of steps starts from 1, with each title marked at the beginning by "{TOP_START}" and at the end by "{TOP_END}". After each step's text content, you will need to provide your estimated faithfulness score and factuality score, marked at the beginning by {FAITH_SCORE_START} and at the end by {FAITH_SCORE_END}, and the factuality score marked at the beginning by {FACT_SCORE_START} and at the end by {FACT_SCORE_END}.
(4) The specific text content of the optimized reasoning process should be primarily output in English.
(5) Please note that the number and titles of steps you generate do not need to match those in the history reasoning process.
""",

    "qwen": f"""
Hello, your task is to observe the given Documents and Question, and refer to the reasoning processes and their quantitative scores (scores) provided in the History, to generate an optimized reasoning process that you believe can obtain a higher score. Specifically, each reasoning process you generate represents what you consider to be the large model's thought pattern or logical reasoning process used to answer the Question according to the Documents.
Please note that when generating the optimized reasoning process, you need to follow the following requirements:
1. Please generate the reasoning process by thinking step-by-step. This process should reflect your thought process or logical reasoning.
2. Please note that the reasoning process you output should be an optimized result of the reasoning processes previously generated in the History and should not repeat the reasoning processes in the History. Specifically, we have performed a quantitative scoring in three dimensions for each reasoning process provided in the History:
    (1) Truthfulness score: Ranging from 0 to 100, this score represents the similarity between the answer you generate using the corresponding reasoning process to answer the Question based on the provided Documents, and the standard correct Answer. The higher the score, the better. A higher score means that the corresponding reasoning process can guide you more truthfully to generate an answer similar to the standard correct Answer. Specifically, the higher the score, the more the text meaning or semantic entailment of the result generated by the corresponding reasoning process corresponds to the standard correct Answer, and the closer the semantics and structure of the generated result to the standard correct Answer.
    (2) Faithfulness score: Ranging from 0 to 100, a faithfulness score will be given for each step of the corresponding reasoning process; the higher, the better. The score for each step represents your faithfulness when generating an answer to the Question relative to the corresponding step of the reasoning process. Specifically, the higher the score, the more faithfully you adopt the reasoning logic of the corresponding step when answering the question—that is, the step more faithfully reflects your way of thinking.
    (3) Factuality score: Ranging from 0 to 100, a factuality score will be given for each step of the corresponding reasoning process; the higher, the better. The score for each step represents the factual degree of the content expressed in the corresponding step of the reasoning process. Specifically, the higher the score, the more the reasoning basis of the corresponding step of the reasoning process is derived from referencing or understanding the provided Documents or Question. Correspondingly, the higher the score, the lower the probability that the reasoning is based on your own guesswork and the lower the probability that the reasoning deviates from the provided Document and Question.
Summing up, when generating an optimized reasoning process, please ensure that you can obtain a higher truthfulness score for the overall reasoning process. Under this premise, please try to improve the faithfulness score and factuality score for each step of the optimized reasoning process you generate. Also, note that the number of steps you generate and their titles do not need to match those in the history reasoning process.
3. Please output the optimized reasoning process in English.
4. Please note that the purpose of generating the reasoning process is to explain the thought pattern or mode of thinking used by the large model when generating answers. The thought or mode of thinking explained through the reasoning process can guide the large model to refer to the Documents correctly to answer the Question, to obtain the correct Answer as accurately as possible.
5. Please ensure not to directly answer the Question in the reasoning process you generate.
6. In the history provided for this input, the format for each of the previously generated reasoning processes is as follows: (1) Each reasoning process instance consists of the specific text content and corresponding scores, starting with {HIST_START_TOKEN} and ending with {HIST_END_TOKEN}. (2) The beginning of each reasoning process's specific text content is marked by {PART_START} and ends with {PART_END}, followed immediately by the overall truthfulness score marked at the start with {TRUTH_SCORE_START} and at the end with {TRUTH_SCORE_END}. (3) The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}. Each step text includes a title and a number (starting from 1). After the text content of each step, the step's faithfulness score and factuality score are given, with the beginning of the faithfulness score marked by {FAITH_SCORE_START} and the end by {FAITH_SCORE_END}; the factuality score starts with {FACT_SCORE_START} and ends with {FACT_SCORE_END}.

When generating the optimized reasoning process, please note that you need to also output the optimized reasoning process along with your estimated truthfulness score, faithfulness score, and factuality score. The specific format should refer to the reasoning processes in the History, outlined as follows:
(1) The instance of the generated optimized reasoning process includes the specific text content and corresponding scores, starting with {OPT_START_TOKEN} and ending with {OPT_END_TOKEN}.
(2) The beginning of each reasoning process’s specific text content is marked by {PART_START} and ends with {PART_END}, followed immediately by your estimated overall truthfulness score, starting with {TRUTH_SCORE_START} and ending with {TRUTH_SCORE_END}.
(3) Each reasoning process’s specific text content is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}. Each step text needs to provide a title and a number. Step numbering starts from 1, with each title starting with "{TOP_START}" and ending with "{TOP_END}". After the text content of each step, you need to provide your estimated faithfulness score and factuality score; the faithfulness score starts with {FAITH_SCORE_START} and ends with {FAITH_SCORE_END}; the factuality score starts with {FACT_SCORE_START} and ends with {FACT_SCORE_END}.
(4) Please generate the specific text content of the optimized reasoning process primarily in English.
(5) Please note that the number and titles of steps you generate do not need to match those in the history reasoning process.
"""
}

import numpy as np


def generate_history_reasoning_process_item(input_pdf, step_id_col="step_id",
                                            line_id_col="line_id", step_title_col="step_title",
                                            faith_col="faith_score_hyb",
                                            truth_col="truth_score_hyb",
                                            fact_col="fact_score",
                                            hist_start_token=HIST_START_TOKEN,
                                            hist_end_token=HIST_END_TOKEN,
                                            truth_start_token=TRUTH_SCORE_START,
                                            truth_end_token=TRUTH_SCORE_END,
                                            fact_score_start=FACT_SCORE_START,
                                            fact_score_end=FACT_SCORE_END,
                                            step_start_token=STEP_START,
                                            step_end_token=STEP_END,
                                            part_start_token=PART_START,
                                            part_end_token=PART_END,
                                            faith_score_start=FAITH_SCORE_START,
                                            faith_score_end=FAITH_SCORE_END,
                                            score_unit=100,use_reliability=True,
                                            use_fidelity=True,
                                            use_factual=True):
    in_pdf = input_pdf.copy()
    in_pdf = in_pdf.sort_values(step_id_col).reset_index(drop=True)
    in_pdf_dict = {}
    for step_id in in_pdf[step_id_col].unique().tolist():
        in_pdf_dict[step_id] = {}
        in_pdf_dict[step_id]["source"] = in_pdf[in_pdf[step_id_col].isin([step_id])].reset_index(drop=True)
        # truthfulity

    truth_score = in_pdf[truth_col].mean() * score_unit

    def generate_reasoning_step(input_dict, line_id_col=line_id_col, step_id_col=step_id_col):
        in_dict = input_dict.copy()
        step_id_list = list(in_dict.keys())
        step_id_list.sort()

        for sid in step_id_list:
            step_in_pdf = input_dict[sid]["source"].copy()

            step_in_pdf = step_in_pdf.sort_values(line_id_col).reset_index(drop=True)
            step_content_dict = {}
            step_content_dict["top"] = step_in_pdf[step_title_col].values[0]
            step_content_dict["content"] = step_in_pdf["content"].to_list()
            in_dict[sid]["content"] = construct_modified_step(input_step_params=step_content_dict,
                                                              give_id=sid + 1, top_start="**", top_end="**")
            tmp_step_content = in_dict[sid]["content"].strip()
            in_dict[sid]["content"] = f"{step_start_token}\n{tmp_step_content}\n{step_end_token}"
            step_faith_score = step_in_pdf[faith_col].mean() * score_unit
            step_fact_score = step_in_pdf[fact_col].mean() * score_unit

            if use_fidelity:
                step_faith_item = f"{faith_score_start}faithfulness score: {step_faith_score}{faith_score_end}"
            else:
                step_faith_item = ""

            if use_factual:
                step_fact_item = f"{fact_score_start}factuality score: {step_fact_score}{fact_score_end}"
            else:
                step_fact_item = ""

            if step_faith_item:
                step_faith_item = f"{step_faith_item}\n"

            if step_fact_item:
                step_fact_item = f"{step_fact_item}\n"

            tmp_step_content = in_dict[sid]["content"].strip()
            in_dict[sid]["content"] = f"{tmp_step_content}\n{step_faith_item}{step_fact_item}"

            # in_dict[sid]["content"] = f"{tmp_step_content}\n{step_faith_item}\n{step_fact_item}\n"
        return in_dict

    in_pdf_dict = generate_reasoning_step(input_dict=in_pdf_dict)
    step_id_list = list(in_pdf_dict.keys())
    step_id_list.sort()
    reasoning_out_content = ""
    for sid in step_id_list:
        reasoning_out_content += in_pdf_dict[sid]["content"]
    reasoning_out_content = reasoning_out_content.strip()
    reasoning_out_content = f"{part_start_token}\n{reasoning_out_content}\n{part_end_token}"
    if use_reliability:
        truth_score_part = f"{truth_start_token}truthfulness score: {truth_score}{truth_end_token}"
    else:
        truth_score_part = ""

    if truth_score_part:
        truth_score_part = f"{truth_score_part}\n"

    reasoning_out_content = f"{reasoning_out_content}\n{truth_score_part}"
    reasoning_out_content = f"{hist_start_token}\n{reasoning_out_content}{hist_end_token}"
    return reasoning_out_content


def generate_prompt_for_evo(hist_rp_set, question_body, answer_col="manual_answer",
                            reference_col="reference_body2",
                            hist_item_limit=10, model_type="gpt",
                            template="[PROMPT] [INSTRUCTION]",
                            faith_col="faith_score_hyb",
                            truth_col="truth_score_hyb",
                            fact_col="fact_score",
                            is_truth=True, is_super=True,
                            use_reliability=True,
                            use_factual=True,use_fidelity=True
                            ):
    aim_hist_rp_process = hist_rp_set[question_body]

    aim_hist_item_key_pdf = {"score": [], "hist_id": []}
    model_type = model_type.lower()
    hist_item_num = np.min([hist_item_limit, len(list(aim_hist_rp_process.keys()))])
    for jk in list(aim_hist_rp_process.keys()):
        aim_hist_item_key_pdf["hist_id"].append(int(jk))
        aim_hist_item_key_pdf["score"].append(aim_hist_rp_process[jk]["truth_score"])

    aim_hist_item_key_pdf = pd.DataFrame(aim_hist_item_key_pdf)
    aim_hist_item_key_pdf = aim_hist_item_key_pdf.sort_values("score", ascending=False).reset_index(drop=True)
    aim_hist_item_key_pdf = aim_hist_item_key_pdf.head(hist_item_num).reset_index(drop=True)
    constructed_history_items = {}

    answer_body = ""
    reference_body = ""

    for aim_id in range(aim_hist_item_key_pdf.shape[0]):
        aim_hist_id = aim_hist_item_key_pdf["hist_id"].values[aim_id]

        if aim_id == 0:
            answer_body = aim_hist_rp_process[aim_hist_id]["source"][answer_col].values[0]
            reference_body = aim_hist_rp_process[aim_hist_id]["source"][reference_col].values[0]

        constructed_history_items[aim_hist_id] = generate_history_reasoning_process_item(
            input_pdf=aim_hist_rp_process[aim_hist_id]["source"],
            step_id_col="step_id",
            line_id_col="line_id", step_title_col="step_title",
            faith_col=faith_col,
            truth_col=truth_col,
            fact_col=fact_col,
            use_reliability=use_reliability, use_fidelity=use_fidelity, use_factual=use_factual)

    constructed_history_items_for_input = "本次进行reasoning process优化需要参考的先前生成的reasoning process，即History为: \n"
    for aim_id in range(aim_hist_item_key_pdf.shape[0]):
        aim_hist_id = aim_hist_item_key_pdf["hist_id"].values[aim_id]
        constructed_history_single_item = constructed_history_items[aim_hist_id]

        constructed_history_single_item = f"reasoning process {aim_id + 1}:\n{constructed_history_single_item}\n\n"

        constructed_history_items_for_input += constructed_history_single_item

    if hist_item_num > 1:
        if use_reliability:
            constructed_history_items_for_input += f"以上的{hist_item_num}个reasoning process实例即为本次参考的History，按照不同reasoning process的truthfulness score由高到低排列"
        else:
            if use_factual and use_fidelity:
                constructed_history_items_for_input += f"以上的{hist_item_num}个reasoning process实例即为本次参考的History，按照不同reasoning process的平均 factuality score与平均 faithfulness score的平均值由高到低排列"
            elif use_factual and not use_fidelity:
                constructed_history_items_for_input += f"以上的{hist_item_num}个reasoning process实例即为本次参考的History，按照不同reasoning process的平均 factuality score由高到低排列"
            else:
                constructed_history_items_for_input += f"以上的{hist_item_num}个reasoning process实例即为本次参考的History，按照不同reasoning process的平均 faithfulness score由高到低排列"

    else:
        constructed_history_items_for_input += f"以上的{hist_item_num}个reasoning process实例即为本次参考的History"

    if is_super:
        user_prompt_instruction = 'Specifically, for this optimization of the reasoning process, the reference materials and documents provided by the user are as follows Documents:\n' + reference_body + '\n The specific target question input by the user is: ' + question_body + '.\n The target Answer provided by the user is: ' + answer_body + "\n" + constructed_history_items_for_input
    else:
        user_prompt_instruction = 'Specifically, for this optimization of the reasoning process, the reference materials and documents provided by the user are as follows Documents:\n' + reference_body + '\n The specific target question input by the user is: ' + question_body + '.\n ' + constructed_history_items_for_input

    if is_super:
        if is_truth:
            use_dimen_num = 0
            score_meaning = ""
            use_dimens = ""
            truth_dimen = ""
            faith_dimen = ""
            fact_dimen = ""

            if use_reliability:
                use_dimen_num += 1
                # truthfulness score, faithfulness score and factuality score
                use_dimens += "truthfulness score"
                truth_dimen = f"({use_dimen_num}) truthfulness score: ranging from 0 to 100, this score represents the similarity between the answer generated by guiding you with the corresponding reasoning process when answering the Question based on the provided Documents and the provided target Answer, the higher the better. The higher the score, the more truthfully (truthful) it guides you to generate an answer similar to the target Answer. Specifically, the higher the score, the higher the degree to which the text meaning or semantic implication (entail) of the result generated by the corresponding reasoning process is at the target Answer, and the closer the semantics of the generated result are to the target Answer."

            if use_fidelity:
                use_dimen_num += 1
                if not use_reliability:
                    use_dimens += "faithfulness score"
                else:
                    use_dimens += ", faithfulness score"
                faith_dimen = f"({use_dimen_num}) faithfulness score: ranging from 0 to 100, a faithfulness score is given for each step of the corresponding reasoning process, the higher the better. This score for each step (each step) represents the degree of fidelity (faithfulness) of the corresponding reasoning process step when you truthfully (truthful) generate an answer similar to the target Answer. Specifically, the higher the score, the greater the contribution of the corresponding reasoning process step to the goal of approaching the target Answer, that is, the higher the score represents the higher degree of fidelity (faithfuly) in employing the reasoning logic of the corresponding step when answering the question, and the greater the benefit of approaching the target Answer due to using the logic of that step."
            if use_factual:
                use_dimen_num += 1
                if not use_reliability and not use_fidelity:
                    use_dimens += "factuality score"
                else:
                    use_dimens += " and factuality score"
                fact_dimen = f"({use_dimen_num}) factuality score: ranging from 0 to 100, a factuality score is given for each step of the corresponding reasoning process, the higher the better. This score for each step (each step) represents the degree of factuality of the content expressed in each step of the corresponding reasoning process. Specifically, the higher the score, the more the reasoning basis of the corresponding reasoning process step comes from the degree of reference or understanding of the provided Documents or Question, accordingly, the higher the score, the lower the probability that the reasoning basis of the step comes from your own guess and the lower the probability of the reasoning basis deviating from the provided Document and Question."

            score_meaning = f'''
            {truth_dimen}
            {faith_dimen}
            {fact_dimen}
            '''
            use_dimen_est = f"When generating the optimized reasoning process, please note that you need to output both the optimized reasoning process and your estimated {use_dimens}. The specific format should refer to the reasoning process in History, as detailed below:"
            score_meaning = score_meaning.strip()
            evo_logic = "综上所述，"
            evo_ff_structure = ""
            evo_tru_structure = ""
            out_tru_structure = ""
            out_ff_structure = ""
            if use_reliability:
                out_tru_structure = f"The beginning of each specific text content of a reasoning process is marked by {PART_START}, and the end is marked by {PART_END}, immediately followed by your estimated overall truthfulness score, starting with {TRUTH_SCORE_START} and ending with {TRUTH_SCORE_END}"
                if use_factual and use_fidelity:
                    evo_logic = f"In summary, please ensure that the optimized reasoning process as a whole can achieve a higher overall truthfulness score when generating the optimized reasoning process. Under this premise, please try to improve the faithfulness score and factuality score of each step of your optimized reasoning process. Also, please note that the number and titles of steps you generate do not need to match those in the reasoning process in History."
                    evo_tru_structure = f"The beginning of each specific text content of a reasoning process is marked by {PART_START}, and the end is marked by {PART_END}, immediately followed by the overall truthfulness score starting with {TRUTH_SCORE_START} and ending with {TRUTH_SCORE_END}"
                    evo_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, each step text will provide a title and number (starting from 1). After the text content of each step, the faithfulness score and factuality score for that step are given, with the beginning of the faithfulness score marked by {FAITH_SCORE_START} and the end by {FAITH_SCORE_END}; the factuality score starts with {FACT_SCORE_START} and ends with {FACT_SCORE_END}"
                    out_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, each step text needs to provide a title and number. The numbering of steps starts at 1, with the beginning of each title marked by \"{TOP_START}\" and the end by \"{TOP_END}\". After the text content of each step, you need to immediately provide your estimated faithfulness score and factuality score for that step, with the beginning of the faithfulness score marked by {FAITH_SCORE_START} and ending by {FAITH_SCORE_END}; the factuality score starts with {FACT_SCORE_START} and ends with {FACT_SCORE_END}"
                elif use_factual and not use_fidelity:
                    evo_logic = f"In summary, please ensure that the optimized reasoning process as a whole can achieve a higher overall truthfulness score when generating the optimized reasoning process. Under this premise, please try to improve the factuality score of each step of your optimized reasoning process. Also, please note that the number and titles of steps you generate do not need to match those in the reasoning process in History."
                    evo_tru_structure = f"The beginning of each specific text content of a reasoning process is marked by {PART_START}, and the end is marked by {PART_END}, immediately followed by the overall truthfulness score starting with {TRUTH_SCORE_START} and ending with {TRUTH_SCORE_END}"
                    evo_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, each step text will provide a title and number (starting from 1). After the text content of each step, the factuality score for that step are given, the factuality score starts with {FACT_SCORE_START} and ends with {FACT_SCORE_END}"
                    out_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, each step text needs to provide a title and number. The numbering of steps starts at 1, with the beginning of each title marked by \"{TOP_START}\" and the end by \"{TOP_END}\". After the text content of each step, you need to immediately provide your estimated factuality score for that step, the factuality score starts with {FACT_SCORE_START} and ends with {FACT_SCORE_END}"
                elif not use_factual and use_fidelity:
                    evo_logic = f"In summary, please ensure that the optimized reasoning process as a whole can achieve a higher overall truthfulness score when generating the optimized reasoning process. Under this premise, please try to improve the faithfulness score of each step of your optimized reasoning process. Also, please note that the number and titles of steps you generate do not need to match those in the reasoning process in History."
                    evo_tru_structure = f"The beginning of each specific text content of a reasoning process is marked by {PART_START}, and the end is marked by {PART_END}, immediately followed by the overall truthfulness score starting with {TRUTH_SCORE_START} and ending with {TRUTH_SCORE_END}"
                    evo_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, each step text will provide a title and number (starting from 1). After the text content of each step, the faithfulness score for that step are given, the faithfulness score starts with {FAITH_SCORE_START} and ends with {FAITH_SCORE_END}"
                    out_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, each step text needs to provide a title and number. The numbering of steps starts at 1, with the beginning of each title marked by \"{TOP_START}\" and the end by \"{TOP_END}\". After the text content of each step, you need to immediately provide your estimated faithfulness score for that step, the faithfulness score starts with {FAITH_SCORE_START} and ends with {FAITH_SCORE_END}"
                else:
                    evo_logic = f"In summary, please ensure that the optimized reasoning process as a whole can achieve a higher overall truthfulness score when generating the optimized reasoning process. Also, please note that the number and titles of steps you generate do not need to match those in the reasoning process in History."
                    evo_tru_structure = f"The beginning of each specific text content of a reasoning process is marked by {PART_START}, and the end is marked by {PART_END}, immediately followed by the overall truthfulness score starting with {TRUTH_SCORE_START} and ending with {TRUTH_SCORE_END}"
                    evo_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, each step text will provide a title and number (starting from 1)."
                    out_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, each step text needs to provide a title and number. The numbering of steps starts at 1, with the beginning of each title marked by \"{TOP_START}\" and the end by \"{TOP_END}\""

            else:
                out_tru_structure = f"The beginning of each specific text content of a reasoning process is marked by {PART_START}, and the end is marked by {PART_END}"
                evo_tru_structure = f"The beginning of each specific text content of a reasoning process is marked by {PART_START}, and the end is marked by {PART_END}"
                if use_factual and use_fidelity:
                    evo_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, each step text will provide a title and number (starting from 1). After the text content of each step, the faithfulness score and factuality score for that step are given, with the beginning of the faithfulness score marked by {FAITH_SCORE_START} and the end by {FAITH_SCORE_END}; the factuality score starts with {FACT_SCORE_START} and ends with {FACT_SCORE_END}"
                    evo_logic = "In summary, when generating the optimized reasoning process, please try to improve the faithfulness score and factuality score of each step of your optimized reasoning process. Also, please note that the number and titles of steps you generate do not need to match those in the reasoning process in History."
                    out_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, each step text needs to provide a title and number. The numbering of steps starts at 1, with the beginning of each title marked by \"{TOP_START}\" and the end by \"{TOP_END}\". After the text content of each step, you need to immediately provide your estimated faithfulness score and factuality score for that step, with the beginning of the faithfulness score marked by {FAITH_SCORE_START} and ending by {FAITH_SCORE_END}; the factuality score starts with {FACT_SCORE_START} and ends with {FACT_SCORE_END}"
                elif use_factual and not use_fidelity:
                    evo_logic = "In summary, when generating the optimized reasoning process, please try to improve the factuality score of each step of your optimized reasoning process. Also, please note that the number and titles of steps you generate do not need to match those in the reasoning process in History."
                    evo_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, each step text will provide a title and number (starting from 1). After the text content of each step, the factuality score for that step are given, the factuality score starts with {FACT_SCORE_START} and ends with {FACT_SCORE_END}"
                    out_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, each step text needs to provide a title and number. The numbering of steps starts at 1, with the beginning of each title marked by \"{TOP_START}\" and the end by \"{TOP_END}\". After the text content of each step, you need to immediately provide your estimated factuality score for that step, the factuality score starts with {FACT_SCORE_START} and ends with {FACT_SCORE_END}"

                else:
                    evo_logic = "In summary, when generating the optimized reasoning process, please try to improve the faithfulness score of each step of your optimized reasoning process. Also, please note that the number and titles of steps you generate do not need to match those in the reasoning process in History."
                    evo_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, each step text will provide a title and number (starting from 1). After the text content of each step, the faithfulness score for that step are given, the faithfulness score starts with {FAITH_SCORE_START} and ends with {FAITH_SCORE_END}"
                    out_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, each step text needs to provide a title and number. The numbering of steps starts at 1, with the beginning of each title marked by \"{TOP_START}\" and the end by \"{TOP_END}\". After the text content of each step, you need to immediately provide your estimated faithfulness score for that step, the faithfulness score starts with {FAITH_SCORE_START} and ends with {FAITH_SCORE_END}"

            evo_meta_prompt_eng_super = f"""Hello, your task is to observe the given Documents, Question, and Answer, and refer to the reasoning process in the provided History and the quantitative scoring (score) of these reasoning processes to generate an optimized reasoning process that you believe can achieve higher scores. Specifically, each generated reasoning process is what you think you need to use or follow in terms of thinking (thought) or logical reasoning process (Reasoning Process) when you approach the target Answer better as you answer the Question based on the Documents.
            Please note the following requirements when generating the optimized reasoning process:
            1. Please think step by step to generate the reasoning process. Reflect your thinking (thought) method or thinking logic during this process.
            2. Please note that the reasoning process you output should be an optimized result of the reasoning process previously provided in History, and should not repeat the reasoning process in History. Specifically, we have quantitatively scored each reasoning process provided in History in {use_dimen_num} dimensions:
            {score_meaning}
            {evo_logic}
            3. Please output the optimized reasoning process in English.
            4. Please note that the purpose of generating a reasoning process is to guide the large model to provide an answer that approximates the target Answer, regardless of whether the given answer is wrong or right.
            5. Please ensure not to answer the Question, nor to provide any optimization suggestions for the target Answer.
            6. In the input History for this session, the format of each previously generated reasoning process provided is as follows:
            (1) Each reasoning process instance includes specific text content and corresponding scores, marked at the beginning by {HIST_START_TOKEN} and at the end by {HIST_END_TOKEN}
            (2) {evo_tru_structure}
            (3) {evo_ff_structure}
            
            {use_dimen_est}
            (1) The instance of the optimized reasoning process generated includes specific text content and corresponding scores, starting with {OPT_START_TOKEN} and ending with {OPT_END_TOKEN}
            (2) {out_tru_structure}
            (3) {out_ff_structure}
            (4) Please output the specific text content of the optimized reasoning process primarily in English
            (5) Please note, the number and titles of steps you generate do not need to match those in the reasoning process in History
            """
            prompt_template = evo_meta_prompt_eng_super
        else:
            use_dimen_num = 0
            score_meaning = ""
            use_dimens = ""
            truth_dimen = ""
            faith_dimen = ""
            fact_dimen = ""

            if use_reliability:
                use_dimen_num += 1
                use_dimens += "truthfulness score"
                truth_dimen = f"({use_dimen_num}) truthfulness score: Ranging between 0 to 100, this score represents the similarity between the answer generated by using the corresponding reasoning process to answer the Question based on the Documents and the target Answer provided, with higher scores being better. The higher the score, the more truthfully the corresponding reasoning process can guide you to generate an answer similar to the target Answer. That is, the higher the score, the greater the extent to which the resulting text meaning or semantic implication (entailment) is in accordance with the target Answer."
            if use_fidelity:
                use_dimen_num += 1
                if not use_reliability:
                    use_dimens += "faithfulness score"
                else:
                    use_dimens += ", faithfulness score"
                faith_dimen = f"({use_dimen_num}) faithfulness score: Also ranging between 0 to 100, a faithfulness score is given to each step of the corresponding reasoning process, with higher scores being better. This score for each step reflects the level of faithfulness to the reasoning logic of that step when generating an answer to the Question. A higher score indicates a higher degree of faithfulness in employing the reasoning logic of the respective step."
            if use_factual:
                use_dimen_num += 1
                if not use_reliability and not use_fidelity:
                    use_dimens += "factuality score"
                else:
                    use_dimens += " and factuality score"
                fact_dimen = f"({use_dimen_num}) factuality score: This is a score given to each step of the reasoning process, ranging between 0 to 100. A higher score indicates a higher degree of factuality of the content expressed in each step. A higher score suggests that the reasoning basis of the respective step comes more from the documents or the understanding of the Question, meaning that the reasoning is based less on guesswork and more on the information provided."

            score_meaning = f'''
            {truth_dimen}
            {faith_dimen}
            {fact_dimen}
            '''

            use_dimen_est = f"When generating the optimized reasoning process, please note that you need to output both the optimized reasoning process and your estimated {use_dimens}. The specific format should refer to the reasoning processes in History, as detailed below:"
            score_meaning = score_meaning.strip()
            evo_logic = "综上所述，"
            evo_ff_structure = ""
            evo_tru_structure = ""
            out_tru_structure = ""
            out_ff_structure = ""

            if use_reliability:
                out_tru_structure = f"The beginning of the specific text content of each reasoning process should start with {PART_START} and end with {PART_END}, followed closely by your estimated overall truthfulness score, beginning with {TRUTH_SCORE_START} and ending with {TRUTH_SCORE_END}"
                if use_factual and use_fidelity:
                    evo_logic = f"In summary, when generating the optimized reasoning process, ensure that the overall process can achieve a higher truthfulness score. Under this premise, aim to improve the faithfulness score and factuality score for each step of the optimized reasoning process. Please note that the number and titles of steps you generate do not need to match those in the reasoning process in History."
                    evo_tru_structure = f"The beginning of the specific text content of each reasoning process is marked by {PART_START}, and the end is marked by {PART_END}, closely followed by the overall truthfulness score which starts with {TRUTH_SCORE_START} and ends with {TRUTH_SCORE_END}"
                    evo_ff_structure = f"Each reasoning process’s specific text content is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, providing a title and number for each step (starting with 1). After the text content of each step, a faithfulness score and a factuality score are given for that step, with faithfulness score starting with {FAITH_SCORE_START} and ending with {FAITH_SCORE_END}, and factuality score starting with {FACT_SCORE_START} and ending with {FACT_SCORE_END}"
                    out_ff_structure = f"The specific text content of each reasoning process should be divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, and each step text should provide a title and number. Steps are numbered starting with 1, with the beginning of each title marked with \"{TOP_START}\" and the end with \"{TOP_END}\". After the text of each step, you need to immediately provide your estimated faithfulness score and factuality score for that step, with faithfulness score starting with {FAITH_SCORE_START} and ending with {FAITH_SCORE_END}, and factuality score beginning with {FACT_SCORE_START} and ending with {FACT_SCORE_END}"
                elif use_factual and not use_fidelity:
                    evo_logic = f"In summary, when generating the optimized reasoning process, ensure that the overall process can achieve a higher truthfulness score. Under this premise, aim to improve the factuality score for each step of the optimized reasoning process. Please note that the number and titles of steps you generate do not need to match those in the reasoning process in History."
                    evo_tru_structure = f"The beginning of the specific text content of each reasoning process is marked by {PART_START}, and the end is marked by {PART_END}, closely followed by the overall truthfulness score which starts with {TRUTH_SCORE_START} and ends with {TRUTH_SCORE_END}"
                    evo_ff_structure = f"Each reasoning process’s specific text content is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, providing a title and number for each step (starting with 1). After the text content of each step, a factuality score is given for that step, with factuality score starting with {FACT_SCORE_START} and ending with {FACT_SCORE_END}"
                    out_ff_structure = f"The specific text content of each reasoning process should be divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, and each step text should provide a title and number. Steps are numbered starting with 1, with the beginning of each title marked with \"{TOP_START}\" and the end with \"{TOP_END}\". After the text of each step, you need to immediately provide your estimated factuality score for that step, with factuality score beginning with {FACT_SCORE_START} and ending with {FACT_SCORE_END}"
                elif not use_factual and use_fidelity:
                    evo_logic = f"In summary, when generating the optimized reasoning process, ensure that the overall process can achieve a higher truthfulness score. Under this premise, aim to improve the faithfulness score for each step of the optimized reasoning process. Please note that the number and titles of steps you generate do not need to match those in the reasoning process in History."
                    evo_tru_structure = f"The beginning of the specific text content of each reasoning process is marked by {PART_START}, and the end is marked by {PART_END}, closely followed by the overall truthfulness score which starts with {TRUTH_SCORE_START} and ends with {TRUTH_SCORE_END}"
                    evo_ff_structure = f"Each reasoning process’s specific text content is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, providing a title and number for each step (starting with 1). After the text content of each step, a faithfulness score is given for that step, with faithfulness score starting with {FAITH_SCORE_START} and ending with {FAITH_SCORE_END}"
                    out_ff_structure = f"The specific text content of each reasoning process should be divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, and each step text should provide a title and number. Steps are numbered starting with 1, with the beginning of each title marked with \"{TOP_START}\" and the end with \"{TOP_END}\". After the text of each step, you need to immediately provide your estimated faithfulness score for that step, with faithfulness score starting with {FAITH_SCORE_START} and ending with {FAITH_SCORE_END}"
                else:
                    evo_logic = f"In summary, when generating the optimized reasoning process, ensure that the overall process can achieve a higher truthfulness score. Please note that the number and titles of steps you generate do not need to match those in the reasoning process in History."
                    evo_tru_structure = f"The beginning of the specific text content of each reasoning process is marked by {PART_START}, and the end is marked by {PART_END}, closely followed by the overall truthfulness score which starts with {TRUTH_SCORE_START} and ends with {TRUTH_SCORE_END}"
                    evo_ff_structure = f"Each reasoning process’s specific text content is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, providing a title and number for each step (starting with 1)"
                    out_ff_structure = f"The specific text content of each reasoning process should be divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, and each step text should provide a title and number. Steps are numbered starting with 1, with the beginning of each title marked with \"{TOP_START}\" and the end with \"{TOP_END}\""

            else:
                out_tru_structure = f"The beginning of the specific text content of each reasoning process should start with {PART_START} and end with {PART_END}"
                evo_tru_structure = f"The beginning of the specific text content of each reasoning process is marked by {PART_START}, and the end is marked by {PART_END}"
                if use_factual and use_fidelity:
                    evo_ff_structure = f"Each reasoning process’s specific text content is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, providing a title and number for each step (starting with 1). After the text content of each step, a faithfulness score and a factuality score are given for that step, with faithfulness score starting with {FAITH_SCORE_START} and ending with {FAITH_SCORE_END}, and factuality score starting with {FACT_SCORE_START} and ending with {FACT_SCORE_END}"
                    evo_logic = "In summary, when generating the optimized reasoning process, please try to improve the faithfulness score and factuality score for each step of the optimized reasoning process. Please note that the number and titles of steps you generate do not need to match those in the reasoning process in History."
                    out_ff_structure = f"The specific text content of each reasoning process should be divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, and each step text should provide a title and number. Steps are numbered starting with 1, with the beginning of each title marked with \"{TOP_START}\" and the end with \"{TOP_END}\". After the text of each step, you need to immediately provide your estimated faithfulness score and factuality score for that step, with faithfulness score starting with {FAITH_SCORE_START} and ending with {FAITH_SCORE_END}, and factuality score beginning with {FACT_SCORE_START} and ending with {FACT_SCORE_END}"
                elif use_factual and not use_fidelity:
                    evo_logic = "In summary, when generating the optimized reasoning process, please try to improve the factuality score for each step of the optimized reasoning process. Please note that the number and titles of steps you generate do not need to match those in the reasoning process in History."
                    evo_ff_structure = f"Each reasoning process’s specific text content is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, providing a title and number for each step (starting with 1). After the text content of each step, a factuality score is given for that step, with factuality score starting with {FACT_SCORE_START} and ending with {FACT_SCORE_END}"
                    out_ff_structure = f"The specific text content of each reasoning process should be divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, and each step text should provide a title and number. Steps are numbered starting with 1, with the beginning of each title marked with \"{TOP_START}\" and the end with \"{TOP_END}\". After the text of each step, you need to immediately provide your estimated factuality score for that step, with factuality score beginning with {FACT_SCORE_START} and ending with {FACT_SCORE_END}"

                else:
                    evo_logic = "In summary, when generating the optimized reasoning process, please try to improve the faithfulness score for each step of the optimized reasoning process. Please note that the number and titles of steps you generate do not need to match those in the reasoning process in History."
                    evo_ff_structure = f"Each reasoning process’s specific text content is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, providing a title and number for each step (starting with 1). After the text content of each step, a faithfulness score is given for that step, with faithfulness score starting with {FAITH_SCORE_START} and ending with {FAITH_SCORE_END}"
                    out_ff_structure = f"The specific text content of each reasoning process should be divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}, and each step text should provide a title and number. Steps are numbered starting with 1, with the beginning of each title marked with \"{TOP_START}\" and the end with \"{TOP_END}\". After the text of each step, you need to immediately provide your estimated faithfulness score for that step, with faithfulness score starting with {FAITH_SCORE_START} and ending with {FAITH_SCORE_END}"

            evo_meta_prompt_no_weight_eng_super = f"""
            Hello, your task is to observe the given Documents, Question, Answer, and refer to the reasoning process and their quantitative scores (score) provided in History to generate what you consider an optimized reasoning process that could result in higher scores. Specifically, each generated reasoning process should represent the way of thinking (thought) or logical reasoning process (Reasoning Process) that you would need to follow to better approximate the given target Answer when you answer the Question based on the Documents.
            Please follow these requirements when generating the optimized reasoning process:
            1. Think step by step to generate the reasoning process. This process should reflect your thinking method or logic.
            2. Your outputted reasoning process should be an optimized result based on reasoning processes previously provided in History, without repeating them. Specifically, we have scored each reasoning process in History on {use_dimen_num} quantitative dimensions:
            {score_meaning}
            {evo_logic}
            3. Please use English to output the optimized reasoning process.
            4. Keep in mind that the purpose of generating the reasoning process is to guide the large model to produce an answer that approximates the target Answer, regardless of whether the given answer is correct or mistaken.
            5. Ensure not to directly answer the Question or offer any optimization suggestions for the target Answer.
            6. In this session's input History, the format for each previously generated reasoning process is as follows:
            (1) Each reasoning process instance includes specific text content and corresponding scoring, marked at the beginning by {HIST_START_TOKEN} and at the end by {HIST_END_TOKEN}
            (2) {evo_tru_structure}
            (3) {evo_ff_structure}
            
            {use_dimen_est}
            (1) Instances of optimized reasoning processes that are generated should include specific text content and corresponding scores, starting with {OPT_START_TOKEN} and ending with {OPT_END_TOKEN}
            (2) {out_tru_structure}
            (3) {out_ff_structure}
            (4) Please output the specific text content of the optimized reasoning process primarily in English
            (5) Please note that the number and titles of the steps you generate do not need to match those in the reasoning process in History
            """
            prompt_template = evo_meta_prompt_no_weight_eng_super

    else:
        if is_truth:
            use_dimen_num = 0
            score_meaning = ""
            use_dimens = ""
            truth_dimen = ""
            faith_dimen = ""
            fact_dimen = ""
            if use_reliability:
                use_dimen_num += 1
                use_dimens += "truthfulness score"
                truth_dimen = f"({use_dimen_num}) Truthfulness score: Ranging between 0 to 100, this score indicates the similarity between the answer generated through the reasoning process and the standard correct Answer when answering the Question based on the Documents. The higher the score, the more truthful the reasoning process is at guiding the answerer to a response similar to the standard correct Answer. The higher the score, the greater the textual meaning or semantic entailment of the result generated by the reasoning process aligns with the standard correct Answer."

            if use_fidelity:
                use_dimen_num += 1
                if not use_reliability:
                    use_dimens += "faithfulness score"
                else:
                    use_dimens += ", faithfulness score"
                faith_dimen = f"({use_dimen_num}) Faithfulness score: Also Ranging between 0 to 100, each step of the reasoning process is given a faithfulness score, with higher scores being better. Each step's score reflects the degree of faithfulness that step has in faithfully generating a response similar to the standard correct Answer. The higher this score, the greater the contribution of that step of the reasoning process in approximating the standard correct Answer."
            if use_factual:
                use_dimen_num += 1
                if not use_reliability and not use_fidelity:
                    use_dimens += "factuality score"
                else:
                    use_dimens += " and factuality score"
                fact_dimen = f"({use_dimen_num}) Factuality score: Also ranging between 0 to 100, this score is given to each step of the reasoning process. The higher the score, the greater the degree of factuality of the content expressed in that step of the reasoning process. A higher score suggests a stronger reliance on the provided Documents or Question, and less on one's own guessing."

            score_meaning = f'''
            {truth_dimen}
            {faith_dimen}
            {fact_dimen}
            '''

            use_dimen_est = f"When generating an optimized reasoning process, please note that you need to output both the optimized reasoning process and your estimated {use_dimens}. The specific format should refer to the reasoning process in the History, as follows:"
            score_meaning = score_meaning.strip()
            evo_logic = "综上所述，"
            evo_ff_structure = ""
            evo_tru_structure = ""
            out_tru_structure = ""
            out_ff_structure = ""
            if use_reliability:
                out_tru_structure = f"The beginning of each reasoning process’s specific text content starts with {PART_START} and ends with {PART_END}, followed immediately by your estimated overall truthfulness score, starting with {TRUTH_SCORE_START} and ending with {TRUTH_SCORE_END}"
                if use_factual and use_fidelity:
                    evo_logic = f"Taking the above into consideration, the overall reasoning process you create should strive for a higher truthfulness score. Under this premise, aim to maximize the faithfulness score and factuality score for each step of the optimized reasoning process. Remember that the number and titles of the steps you create don’t need to match those found in the reasoning process in History."
                    evo_tru_structure = f"The beginning of each reasoning process's specific text content is marked by {PART_START} and ends with {PART_END}, followed immediately by the overall truthfulness score marked at the beginning by {TRUTH_SCORE_START}, and at the end by {TRUTH_SCORE_END}"
                    evo_ff_structure = f"The specific text content of each reasoning process is composed of multiple steps, each step's specific text content starts with {STEP_START} and ends with {STEP_END}, each step text provides a title and a number (starting from 1). After each step's text content, the step's faithfulness score and factuality score will be provided; the beginning of the faithfulness score is marked by {FAITH_SCORE_START} and ends with {FAITH_SCORE_END}; the beginning of the factuality score is marked by {FACT_SCORE_START} and ends with {FACT_SCORE_END}"
                    out_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, each step's specific text content starts with {STEP_START} and ends with {STEP_END}, each step text needs to provide a title and number. The numbering of steps starts from 1, with each title marked at the beginning by \"{TOP_START}\" and at the end by \"{TOP_END}\". After each step's text content, you need to provide your estimated faithfulness score and factuality score; the beginning of the faithfulness score is marked by {FAITH_SCORE_START} and ends with {FAITH_SCORE_END}; the beginning of the factuality score is marked by {FACT_SCORE_START} and ends with {FACT_SCORE_END}"
                elif use_factual and not use_fidelity:
                    evo_logic = f"Taking the above into consideration, the overall reasoning process you create should strive for a higher truthfulness score. Under this premise, aim to maximize the factuality score for each step of the optimized reasoning process. Remember that the number and titles of the steps you create don’t need to match those found in the reasoning process in History."
                    evo_tru_structure = f"The beginning of each reasoning process's specific text content is marked by {PART_START} and ends with {PART_END}, followed immediately by the overall truthfulness score marked at the beginning by {TRUTH_SCORE_START}, and at the end by {TRUTH_SCORE_END}"
                    evo_ff_structure = f"The specific text content of each reasoning process is composed of multiple steps, each step's specific text content starts with {STEP_START} and ends with {STEP_END}, each step text provides a title and a number (starting from 1). After each step's text content, the step's factuality score will be provided; the beginning of the factuality score is marked by {FACT_SCORE_START} and ends with {FACT_SCORE_END}"
                    out_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, each step's specific text content starts with {STEP_START} and ends with {STEP_END}, each step text needs to provide a title and number. The numbering of steps starts from 1, with each title marked at the beginning by \"{TOP_START}\" and at the end by \"{TOP_END}\". After each step's text content, you need to provide your estimated factuality score; the beginning of the factuality score is marked by {FACT_SCORE_START} and ends with {FACT_SCORE_END}"
                elif not use_factual and use_fidelity:
                    evo_logic = f"Taking the above into consideration, the overall reasoning process you create should strive for a higher truthfulness score. Under this premise, aim to maximize the faithfulness score for each step of the optimized reasoning process. Remember that the number and titles of the steps you create don’t need to match those found in the reasoning process in History."
                    evo_tru_structure = f"The beginning of each reasoning process's specific text content is marked by {PART_START} and ends with {PART_END}, followed immediately by the overall truthfulness score marked at the beginning by {TRUTH_SCORE_START}, and at the end by {TRUTH_SCORE_END}"
                    evo_ff_structure = f"The specific text content of each reasoning process is composed of multiple steps, each step's specific text content starts with {STEP_START} and ends with {STEP_END}, each step text provides a title and a number (starting from 1). After each step's text content, the step's faithfulness score will be provided; the beginning of the faithfulness score is marked by {FAITH_SCORE_START} and ends with {FAITH_SCORE_END}"
                    out_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, each step's specific text content starts with {STEP_START} and ends with {STEP_END}, each step text needs to provide a title and number. The numbering of steps starts from 1, with each title marked at the beginning by \"{TOP_START}\" and at the end by \"{TOP_END}\". After each step's text content, you need to provide your estimated faithfulness score; the beginning of the faithfulness score is marked by {FAITH_SCORE_START} and ends with {FAITH_SCORE_END}"
                else:
                    evo_logic = f"Taking the above into consideration, the overall reasoning process you create should strive for a higher truthfulness score. Remember that the number and titles of the steps you create don’t need to match those found in the reasoning process in History."
                    evo_tru_structure = f"The beginning of each reasoning process's specific text content is marked by {PART_START} and ends with {PART_END}, followed immediately by the overall truthfulness score marked at the beginning by {TRUTH_SCORE_START}, and at the end by {TRUTH_SCORE_END}"
                    evo_ff_structure = f"The specific text content of each reasoning process is composed of multiple steps, each step's specific text content starts with {STEP_START} and ends with {STEP_END}, each step text provides a title and a number (starting from 1)"
                    out_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, each step's specific text content starts with {STEP_START} and ends with {STEP_END}, each step text needs to provide a title and number. The numbering of steps starts from 1, with each title marked at the beginning by \"{TOP_START}\" and at the end by \"{TOP_END}\""

            else:
                out_tru_structure = f"The beginning of each reasoning process’s specific text content starts with {PART_START} and ends with {PART_END}"
                evo_tru_structure = f"The beginning of each reasoning process's specific text content starts with {PART_START} and ends with {PART_END}"
                if use_factual and use_fidelity:
                    evo_ff_structure = f"The specific text content of each reasoning process is composed of multiple steps, each step's specific text content starts with {STEP_START} and ends with {STEP_END}, each step text provides a title and a number (starting from 1). After each step's text content, the step's faithfulness score and factuality score will be provided; the beginning of the faithfulness score is marked by {FAITH_SCORE_START} and ends with {FAITH_SCORE_END}; the beginning of the factuality score is marked by {FACT_SCORE_START} and ends with {FACT_SCORE_END}"
                    evo_logic = f"Taking the above into consideration, the reasoning process you create should maximize the faithfulness score and factuality score for each step of the optimized reasoning process. Remember that the number and titles of the steps you create don’t need to match those found in the reasoning process in History."
                    out_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, each step's specific text content starts with {STEP_START} and ends with {STEP_END}, each step text needs to provide a title and number. The numbering of steps starts from 1, with each title marked at the beginning by \"{TOP_START}\" and at the end by \"{TOP_END}\". After each step's text content, you need to provide your estimated faithfulness score and factuality score; the beginning of the faithfulness score is marked by {FAITH_SCORE_START} and ends with {FAITH_SCORE_END}; the beginning of the factuality score is marked by {FACT_SCORE_START} and ends with {FACT_SCORE_END}"
                elif use_factual and not use_fidelity:
                    evo_logic = f"Taking the above into consideration, the reasoning process you create should maximize the factuality score for each step of the optimized reasoning process. Remember that the number and titles of the steps you create don’t need to match those found in the reasoning process in History."
                    evo_ff_structure = f"The specific text content of each reasoning process is composed of multiple steps, each step's specific text content starts with {STEP_START} and ends with {STEP_END}, each step text provides a title and a number (starting from 1). After each step's text content, the step's factuality score will be provided; the beginning of the factuality score is marked by {FACT_SCORE_START} and ends with {FACT_SCORE_END}"
                    out_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, each step's specific text content starts with {STEP_START} and ends with {STEP_END}, each step text needs to provide a title and number. The numbering of steps starts from 1, with each title marked at the beginning by \"{TOP_START}\" and at the end by \"{TOP_END}\". After each step's text content, you need to provide your estimated factuality score; the beginning of the factuality score is marked by {FACT_SCORE_START} and ends with {FACT_SCORE_END}"

                else:
                    evo_logic = f"Taking the above into consideration, the reasoning process you create should maximize the faithfulness score for each step of the optimized reasoning process. Remember that the number and titles of the steps you create don’t need to match those found in the reasoning process in History."
                    evo_ff_structure = f"The specific text content of each reasoning process is composed of multiple steps, each step's specific text content starts with {STEP_START} and ends with {STEP_END}, each step text provides a title and a number (starting from 1). After each step's text content, the step's faithfulness score will be provided; the beginning of the faithfulness score is marked by {FAITH_SCORE_START} and ends with {FAITH_SCORE_END}"
                    out_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, each step's specific text content starts with {STEP_START} and ends with {STEP_END}, each step text needs to provide a title and number. The numbering of steps starts from 1, with each title marked at the beginning by \"{TOP_START}\" and at the end by \"{TOP_END}\". After each step's text content, you need to provide your estimated faithfulness score; the beginning of the faithfulness score is marked by {FAITH_SCORE_START} and ends with {FAITH_SCORE_END}"

            evo_meta_prompt_eng_unsuper = f"""
            Hello, your task is to review the given Documents and Question, and to reference the reasoning processes within the provided History as well as their quantitative scores (score), in order to generate an optimized reasoning process that you believe could result in higher scores. Specifically, each reasoning process you create should represent your conception of the large model's thought process or logical reasoning (Reasoning Process) it would use to answer Questions based on the given Documents. 
            Keep in mind the following requirements when generating the optimized reasoning process:
            1. Generate the reasoning process step by step. This process should reflect your view of the LLMs's way of thinking or logic.
            2. The reasoning process you provide should be an optimized version of those previously generated in History, without duplicating them. Specifically, each reasoning process in History has been assessed on {use_dimen_num} dimensions:
            {score_meaning}
            {evo_logic}
            3. Please output the optimized reasoning process in English.
            4. Note that the purpose of creating a reasoning process is to explain the thinking or method used by the large model in generating answers. This explanation should be capable of guiding the large model to refer to the Documents correctly when answering the Question, in order to provide a correct Answer.
            5. Make sure not to directly answer the Question in the reasoning process you generate.
            6. In the input history provided, each previously generated reasoning process follows the format below: 
            (1) Each reasoning process instance includes specific text content and its corresponding score, marked by {HIST_START_TOKEN} at the beginning and {HIST_END_TOKEN} at the end
            (2) {evo_tru_structure}
            (3) {evo_ff_structure}

            {use_dimen_est}
            (1) The optimized reasoning process instance includes specific text content and corresponding scores marked at the beginning by {OPT_START_TOKEN} and at the end by {OPT_END_TOKEN}
            (2) {out_tru_structure}
            (3) {out_ff_structure}
            (4) Please produce the specific text content of the optimized reasoning process primarily in English
            (5) Please note, the number and titles of steps you generate do not need to match those in the history reasoning process
            """

            prompt_template = evo_meta_prompt_eng_unsuper

        else:
            use_dimen_num = 0
            score_meaning = ""
            use_dimens = ""
            truth_dimen = ""
            faith_dimen = ""
            fact_dimen = ""
            if use_reliability:
                use_dimen_num += 1
                use_dimens += "truthfulness score"
                truth_dimen = f"({use_dimen_num}) Truthfulness score: Ranging from 0 to 100, this score represents the similarity between the answer you generate using the corresponding reasoning process to answer the Question based on the provided Documents, and the standard correct Answer. The higher the score, the better. A higher score means that the corresponding reasoning process can guide you more truthfully to generate an answer similar to the standard correct Answer. Specifically, the higher the score, the more the text meaning or semantic entailment of the result generated by the corresponding reasoning process corresponds to the standard correct Answer, and the closer the semantics and structure of the generated result to the standard correct Answer."

            if use_fidelity:
                use_dimen_num += 1
                if not use_reliability:
                    use_dimens += "faithfulness score"
                else:
                    use_dimens += ", faithfulness score"
                faith_dimen = f"({use_dimen_num}) Faithfulness score: Ranging from 0 to 100, a faithfulness score will be given for each step of the corresponding reasoning process; the higher, the better. The score for each step represents your faithfulness when generating an answer to the Question relative to the corresponding step of the reasoning process. Specifically, the higher the score, the more faithfully you adopt the reasoning logic of the corresponding step when answering the question—that is, the step more faithfully reflects your way of thinking."
            if use_factual:
                use_dimen_num += 1
                if not use_reliability and not use_fidelity:
                    use_dimens += "factuality score"
                else:
                    use_dimens += " and factuality score"
                fact_dimen = f"({use_dimen_num}) Factuality score: Ranging from 0 to 100, a factuality score will be given for each step of the corresponding reasoning process; the higher, the better. The score for each step represents the factual degree of the content expressed in the corresponding step of the reasoning process. Specifically, the higher the score, the more the reasoning basis of the corresponding step of the reasoning process is derived from referencing or understanding the provided Documents or Question. Correspondingly, the higher the score, the lower the probability that the reasoning is based on your own guesswork and the lower the probability that the reasoning deviates from the provided Document and Question."

            score_meaning = f'''
            {truth_dimen}
            {faith_dimen}
            {fact_dimen}
            '''

            use_dimen_est = f"When generating the optimized reasoning process, please note that you need to also output the optimized reasoning process along with your estimated {use_dimens}. The specific format should refer to the reasoning processes in the History, outlined as follows:"
            score_meaning = score_meaning.strip()
            evo_logic = "综上所述，"
            evo_ff_structure = ""
            evo_tru_structure = ""
            out_tru_structure = ""
            out_ff_structure = ""
            if use_reliability:
                out_tru_structure = f"The beginning of each reasoning process’s specific text content is marked by {PART_START} and ends with {PART_END}, followed immediately by your estimated overall truthfulness score, starting with {TRUTH_SCORE_START} and ending with {TRUTH_SCORE_END}"
                if use_factual and use_fidelity:
                    evo_logic = f"In summary, when generating an optimized reasoning process, please ensure that you can obtain a higher truthfulness score for the overall reasoning process. Under this premise, please try to improve the faithfulness score and factuality score for each step of the optimized reasoning process you generate. Also, please note that the number of steps you generate and their titles do not need to match those in the history reasoning process."
                    evo_tru_structure = f"The beginning of each reasoning process's specific text content is marked by {PART_START} and ends with {PART_END}, followed immediately by the overall truthfulness score marked at the start with {TRUTH_SCORE_START} and at the end with {TRUTH_SCORE_END}"
                    evo_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}. Each step text includes a title and a number (starting from 1). After the text content of each step, the step's faithfulness score and factuality score are given, with the beginning of the faithfulness score marked by {FAITH_SCORE_START} and the end by {FAITH_SCORE_END}; the factuality score starts with {FACT_SCORE_START} and ends with {FACT_SCORE_END}"
                    out_ff_structure = f"Each reasoning process’s specific text content is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}. Each step text needs to provide a title and a number. Step numbering starts from 1, with each title starting with \"{TOP_START}\" and ending with \"{TOP_END}\". After the text content of each step, you need to provide your estimated faithfulness score and factuality score; the faithfulness score starts with {FAITH_SCORE_START} and ends with {FAITH_SCORE_END}; the factuality score starts with {FACT_SCORE_START} and ends with {FACT_SCORE_END}"
                elif use_factual and not use_fidelity:
                    evo_logic = f"In summary, when generating an optimized reasoning process, please ensure that you can obtain a higher truthfulness score for the overall reasoning process. Under this premise, please try to improve the factuality score for each step of the optimized reasoning process you generate. Also, please note that the number of steps you generate and their titles do not need to match those in the history reasoning process."
                    evo_tru_structure = f"The beginning of each reasoning process's specific text content is marked by {PART_START} and ends with {PART_END}, followed immediately by the overall truthfulness score marked at the start with {TRUTH_SCORE_START} and at the end with {TRUTH_SCORE_END}"
                    evo_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}. Each step text includes a title and a number (starting from 1). After the text content of each step, the step's factuality score is given, the factuality score starts with {FACT_SCORE_START} and ends with {FACT_SCORE_END}"
                    out_ff_structure = f"Each reasoning process’s specific text content is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}. Each step text needs to provide a title and a number. Step numbering starts from 1, with each title starting with \"{TOP_START}\" and ending with \"{TOP_END}\". After the text content of each step, you need to provide your estimated factuality score; the factuality score starts with {FACT_SCORE_START} and ends with {FACT_SCORE_END}"
                elif not use_factual and use_fidelity:
                    evo_logic = f"In summary, when generating an optimized reasoning process, please ensure that you can obtain a higher truthfulness score for the overall reasoning process. Under this premise, please try to improve the faithfulness score for each step of the optimized reasoning process you generate. Also, please note that the number of steps you generate and their titles do not need to match those in the history reasoning process."
                    evo_tru_structure = f"The beginning of each reasoning process's specific text content is marked by {PART_START} and ends with {PART_END}, followed immediately by the overall truthfulness score marked at the start with {TRUTH_SCORE_START} and at the end with {TRUTH_SCORE_END}"
                    evo_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}. Each step text includes a title and a number (starting from 1). After the text content of each step, the step's faithfulness score is given, with the beginning of the faithfulness score marked by {FAITH_SCORE_START} and the end by {FAITH_SCORE_END}"
                    out_ff_structure = f"Each reasoning process’s specific text content is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}. Each step text needs to provide a title and a number. Step numbering starts from 1, with each title starting with \"{TOP_START}\" and ending with \"{TOP_END}\". After the text content of each step, you need to provide your estimated faithfulness score; the faithfulness score starts with {FAITH_SCORE_START} and ends with {FAITH_SCORE_END}"
                else:
                    evo_logic = f"In summary, when generating an optimized reasoning process, please ensure that you can obtain a higher truthfulness score for the overall reasoning process. Also, please note that the number of steps you generate and their titles do not need to match those in the history reasoning process."
                    evo_tru_structure = f"The beginning of each reasoning process's specific text content is marked by {PART_START} and ends with {PART_END}, followed immediately by the overall truthfulness score marked at the start with {TRUTH_SCORE_START} and at the end with {TRUTH_SCORE_END}"
                    evo_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}. Each step text includes a title and a number (starting from 1)"
                    out_ff_structure = f"Each reasoning process’s specific text content is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}. Each step text needs to provide a title and a number. Step numbering starts from 1, with each title starting with \"{TOP_START}\" and ending with \"{TOP_END}\""

            else:
                out_tru_structure = f"The beginning of each reasoning process’s specific text content is marked by {PART_START} and ends with {PART_END}"
                evo_tru_structure = f"The beginning of each reasoning process's specific text content is marked by {PART_START} and ends with {PART_END}"
                if use_factual and use_fidelity:
                    evo_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}. Each step text includes a title and a number (starting from 1). After the text content of each step, the step's faithfulness score and factuality score are given, with the beginning of the faithfulness score marked by {FAITH_SCORE_START} and the end by {FAITH_SCORE_END}; the factuality score starts with {FACT_SCORE_START} and ends with {FACT_SCORE_END}"
                    evo_logic = "In summary, when generating an optimized reasoning process, please try to improve the faithfulness score and factuality score for each step of the optimized reasoning process you generate. Also, please note that the number of steps you generate and their titles do not need to match those in the history reasoning process."
                    out_ff_structure = f"Each reasoning process’s specific text content is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}. Each step text needs to provide a title and a number. Step numbering starts from 1, with each title starting with \"{TOP_START}\" and ending with \"{TOP_END}\". After the text content of each step, you need to provide your estimated faithfulness score and factuality score; the faithfulness score starts with {FAITH_SCORE_START} and ends with {FAITH_SCORE_END}; the factuality score starts with {FACT_SCORE_START} and ends with {FACT_SCORE_END}"
                elif use_factual and not use_fidelity:
                    evo_logic = "In summary, when generating an optimized reasoning process, please try to improve the factuality score for each step of the optimized reasoning process you generate. Also, please note that the number of steps you generate and their titles do not need to match those in the history reasoning process."
                    evo_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}. Each step text includes a title and a number (starting from 1). After the text content of each step, the step's factuality score is given, the factuality score starts with {FACT_SCORE_START} and ends with {FACT_SCORE_END}"
                    out_ff_structure = f"Each reasoning process’s specific text content is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}. Each step text needs to provide a title and a number. Step numbering starts from 1, with each title starting with \"{TOP_START}\" and ending with \"{TOP_END}\". After the text content of each step, you need to provide your estimated factuality score; the factuality score starts with {FACT_SCORE_START} and ends with {FACT_SCORE_END}"

                else:
                    evo_logic = "In summary, when generating an optimized reasoning process, please try to improve the faithfulness score for each step of the optimized reasoning process you generate. Also, please note that the number of steps you generate and their titles do not need to match those in the history reasoning process."
                    evo_ff_structure = f"The specific text content of each reasoning process is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}. Each step text includes a title and a number (starting from 1). After the text content of each step, the step's faithfulness score is given, with the beginning of the faithfulness score marked by {FAITH_SCORE_START} and the end by {FAITH_SCORE_END}"
                    out_ff_structure = f"Each reasoning process’s specific text content is divided into multiple steps, with each step's specific text content starting with {STEP_START} and ending with {STEP_END}. Each step text needs to provide a title and a number. Step numbering starts from 1, with each title starting with \"{TOP_START}\" and ending with \"{TOP_END}\". After the text content of each step, you need to provide your estimated faithfulness score; the faithfulness score starts with {FAITH_SCORE_START} and ends with {FAITH_SCORE_END}"

            evo_meta_prompt_no_weight_eng_unsuper = f"""
            Hello, your task is to observe the given Documents and Question, and refer to the reasoning processes and their quantitative scores (scores) provided in the History, to generate an optimized reasoning process that you believe can obtain a higher score. Specifically, each reasoning process you generate represents what you consider to be the large model's thought pattern or logical reasoning process used to answer the Question according to the Documents.
            Please note that when generating the optimized reasoning process, you need to follow the following requirements:
            1. Please generate the reasoning process by thinking step-by-step. This process should reflect your thought process or logical reasoning.
            2. Please note that the reasoning process you output should be an optimized result of the reasoning processes previously generated in the History and should not repeat the reasoning processes in the History. Specifically, we have performed a quantitative scoring in {use_dimen_num} dimensions for each reasoning process provided in the History:
            {score_meaning}
            {evo_logic}
            3. Please output the optimized reasoning process in English.
            4. Please note that the purpose of generating the reasoning process is to explain the thought pattern or mode of thinking used by the large model when generating answers. The thought or mode of thinking explained through the reasoning process can guide the large model to refer to the Documents correctly to answer the Question, to obtain the correct Answer as accurately as possible
            5. Please ensure not to directly answer the Question in the reasoning process you generate
            6. In the history provided for this input, the format for each of the previously generated reasoning processes is as follows: 
            (1) Each reasoning process instance consists of the specific text content and corresponding scores, starting with {HIST_START_TOKEN} and ending with {HIST_END_TOKEN}
            (2) {evo_tru_structure}
            (3) {evo_ff_structure}

            {use_dimen_est}
            (1) The instance of the generated optimized reasoning process includes the specific text content and corresponding scores, starting with {OPT_START_TOKEN} and ending with {OPT_END_TOKEN}
            (2) {out_tru_structure}
            (3) {out_ff_structure}
            (4) Please generate the specific text content of the optimized reasoning process primarily in English
            (5) Please note that the number and titles of steps you generate do not need to match those in the history reasoning process
            """
            prompt_template = evo_meta_prompt_no_weight_eng_unsuper

    if "gpt" in model_type:
        if is_super:
            user_prompt_instruction += f"""\nPlease, according to the format requirements, output what you believe to be the optimized reasoning process that can achieve a higher score and can better approximate the target Answer provided through guiding the large model based on the Documents to answer the Question, as well as your estimated {use_dimens} of the optimized reasoning process:"""
        else:
            user_prompt_instruction += f"""\nPlease, according to the format requirements, output what you believe to be the optimized reasoning process that can achieve a higher score and can guide the large model to correctly answer (i.e., approximate the correct standard Answer) the Question based on the Documents, as well as your estimated {use_dimens} of the optimized reasoning process:"""

        pro_template = prompt_template
        # if is_super:
        #     if is_truth:
        #         pro_template = EVO_META_PROMPT_ENG_SUPER["qwen"]
        #     else:
        #         pro_template = EVO_META_PROMPT_NO_WEIGHT_ENG_SUPER["qwen"]
        # else:
        #     if is_truth:
        #         pro_template = EVO_META_PROMPT_ENG_UNSUPER["qwen"]
        #     else:
        #         pro_template = EVO_META_PROMPT_NO_WEIGHT_ENG_UNSUPER["qwen"]

        pro_template = pro_template + "\n"

        # gen_prompt = re.sub("\[PROMPT\]", pro_template, template)
        # gen_prompt = re.sub("\[INSTRUCTION\]", user_prompt_instruction, gen_prompt)

        gen_prompt = pro_template + user_prompt_instruction
    else:
        if is_super:
            user_prompt_instruction += f"\nPlease, according to the format requirements, output the optimized reasoning process you believe can achieve a higher score and better approximate the target Answer provided by guiding you to answer the Question based on the Documents, along with your estimated {use_dimens} of the optimized reasoning process:"
        else:
            user_prompt_instruction += f"\nPlease, according to the format requirements, output the optimized reasoning process you believe can achieve a higher score and can better guide you to provide the correct answer (i.e., approximate the standard correct Answer) based on the Documents, as well as your estimated {use_dimens} of the optimized reasoning process:"

        # if is_super:
        #     if is_truth:
        #         pro_template = EVO_META_PROMPT_ENG_SUPER["qwen"]
        #     else:
        #         pro_template = EVO_META_PROMPT_NO_WEIGHT_ENG_SUPER["qwen"]
        # else:
        #     if is_truth:
        #         pro_template = EVO_META_PROMPT_ENG_UNSUPER["qwen"]
        #     else:
        #         pro_template = EVO_META_PROMPT_NO_WEIGHT_ENG_UNSUPER["qwen"]

        pro_template = prompt_template

        gen_prompt = [{"role": "system",
                       "content": 'Hello, you are the intelligent robot of the smart Q&A project.' + pro_template},
                      {"role": "user", "content": user_prompt_instruction}]
    return gen_prompt