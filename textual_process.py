import re

# preprocess data

def format_token_for_re(input_tokens):
    if "[" in input_tokens:
        input_tokens = input_tokens.replace("[","\[")
    if "]" in input_tokens:
        input_tokens = input_tokens.replace("]","\]")
    if "?" in input_tokens:
        input_tokens = input_tokens.replace("?","\?")
    if "*" in input_tokens:
        input_tokens = input_tokens.replace("*","\*")
    return input_tokens


def un_labeled_index(input_tokens_list):
    out_token_list = []
    for j in range(len(input_tokens_list)):
        tmp_input_tokens = input_tokens_list[j].strip()
        if tmp_input_tokens.startswith(f"{j + 1}. **"):
            tmp_input_tokens = tmp_input_tokens.replace(f"{j + 1}. **", "**", 1)
        out_token_list.append(tmp_input_tokens)

    return out_token_list


def split_sentences(input_sentences, split_token="\n", remove_last_token=False):
    # .strip()
    tmp_sentence_list = input_sentences.split(split_token)
    # print(tmp_sentence_list)
    use_sentence_list = []
    for tut_id in range(len(tmp_sentence_list)):
        tut = tmp_sentence_list[tut_id]
        if tut.strip() and len(tut.strip()) > 1:
            use_sentence_list.append(f"{tut.strip()}{split_token}")
    out_use_sentence_list = []
    for idx in range(len(use_sentence_list)):
        if idx < (len(use_sentence_list) - 1):
            out_use_sentence_list.append(use_sentence_list[idx])
        else:
            if remove_last_token:
                if (not tmp_sentence_list[-1]) or (input_sentences.endswith(split_token)) or (
                input_sentences.strip().endswith(split_token)):
                    out_use_sentence_list.append(use_sentence_list[idx])
                else:
                    out_use_sentence_list.append(use_sentence_list[idx].replace(split_token, ""))
            else:
                out_use_sentence_list.append(use_sentence_list[idx])

    return out_use_sentence_list


def split_step_thinking(input_thinks, depth=0, top_start="**",
                        top_end="**", first_token="\n",
                        second_token="。"):
    use_top_start = format_token_for_re(top_start)
    use_top_end = format_token_for_re(top_end)
    re_pattern = f"([\d]*?\.[ \t]*?{use_top_start}[\S\s]*?{use_top_end}[：:\t \n]?)"
    total_top_items = re.findall(re_pattern, input_thinks)
    if len(total_top_items) > 0:
        top_item = total_top_items[0]
    else:
        # [：:\t \n]?
        re_re_pattern = f"({use_top_start}[\S\s]*?{use_top_end}[：:\t \n]?)"
        total_top_items = re.findall(re_re_pattern, input_thinks)
        if len(total_top_items) > 0:
            top_item = total_top_items[0]
        else:
            re_re_pattern = f"({use_top_start}[\S\s]*?{use_top_end})"
            total_top_items = re.findall(re_re_pattern, input_thinks)
            if len(total_top_items) > 0:
                top_item = total_top_items[0]
            else:
                top_item = ""

    # if not top_item:
    #     return {}

    use_depth = int(depth)

    if depth > 2:
        depth = 2
    if depth < 0:
        depth = 0

    use_input_thinks = input_thinks.strip()
    if top_item:
        if use_input_thinks.strip().startswith(top_item) and top_item:
            use_input_thinks = use_input_thinks.replace(top_item, "")
            top_tokens = top_item
        elif top_item in use_input_thinks and top_item:
            use_input_thinks = use_input_thinks.replace(top_item, "")
            top_tokens = top_item
        else:
            top_tokens = ""
    else:
        top_tokens = ""


    re_top_pattern = f"[\d]*?\.[ \t]*?{use_top_start}([\S\s]*?){use_top_end}[：:\t \n]?"
    top_item_out = ""

    candidate_top_items = re.findall(re_top_pattern, top_item)
    for j in range(len(candidate_top_items)):
        if candidate_top_items[j]:
            top_item_out = candidate_top_items[j]
            break

    if not top_item_out:
        re_top_pattern = f"{use_top_start}([\S\s]*?){use_top_end}"
        candidate_top_items = re.findall(re_top_pattern, top_item)
        for j in range(len(candidate_top_items)):
            if candidate_top_items[j]:
                top_item_out = candidate_top_items[j]
                break


    if depth <= 0:
        use_input_think_list = [use_input_thinks]
    elif depth < 2:
        use_input_think_list = split_sentences(use_input_thinks, split_token=first_token)
    else:
        first_input_think_list = split_sentences(input_sentences=use_input_thinks, split_token=first_token)
        # print(first_input_think_list)
        use_input_think_list = []
        for fit in first_input_think_list:
            # print(fit)
            tmp_sens = split_sentences(input_sentences=fit, split_token=second_token)
            # print(tmp_sens)
            if len(tmp_sens) > 0:
                tmp_last_sen = tmp_sens.pop(-1)
                tmp_last_sen += first_token
                tmp_sens.append(tmp_last_sen)
                use_input_think_list.append(tmp_sens)

    return {"top": top_item_out, "content": use_input_think_list[:]}


def extract_format_comp(prop, start_token="\[PART\]",
                        end_token="\[/PART\]",
                        raw_start_token="[PART]",
                        raw_end_token="[/PART]"):
    '''
    对返回的文本进行切分后，从切分文本中逐条提取生成的reasoning的内容（使用正则表达式实现）
    '''
    # 目标PROMPT格式的正则模板
    # re_pattern = f"[\S\s]*?{start_token}(.*?){end_token}[\S\s]*?"
    # re_pattern = f"{start_token}(.*?){end_token}"
    # [\S\s]*?
    re_pattern = f"{start_token}([\S\s]*?){end_token}"
    # print(relink)
    info = prop.strip()
    extract_infos = re.findall(re_pattern, info, re.DOTALL)
    output_extract_infos = []
    for einfo in extract_infos:
        if raw_start_token in einfo:
            einfo_splits = einfo.split(raw_start_token)
            for es in einfo_splits:
                if es:
                    if raw_end_token in es:
                        es_out = es.replace(raw_end_token, "")
                    else:
                        es_out = es
                    if raw_start_token in es_out:
                        es_out = es_out.replace(raw_start_token, "")
                    if es_out:
                        output_extract_infos.append(es_out)
        else:
            if raw_end_token in einfo:
                einfo = einfo.replace(raw_end_token, "")
            if einfo:
                output_extract_infos.append(einfo)

    return output_extract_infos[:]






