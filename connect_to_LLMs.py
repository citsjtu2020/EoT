import requests
import time
# connect with GPT. Beacaus of anonymity requirement, we do not provide the specific url
GPT_BASE_URL = ""
def model_connect(prompt, base_url=GPT_BASE_URL,
                  model_name="gpt-4-turbo-128k",
                  repeat_time=2,
                  headers={"X-AK": "6256c3d56c519ad98858d6c274e88a70"}):



    url = base_url
    headers = headers

    params = {
        "model": model_name,
        "prompt": prompt
    }
    errs = None
    for jj in range(repeat_time):
        try:
            response = requests.post(url, headers=headers, json=params)
            results = response.json()['data']['content']
            errs = None
        except Exception as ee0:
            print(ee0)
            errs = ee0

        if not errs:
            break
        else:
            time.sleep(8.50)
    if errs:
        print(errs)
        results = ""

    return results

# connect with Qwen2-72B. Beacaus of anonymity requirement, we do not provide the specific url
QWEN_BASE_URL = ""
def call_LLM_response_for_prod(prompts, base_url=QWEN_BASE_URL,
                               debug=False, repeat_time=2):


    if debug:
        print("prompts：" + "\n")
        print(prompts)

    # 请求模型
    out_results = None
    for rt in range(repeat_time):
        try:
            response = requests.post(base_url, json=prompts)
            results = response.json()


            if response.status_code == 200:
                out_results = results["context"]
                if debug:
                    print('\n 输出结果为：')
                    print(results)
            else:
                time.sleep(1.25)
                if debug:
                    print(response)

                out_results = None
        except Exception as ee:
            print(ee)

        if out_results:
            break

    return out_results

