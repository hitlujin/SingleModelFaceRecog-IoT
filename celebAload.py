from prompttemplate import celeb_prompt,attr_index
import random
from prompt import celeba_forward
celeb_attr_path = "F:\DLdataset\CelebA\Anno\list_attr_celeba.txt"
celeb_path = "F:\DLdataset\CelebA\Img\img_celeba\img_celeba\\"
def read_attr_txt(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    return lines


# read file
# line 0, to number
# random sample n lines
# return image path array and attr array
# for each attr element, call a prompt template


# 以下两个函数抽取样本，使用手工拼接
def sample_n_samples(lines, n,level):
    total = int(lines[0])
    keys = random.sample(range(1, total), n)
    img_paths = []
    prompts = []
    lines = lines[2:]
    for key in keys:
        img_paths.append(celeb_path + lines[key].split()[0])
        attr = lines[key].split()[1:]
        prompt = celeb_prompt(attr,level)
        prompts.append(prompt)
    return img_paths, prompts

def load_samples(n,level):
    lines = read_attr_txt(celeb_attr_path)
    img_paths, prompts = sample_n_samples(lines, n,level)
    return img_paths, prompts


# 以下函数使用llm创建celeba的caption，并保存到一个文件
def cancat_attr(attr,level):
    # read attr from celeb_attr_path line 1 (not zero)
    celeb_attr = read_attr_txt(celeb_attr_path)[1].split()
    attr_str = ""
    seleted_index = attr_index(level)
    for i in seleted_index:
        if attr[i] == '1':
            attr_str += celeb_attr[i] + ","
    return attr_str[:-1]

# a new load function use cancat_attr as prompt
def samples(lines,n,level):
    total = int(lines[0])
    keys = random.sample(range(1, total), n)
    img_paths = []
    prompts = []
    lines = lines[2:]
    for key in keys:
        img_paths.append(celeb_path + lines[key].split()[0])
        attr = lines[key].split()[1:]
        prompt = cancat_attr(attr,level)
        prompts.append(prompt)
    return img_paths, prompts

def load_llm_samples(n):
    lines = read_attr_txt(celeb_attr_path)
    img_paths, prompts = samples(lines, n)
    llm_samples = [celeba_forward(prompt) for prompt in prompts]
    return img_paths, llm_samples

def save_samples_to_txt(n, output_file):
    # 调用load_llm_samples函数来获取数据
    image_paths, llm_samples = load_llm_samples(n)

    # 打开一个文件用于写入数据
    with open(output_file, 'w') as file:
        for img_path, sample in zip(image_paths, llm_samples):
            # 将每个图像路径和对应的LLM样本写入文件
            file.write(img_path + '|' + sample + '\n')


def load_and_save_llm_samples(n, level, output_file):
    lines = read_attr_txt(celeb_attr_path)
    img_paths, prompts = samples(lines, n, level)

    count = 0
    with open(output_file, 'w') as file:
        for img_path, prompt in zip(img_paths, prompts):
            try:
                if prompt == "":
                    llm_sample = "A person"
                else:
                    llm_sample = celeba_forward(prompt)
                # 将每个图像路径和对应的LLM样本写入文件
                print(count, img_path + ':' + llm_sample)
                file.write(img_path + ':' + llm_sample + '\n')
                count += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")


# for i in range(28,31):
#     print(i)
#     load_and_save_llm_samples(130,i,f"llmcaption/celebA_samples_{i}.txt")

def load_llm_samples_from_file(n,level):
    with open(f"llmcaption/celebA_samples_{level}.txt", 'r') as file:
        lines = file.readlines()
    img_paths = []
    llm_samples = []
    for line in lines[:n]:
        # line example: F:\DLdataset\CelebA\Img\img_celeba\img_celeba\154751.jpg:a young person with narrow eyes, no beard, wavy hair, and wearing lipstick
        items = line.split(':')
        image_path = items[0] + ":" + items[1]
        llm_sample = items[2]
        img_paths.append(image_path)
        llm_samples.append(llm_sample)
    return img_paths, llm_samples


