import numpy as np
from datasets import load_dataset


def load_conala():
    
    dataset = load_dataset("neulab/conala")

    data, output_sizes = [], []
    for idx in range(dataset['test'].num_rows):
        data.append({
            'input': dataset['test'][idx]['rewritten_intent'],
            'output': dataset['test'][idx]['snippet']
        })
        output_sizes.append(len(dataset['test'][idx]['snippet'].split()))

    print(f"[INFO] Average number of tokens: {np.mean(output_sizes)}")
    print(f"[INFO] P90 number of tokens: {np.percentile(output_sizes, 90)}")
    return data

def load_code_contests():

    dataset = load_dataset("deepmind/code_contests", split="test", streaming=True)
    data = []
    output_sizes = []
    for item in dataset:

        data.append({
            'input': item['description'],
            'output': item['solutions']['solution'][0]
        })
        output_sizes.append(len(item['solutions']['solution'][0].split()))

    print(f"[INFO] Average number of tokens: {np.mean(output_sizes)}")
    print(f"[INFO] P90 number of tokens: {np.percentile(output_sizes, 90)}")

    return data

def load_dummy_data():

    num_samples = 10
    input_text = """Mikhail walks on a Cartesian plane. He starts at the point $(0, 0)$, and in one move he can go to any of eight adjacent points. For example, if Mikhail is currently at the point $(0, 0)$, he can go to any of the following points in one move: $(1, 0)$; $(1, 1)$; $(0, 1)$; $(-1, 1)$; $(-1, 0)$; $(-1, -1)$; $(0, -1)$; $(1, -1)$. If Mikhail goes from the point $(x1, y1)$ to the point $(x2, y2)$ in one move, and $x1 \ne x2$ and $y1 \ne y2$, then such a move is called a diagonal move. Mikhail has $q$ queries. For the $i$-th query Mikhail's target is to go to the point $(n_i, m_i)$ from the point $(0, 0)$ in exactly $k_i$ moves. Among all possible movements he want to choose one with the maximum number of diagonal moves. Your task is to find the maximum number of diagonal moves or find that it is impossible to go from the point $(0, 0)$ to the point $(n_i, m_i)$ in $k_i$ moves. Note that Mikhail can visit any point any number of times (even the destination point!). -----Input----- The first line of the input contains one integer $q$ ($1 \le q \le 10^4$) — the number of queries. Then $q$ lines follow. The $i$-th of these $q$ lines contains three integers $n_i$, $m_i$ and $k_i$ ($1 \le n_i, m_i, k_i \le 10^{18}$) — $x$-coordinate of the destination point of the query, $y$-coordinate of the destination point of the query and the number of moves in the query, correspondingly. -----Output----- Print $q$ integers. The $i$-th integer should be equal to -1 if Mikhail cannot go from the point $(0, 0)$ to the point $(n_i, m_i)$ in exactly $k_i$ moves described above. Otherwise the $i$-th integer should be equal to the the maximum number of diagonal moves among all possible movements. -----Example----- Input 3 2 2 3 4 3 7 10 1 9 Output 1 6 -1 -----Note----- One of the possible answers to the first test case: $(0, 0) \to (1, 0) \to (1, 1) \to (2, 2)$. One of the possible answers to the second test case: $(0, 0) \to (0, 1) \to (1, 2) \to (0, 3) \to (1, 4) \to (2, 3) \to (3, 2) \to (4, 3)$. In the third test case Mikhail cannot reach the point $(10, 1)$ in 9 move"""

    data = []
    for i in range(num_samples):
        data.append({
            'input': input_text,
            'output': input_text
        })

    return data


if __name__ == "__main__":

    # data = load_conala()
    data = load_code_contests()
    
    
    