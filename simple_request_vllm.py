import requests
import json
import time

if __name__ == "__main__":
        

    # Define the URL and the data payload
    url = 'http://localhost:7000/v2/models/vllm_model/generate'
    
    input_text = """Mikhail walks on a Cartesian plane. He starts at the point $(0, 0)$, and in one move he can go to any of eight adjacent points. For example, if Mikhail is currently at the point $(0, 0)$, he can go to any of the following points in one move: $(1, 0)$; $(1, 1)$; $(0, 1)$; $(-1, 1)$; $(-1, 0)$; $(-1, -1)$; $(0, -1)$; $(1, -1)$. If Mikhail goes from the point $(x1, y1)$ to the point $(x2, y2)$ in one move, and $x1 \ne x2$ and $y1 \ne y2$, then such a move is called a diagonal move. Mikhail has $q$ queries. For the $i$-th query Mikhail's target is to go to the point $(n_i, m_i)$ from the point $(0, 0)$ in exactly $k_i$ moves. Among all possible movements he want to choose one with the maximum number of diagonal moves. Your task is to find the maximum number of diagonal moves or find that it is impossible to go from the point $(0, 0)$ to the point $(n_i, m_i)$ in $k_i$ moves. Note that Mikhail can visit any point any number of times (even the destination point!). -----Input----- The first line of the input contains one integer $q$ ($1 \le q \le 10^4$) — the number of queries. Then $q$ lines follow. The $i$-th of these $q$ lines contains three integers $n_i$, $m_i$ and $k_i$ ($1 \le n_i, m_i, k_i \le 10^{18}$) — $x$-coordinate of the destination point of the query, $y$-coordinate of the destination point of the query and the number of moves in the query, correspondingly. -----Output----- Print $q$ integers. The $i$-th integer should be equal to -1 if Mikhail cannot go from the point $(0, 0)$ to the point $(n_i, m_i)$ in exactly $k_i$ moves described above. Otherwise the $i$-th integer should be equal to the the maximum number of diagonal moves among all possible movements. -----Example----- Input 3 2 2 3 4 3 7 10 1 9 Output 1 6 -1 -----Note----- One of the possible answers to the first test case: $(0, 0) \to (1, 0) \to (1, 1) \to (2, 2)$. One of the possible answers to the second test case: $(0, 0) \to (0, 1) \to (1, 2) \to (0, 3) \to (1, 4) \to (2, 3) \to (3, 2) \to (4, 3)$. In the third test case Mikhail cannot reach the point $(10, 1)$ in 9 move. Write C++ code to solve this problem. """

    data = {"text_input": input_text, 
            "parameters": 
                {
                "stream": False, 
                "temperature": 1,
                "max_tokens": 1036,
                }
            }
    # Send the POST request
    start_time = time.time()
    response = requests.post(url, json=data)
    end_time = time.time()

    print(f"Total time: {end_time-start_time:.2f}s")

    # Check if the request was successful and print the response
    if response.status_code == 200:
        # print("Response from server:", response.json())
        start_idx = len(input_text)
        text_output = response.json()['text_output'][start_idx:]
        print(text_output)
        print(len(text_output.split()))
        # print(response.json())

    else:
        print("Error:", response.status_code)
