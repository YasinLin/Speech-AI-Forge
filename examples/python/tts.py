import random
from client import SAFClient
client = SAFClient()

import concurrent.futures

import time


def process_iteration(i):
    audio = client.tts({"text": "多少钱？才我们的9块9毛钱，5斤啊，宝贝。你去外面买个两个三个，是不是可能都得花个10六十多块钱，20来块钱。你们经常买菜知道的，您在我这里买一大堆回去，宝贝。看到没有，5斤没有？" + str(random.randint(1, 1000)), "spk":"啊啊", "model": "cosy-voice"})
    with open("example"+str(i)+".mp3", "wb") as f:
        f.write(audio)

# 总迭代次数
total_iterations = 1
# 线程池中的线程数量
num_threads = 1
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    start_time = time.time()
    # 提交任务到线程池
    futures = [executor.submit(process_iteration, i) for i in range(total_iterations)]
    # 等待所有任务完成
    concurrent.futures.wait(futures)

average_time =  (time.time() - start_time) / total_iterations
print(f"平均迭代时间: {average_time} 秒")
print("All iterations are processed.")



# total_iterations = 10
# total_time = 0
# for i in range(1, total_iterations):
#     start_time = time.time()
#     thread = threading.Thread(target=process_iteration, args=(start, end))
#     threads.append(thread)
#     thread.start()
#     end_time = time.time()
#     total_time += end_time - start_time
#     average_time = total_time / i
#     print(f"平均迭代时间: {average_time} 秒")