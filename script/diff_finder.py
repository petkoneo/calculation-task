import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import multiprocessing
from timeit import default_timer as timer   
from statistics import mode

start = timer()

sample_set = pd.read_table('../test_set_2/ts2_input.txt', names = ['Test'])

# Prvý riadok vstupu udáva počet testovacích prípadov T. 
T = sample_set.Test[0]

# Druhý riadok vstupu udáva percento testovacích prípadov P, na ktoré musíte správne odpovedať, 
# aby sa vaše riešenie považovalo za správne.
P = sample_set.Test[1]

n = 2
test_cases = sample_set.drop(index = sample_set.index[:n])

decision_array = [-3.00, 3.00]

def split(word):
    return [char for char in word]

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def fair_probability(si, qj):
    return sigmoid(si - qj)

def calculate_differences(data_frame):
    row_lengths = 10000
    Qjs = [np.random.uniform(decision_array[0], decision_array[1]) for i in range(row_lengths)]
    diff_vals = []
    for index, row in data_frame.iterrows():
        const_of_diff = 0
        Si = np.random.uniform(decision_array[0], decision_array[1])

        responses = np.array(split(row['Test'])).astype('int32')
        sim_responses = np.array([fair_probability(Si, Qjs[i]) > 0.5 for i in range(row_lengths)])

        for (sim_res, response) in zip(sim_responses, responses):
            if sim_res != response:
                const_of_diff += 1

        diff_vals.append(const_of_diff)

    max_value = max(diff_vals)
    return diff_vals.index(max_value)

def post_processing(sol):
    final_counter = {}

    for j in range(50):
        final_counter[j] = []

    for key in sol:
      for i in range(len(sol[key])):
          final_counter[i].append(sol[key][i])

    for key in final_counter:
      out = mode(final_counter[key])
      final_counter[key] = out
    output = {}

    for key in final_counter:
        output[key + 1] = final_counter[key] + 1
    
    return output

# reset index
test_cases.reset_index(inplace=True)

## Chunking
n = int(len(test_cases['Test']) / int(T))  #chunk row size
list_df = [test_cases[i:i+n] for i in range(0, test_cases.shape[0],n)]

if __name__ == '__main__':
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
      bootstrapped_sol = {}
      for m in range(1000):
          solutions = p.map(calculate_differences, list_df)
          bootstrapped_sol[m] = list(solutions)

      with open('midput3_test.txt', 'w') as f:
          f.write(json.dumps(bootstrapped_sol))

      out = post_processing(bootstrapped_sol)
      with open('out_3.txt', 'w') as f:
        for key in out:
            message = 'Case #' +  str(key) + ': ' + str(out[key]) + '\n'
            f.write(message)
      print("time without multiprovessing:", timer()-start) 
