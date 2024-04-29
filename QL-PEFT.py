import random

import numpy as np


class QL_PEFT:
    def __init__(self, task, LearningRate, DiscountFactor, VM):
        self.VM = VM
        self.task = task
        self.LearningRate = LearningRate
        self.DiscountFactor = DiscountFactor

        self.Q_table = np.zeros((task + 1, task + 1), dtype=int)
        self.oct_table = np.zeros((self.task, self.VM), dtype=int)
        self.computation_costs = []
        for i in range(self.task + 1):
            self.Q_table[0][i] = i
            self.Q_table[i][0] = i

    def read_dag_pre(self):
        self.dag_relation_pre = {}
        filename = 'dag5_PEFT'
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split()
                task_list_pre = {}
                for line_ in lines:
                    line_list_ = line_.split()
                    if line_list_[0] == line_list[0]:
                        task_list_pre[int(line_list_[1])] = int(line_list_[2])
                        self.dag_relation_pre[int(line_list_[0])] = task_list_pre
        print(self.dag_relation_pre)

    def read_dag_suc(self):
        self.dag_relation_suc = {}
        filename = 'dag5_PEFT'
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split()
                task_list_suc = {}
                for line_ in lines:
                    line_list_ = line_.split()
                    if line_list_[1] == line_list[1]:
                        task_list_suc[int(line_list_[0])] = int(line_list_[2])
                        self.dag_relation_suc[int(line_list_[1])] = task_list_suc
        print(self.dag_relation_suc)

    def read_pro_avg(self):
        self.computation_costs = []
        self.avg_list = []
        filename = 'computation costs_PEFT.txt'
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                temp_sum = 0
                current_line = []
                for i in range(self.VM):
                    current_line.append(int(line.split()[i]))
                    temp_sum += float(line.split()[i])
                self.computation_costs.append(current_line)
                temp_avg = temp_sum / 3
                self.avg_list.append(temp_avg)

    def read_computation_costs(self):
        filename = 'computation costs_PEFT.txt'
        with open(filename, 'r') as file_object:
            lines = file_object.readlines()
            for line in lines:
                line_list = line.split()
                temp_list = []
                for i in range(len(line_list)):
                    temp_list.append(float(line_list[i]))
                self.computation_costs.append(temp_list)
        return self.computation_costs
    
    def cul_ranku(self):
        self.read_dag_pre()
        self.read_pro_avg()
        self.ranku_list = {}
        temp_task = self.task
        while temp_task > 0:
            if temp_task == self.task:
                self.ranku_list[temp_task] = self.avg_list[temp_task - 1]
                temp_task -= 1
            else:
                pre_list = list(self.dag_relation_pre[temp_task].keys())
                cost_list = list(self.dag_relation_pre[temp_task].values())
                if len(self.dag_relation_pre[temp_task]) == 1:
                    print(self.ranku_list)
                    self.ranku_list[temp_task] = self.avg_list[temp_task - 1] + cost_list[0] + self.ranku_list[
                        pre_list[0]]
                else:
                    max = 0
                    for suc in pre_list:
                        temp = self.ranku_list[suc] + self.dag_relation_pre[temp_task][suc] + self.avg_list[
                            temp_task - 1]
                        if temp > max:
                            max = temp
                    self.ranku_list[temp_task] = max
                temp_task -= 1
        print(self.ranku_list)

    def get_avaiable(self, ava_set, forbid_set, initial_set):
        ava_set.clear()
        for forbid in initial_set:
            if forbid not in forbid_set:
                ava_set.add(forbid)

    def check_father_node(self, temp_task, forbid_set):
        for father in list(self.dag_relation_suc[temp_task]):
            if father not in forbid_set:
                return False
        return True

    def update_Q(self):
        self.cul_ranku()
        self.read_dag_suc()
        route = []
        entry_task = 1
        for i in range(300000):
            temp_route = [1]
            temp_task = entry_task
            forbid_set = set()
            ava_set = set()
            initial_set = set()
            for j in range(1, self.task + 1):
                initial_set.add(j)
            # Q_table_temp = self.Q_table
            while temp_task != self.task:
                chosen_set = set()
                temp_Q = 0
                if temp_task == 1:
                    forbid_set.add(temp_task)
                    chosen_set = self.dag_relation_pre[temp_task].keys()
                    suc_task = random.choice(list(chosen_set))
                    for choice in chosen_set:
                        if self.Q_table[temp_task][choice] > temp_Q:
                            temp_Q = self.Q_table[temp_task][choice]
                    self.Q_table[temp_task][suc_task] = self.Q_table[temp_task][suc_task] + self.LearningRate * (
                            self.ranku_list[suc_task] + self.DiscountFactor * temp_Q - self.Q_table[temp_task][
                        suc_task])
                    temp_task = suc_task
                    forbid_set.add(temp_task)
                    temp_route.append(temp_task)
                    self.get_avaiable(ava_set, forbid_set, initial_set)
                else:
                    for choice in list(ava_set):
                        if self.check_father_node(choice, forbid_set):
                            chosen_set.add(choice)
                    suc_task = random.choice(list(chosen_set))
                    for choice in chosen_set:
                        if self.Q_table[temp_task][choice] > temp_Q:
                            temp_Q = self.Q_table[temp_task][choice]
                    self.Q_table[temp_task][suc_task] = self.Q_table[temp_task][suc_task] + self.LearningRate * (
                            self.ranku_list[suc_task] + self.DiscountFactor * temp_Q - self.Q_table[temp_task][
                        suc_task])
                    temp_task = suc_task
                    temp_route.append(temp_task)
                    forbid_set.add(temp_task)
                    self.get_avaiable(ava_set, forbid_set, initial_set)

        print(self.Q_table)

    def get_order(self):
        self.update_Q()
        task_order = [1]
        temp_task = 1
        while temp_task != self.task:
            max_q = 0
            temp_task_ = temp_task
            for q in range(len(self.Q_table[temp_task])):
                if self.Q_table[temp_task][q] > max_q and q not in task_order:
                    max_q = self.Q_table[temp_task][q]
                    temp_task_ = q
            task_order.append(temp_task_)
            temp_task = temp_task_
        print(task_order)
        return task_order
    
    def compute_OCT(self):
        self.read_computation_costs()
        self.read_dag_pre()
        self.read_pro_avg()
        self.read_dag_suc()
        for q in range(self.task - 1, -1, -1):
            if q + 1 == self.task:
                for i in range(self.VM):
                    self.oct_table[self.task - 1][i] = 0
            else:
                for i in range(self.VM):
                    max_outer = 0
                    for suc in self.dag_relation_pre[q + 1].keys():
                        min_inner = self.oct_table[suc - 1][0] + self.computation_costs[suc - 1][0] + \
                                    self.dag_relation_suc[suc][q + 1]
                        for j in range(self.VM):
                            if i != j:
                                inner = self.oct_table[suc - 1][j] + self.computation_costs[suc - 1][j] + \
                                        self.dag_relation_suc[suc][q + 1]
                            else:
                                inner = self.oct_table[suc - 1][j] + self.computation_costs[suc - 1][j]
                            min_inner = min(min_inner, inner)
                        max_outer = max(max_outer, min_inner)
                    self.oct_table[q][i] = max_outer
        
    def select_processor(self):
        self.compute_OCT()
        oct_list = self.oct_table
        task_order = self.get_order()
        EFT_relation = {}
        pre_set = set()
        ava_set = {}
        for current_task in task_order:
            EST = [0, 0, 0]
            if current_task == 1:
                for i in range(self.VM):
                    EST[i] = self.computation_costs[current_task - 1][i]
                O_eft = EST[0] + oct_list[0][0]
                min_processor = 0
                for i in range(self.VM):
                    O_eft_temp = EST[i] + oct_list[0][i]
                    if O_eft_temp <= O_eft:
                        O_eft = O_eft_temp
                        EFT_relation[1] = {i + 1: EST[i]}
                        min_processor = i
                        print(EFT_relation)
                ava_set[min_processor + 1] = EST[min_processor]
                pre_set.add(1)
            else:
                for i in range(len(EST)):
                    for pre in list(self.dag_relation_suc[current_task].keys()):
                        if i + 1 == list(EFT_relation[pre].keys())[0]:
                            EST[i] = max(ava_set[i + 1], max(EST[i], list(EFT_relation[pre].values())[0]))
                        else:
                            if i + 1 not in ava_set:
                                EST[i] = max(
                                    (EST[i],
                                     list(EFT_relation[pre].values())[0] + self.dag_relation_suc[current_task][pre]))
                            else:
                                EST[i] = max(ava_set[i + 1], max((EST[i], list(EFT_relation[pre].values())[0] +
                                                                  self.dag_relation_suc[current_task][pre])))
                print(EST)
                EFT = [0, 0, 0]
                for i in range(self.VM):
                    EFT[i] = EST[i] + self.computation_costs[current_task - 1][i]
                O_eft = EFT[0] + oct_list[current_task - 1][0]
                min_processor = 0
                for i in range(len(EST)):
                    O_eft_temp = EST[i] + self.computation_costs[current_task - 1][i] + oct_list[current_task - 1][i]
                    if O_eft_temp <= O_eft:
                        O_eft = O_eft_temp
                        EFT_relation[current_task] = {i + 1: EFT[i]}
                        min_processor = i
                ava_set[min_processor + 1] = EFT[min_processor]
                pre_set.add(current_task)

        makespan = max(list(ava_set.values()))
        print("makespan=" + f"{makespan}")
instance = QL_PEFT(10, 1.0, 0.8, 3)
instance.select_processor()

