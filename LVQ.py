# -*- coding:utf-8 -*-
import random
import math

input_layer = [[39, 281], [18, 307], [24, 242], [54, 333], [322, 35], [352, 17], [278, 22], [382, 48]]  # 输入节点
category_list = [0, 0, 0, 0, 1, 1, 1, 1]  # 对应着输入节点的类别（两类:0/1）
category = 2


class LVQ:
    """
    Learning Vector Quantization

    <1>所以对于输入的数据，需要给出其种类标签label。
    <2>找到距离当前输入节点（i）最近的输出层节点（o）之后，som直接调节o的特征使之趋近于i，
    而lvq则首先比较i与o的种类标签label是否是相同的，如果相同，则代表二者属于相同的种类，
    并调节o的特征使之趋近于i；反之，则调节o的特征使之远离于i。
    """

    def __init__(self, category):
        self.input = input_layer  # 输入样本
        self.output = []  # 输出数据
        self.output_cate = []  # 存储着输出样本的种类
        self.step_alpha = 0.5  # 步长 初始化为0.5
        self.step_alpha_del_rate = 0.95  # 步长衰变率
        self.category = category  # 类别个数
        self.category_list = category_list
        self.output_length = len(self.input[0])  # 输出样本特征个数 2
        self.d = [0.0] * self.category

    # 初始化 output_layer
    def initial_output(self):
        for i in range(self.category):
            self.output_cate.append(i)  # 主要分为两类 0/1 。初始化输出类别列表,分别对应着两个输出节点
            self.output.append([])
            for _ in range(self.output_length):
                self.output[i].append(random.randint(0, 400))

    # lvq 算法的主要逻辑
    # 计算某个输入样本 与 所有的输出节点之间的距离,存储于 self.d 之中
    def calc_distance(self, a_input):
        self.d = [0.0] * self.category
        for i in range(self.category):
            w = self.output[i]
            # self.d[i] =
            for j in range(len(a_input)):
                self.d[i] += math.pow((a_input[j] - w[j]), 2)  # 就不开根号了

    # 计算一个列表中的最小值 ，并将最小值的索引返回
    @staticmethod
    def get_min(a_list):
        min_index = a_list.index(min(a_list))
        return min_index

    # 将输出节点朝着当前的节点逼近或远离
    def move(self, a_input_index, a_input, min_output_index):
        # 作为有监督式学习 这里体现了与som算法不同之处（也是唯一的不同之处）
        if self.category_list[a_input_index] == self.output_cate[min_output_index]:
            for i in range(len(self.output[min_output_index])):
                self.output[min_output_index][i] = self.output[min_output_index][i] + self.step_alpha * (a_input[i] - self.output[min_output_index][i])
        else:
            for i in range(len(self.output[min_output_index])):
                self.output[min_output_index][i] = self.output[min_output_index][i] - self.step_alpha * (a_input[i] - self.output[min_output_index][i])

    # lvq 主要逻辑 (一次循环)
    def train(self):
        # for a_input in self.input_layer:
        for i in range(len(self.input)):
            a_input = self.input[i]
            self.calc_distance(a_input)
            min_output_index = self.get_min(self.d)
            self.move(i, a_input, min_output_index)

    # 循环执行 train 直到稳定
    def loop(self):
        generate = 0
        text = "代数:{0} 此时步长:{1} 输出节点:{2} 对应的类别为:{3}"
        while self.step_alpha >= 0.0001:  # 这样子会执行167代
            print(text.format(generate, self.step_alpha, self.output, self.output_cate))
            self.train()
            generate += 1
            self.step_alpha *= self.step_alpha_del_rate  # 步长衰减


if __name__ == '__main__':
    lvq = LVQ(category)
    lvq.initial_output()
    lvq.loop()
