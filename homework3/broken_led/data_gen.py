def data_generation():
    inputs = []
    outputs_2 = []
    outputs_3 = []

    nums = [-1, -1, -1, -1, -1, -1, -1]
    vec_2 = [1, -1, 1, 1, 1, -1, 1]
    vec_3 = [1, -1, 1, 1, -1, 1, 1]
    for i in range(128):
        inputs.append(nums.copy())
        nums[6] = -nums[6]
        if i % 2 == 0:
            nums[5] = -nums[5]
        if i % 4 == 0:
            nums[4] = -nums[4]
        if i % 8 == 0:
            nums[3] = -nums[3]
        if i % 16 == 0:
            nums[2] = -nums[2]
        if i % 32 == 0:
            nums[1] = -nums[1]
        if i % 64 == 0:
            nums[0] = -nums[0]

    for i in range(128):
        prob_2 = 1.0
        prob_3 = 1.0

        for j in range(7):
            if inputs[i][j] == vec_2[j]:
                prob_2 *= 0.9
            else:
                prob_2 *= 0.1
            if inputs[i][j] == vec_3[j]:
                prob_3 *= 0.9
            else:
                prob_3 *= 0.1

        outputs_2.append(prob_2)
        outputs_3.append(prob_3)

    print(inputs)
    print(outputs_2)
    print(outputs_3)
    return inputs, outputs_2, outputs_3


if __name__ == '__main__':
    data_generation()
