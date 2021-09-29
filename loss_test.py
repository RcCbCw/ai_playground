import torch
from torch import nn

out_test = torch.tensor([[6.9, 24],
                         [8.3, 52],
                         [9, 25]])
out_answers = torch.tensor([[7, 25],
                            [10, 45],
                            [8.95, 24.5]])
loss_fn = nn.MSELoss()

loss1 = loss_fn(out_test[0], out_answers[0])
# print(loss1)
loss2 = loss_fn(out_test[1], out_answers[1])
# print(loss2)

loss3 = loss_fn(out_test, out_answers)
# print(loss3)

diff_tensor = out_answers - out_test
diff_tensor = diff_tensor.abs()
correct_ans = torch.zeros(len(diff_tensor))
for i, (x, y) in enumerate(diff_tensor):
    correct_ans[i] = 1 if x < 0.1 and y < 2 else 0
# diff_tensor = torch.where(
#     diff_tensor[:] < torch.tensor([0.2, 8]), 1, 0)
# print(diff_tensor)
# diff_tensor = diff_tensor[:][0] * diff_tensor[:][1]
print(diff_tensor)
print(correct_ans)
ratio = correct_ans.sum() / len(correct_ans)
print(ratio)
