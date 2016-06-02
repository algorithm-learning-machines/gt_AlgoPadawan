require "CustomSharpeners"
print(nn.MulSoftMax)

gigi = nn.MulSoftMax()
print(gigi:forward(torch.Tensor{0.1, 0.3}))
