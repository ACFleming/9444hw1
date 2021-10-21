"""
   encoder.py
   COMP9444, CSE, UNSW
"""
import torch

# REPLACE ch34 WITH YOUR OWN DATA
# TO REPRODUCE IMAGE SHOWN IN SPEC
ch34 = torch.Tensor(
    [
       # 1 2 3 4 5 6 7 8 9 0 1 2 3 4      5 6 7 8 9 0 1 2 3
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,     1,1,1,1,1,1,1,1,1], #1
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,     0,0,0,0,0,0,0,0,0], #2
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,     0,0,0,0,0,1,1,1,1], #3
        [0,0,1,1,1,1,1,1,1,1,1,1,1,1,     0,0,0,0,1,1,1,1,1], #4
        [0,0,1,1,1,1,1,1,1,1,1,1,1,1,     0,0,0,0,0,0,1,1,1], #5
        [0,0,0,1,1,1,1,1,1,1,1,1,1,1,     0,0,0,1,1,1,1,1,1], #6
        [0,0,0,1,1,1,1,1,1,1,1,1,1,1,     0,0,0,0,0,0,0,1,1], #7
        [0,0,0,0,1,1,1,1,1,1,1,1,1,1,     0,0,1,1,1,1,1,1,1], #8
        [0,0,0,0,1,1,1,1,1,1,1,1,1,1,     0,0,0,0,0,0,0,1,1], #9
        [0,0,0,0,0,1,1,1,1,1,1,1,1,1,     0,0,0,1,1,1,1,1,1], #10
        [0,0,0,0,0,1,1,1,1,1,1,1,1,1,     0,0,0,0,0,0,0,1,1], #11
        [0,0,0,0,0,0,1,1,1,1,1,1,1,1,     0,0,0,0,1,1,1,1,1], #12
        [0,0,0,0,0,0,1,1,1,1,1,1,1,1,     0,0,0,0,0,0,0,1,1], #13
        [0,0,0,0,0,0,0,1,1,1,1,1,1,1,     1,1,1,1,1,1,1,1,1], #14
        [0,0,0,0,0,0,0,1,1,1,1,1,1,1,     0,0,0,0,1,1,1,1,1], #15
        [0,0,0,0,0,0,0,1,1,1,1,1,1,1,     0,0,0,0,0,0,0,0,1], #16
        [0,0,0,0,0,0,0,0,1,1,1,1,1,1,     0,0,0,0,1,1,1,1,1], #17
        [0,0,0,0,0,0,0,0,1,1,1,1,1,1,     0,0,0,0,0,0,0,0,1], #18
        [0,0,0,0,0,0,0,0,0,1,1,1,1,1,     0,0,0,1,1,1,1,1,1], #19
        [0,0,0,0,0,0,0,0,0,1,1,1,1,1,     0,0,0,0,0,0,0,0,1], #20
        [0,0,0,0,0,0,0,0,0,0,1,1,1,1,     0,1,1,1,1,1,1,1,1], #21
        [0,0,0,0,0,0,0,0,0,0,1,1,1,1,     0,0,1,1,1,1,1,1,1], #22
        [0,0,0,0,0,0,0,0,0,0,1,1,1,1,     0,0,0,0,0,0,0,0,1], #23
        [0,0,0,0,0,0,0,0,0,0,0,1,1,1,     0,1,1,1,1,1,1,1,1], #24
        [0,0,0,0,0,0,0,0,0,0,0,1,1,1,     0,0,0,0,1,1,1,1,1], #25
        [0,0,0,0,0,0,0,0,0,0,0,1,1,1,     0,0,0,0,0,0,0,0,1], #26
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,     0,1,1,1,1,1,1,1,1], #27
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,     0,0,0,1,1,1,1,1,1], #28
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,     0,0,0,0,0,1,1,1,1], #29
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,     0,0,0,0,0,0,1,1,1], #30
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,     0,0,0,0,0,0,0,1,1], #31
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,     0,0,1,1,1,1,1,1,1], #32
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,     1,1,1,1,1,1,1,1,1], #33
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,     0,0,0,0,0,0,0,0,0], #34
    ])


# print(torch.count_nonzero(ch34, dim=0))
# print(torch.count_nonzero(ch34,dim=1))


star16 = torch.Tensor(
    [[1,1,0,0,0,0,0,0],
     [0,1,1,0,0,0,0,0],
     [0,0,1,1,0,0,0,0],
     [0,0,0,1,1,0,0,0],
     [0,0,0,0,1,1,0,0],
     [0,0,0,0,0,1,1,0],
     [0,0,0,0,0,0,1,1],
     [1,0,0,0,0,0,0,1],
     [1,1,0,0,0,0,0,1],
     [1,1,1,0,0,0,0,0],
     [0,1,1,1,0,0,0,0],
     [0,0,1,1,1,0,0,0],
     [0,0,0,1,1,1,0,0],
     [0,0,0,0,1,1,1,0],
     [0,0,0,0,0,1,1,1],
     [1,0,0,0,0,0,1,1]])
