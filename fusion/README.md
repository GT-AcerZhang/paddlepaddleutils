```
[elementwise,...]->elementwise
[elementwise,...,reduction]->redcution
[reduction,broadcast]->broadcast
[broadcast,elementwise]->elementwise
[reduction,broadcast,mul]->mul
```

1. reduce to 1
1. reduce at one axis
1. elementwise which need broadcast

