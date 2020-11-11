```
[elementwise,...]->fused_elementwise
    expose kernel
[elementwise,...,reduction]->fused_elementwise_redcution
[elementwise,...,reduction,broadcast,elementwise...]->fused_elementwise_reduction_broadcast_elementwise
[elementwise,...,reduction,dot,elementwise]->fused_elementwise_reduction_dot_elementwise
```

1. reduce to 1
1. reduce at one axis
1. elementwise which need broadcast
1. dynamic shape?

