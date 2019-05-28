r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.5
    lr = 0.1
    reg = 0.001
    # raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.3
    lr_vanilla = 0.021
    lr_momentum = 0.004
    lr_rmsprop = 0.0003
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.3
    lr = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)

# Explain the graphs of no-dropout vs dropout. Do they match what you expected to see?
#
# If yes, explain why and provide examples based on the graphs.
# If no, explain what you think the problem is and what should be modified to fix it.
# Compare the low-dropout setting to the high-dropout setting and explain based on your graphs.
part2_q1 = r"""
**Your answer:**
1. Dropout is a technique to improve generalization of deep models.
Therefore we thought that the models with dropout are less likely to overfit (low train loss and high test loss).
As seen in the graphs the results support our expectations.
To the non-dropout model has a test loss curve that goes down at the first iterations 
and then rises up at the later iterations, while the train loss curve goes down consistently. 
This phenomenon points that this model has an overfit problem.

1. The differences between the low dropout model and the high dropout model can be seen mainly in the train-loss and train-accuracy graphs. 
It seems that a high dropout factor leads to a faster learning rate. 
We note that the test results are very similar in both models.
"""

part2_q2 = r"""
**Your answer:**
Yes, the cross entropy loss function takes into account all of the class probabilities 
and thus it might change (for better or worse) while the accuracy stays the same 
(because the model chooses the most probable class) or changes."""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
1. First, we can see that less when we have small filters, depth decreases accuracy. K32_L2 is the best at test_acc,
while any L>4 doesn't study well at all. This because of low filter count, which limits the contribution of additional
layers.

2. Yes, any L>4 was untrainable. This is due to the amount of calculations needed for each feature extraction.
There are so many convolutions that distort the input into unneeded data, and the network is too complex for the small
size of parameters in each layer. we can fix this by increasing learning rate, or increasing filter sizes, or 
limiting the pooling frequency, so we won't lose vital information, at the cost of training time.
"""

part3_q2 = r"""
We can see a few interesting thins:
1. When L>4, the accuracy increased dramatically, when K has grown. This is as expected, because with bigger filters
We can allow ourselves to deepen the network. This is one of the solutions we offered in previous question.
2. At low L's We can see there is a better accuracy for the higher K's,  
"""

part3_q3 = r"""
In previous experiments, we saw, at the better cases ~50% accuracy. This model gets ~55% accuracy which is better,
but this only happens on L=1,2. At L>2, we get untrainable networks.
This could be explained by the destructive behavior of using same-size filters in consecutive order. As we saw in previous
questions, depth gets better when we increase filter size as well.  
We also see better performance for the L=1, which supports our claim. This net studied both faster and with higher accuracy.
"""


part3_q4 = r"""
1. We've made the following modifications:
    a. added a dropout to the network of 0.6
    b. added batch normalization layers to both the feature extractor, after each pooling.
    c. 
    
2. We can see this model reaches ~70-80% accuracy, which is better than all we've seen, which suggests that batch
normalization is a vital tool for deep networks.
"""
# ==============