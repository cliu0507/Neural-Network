# Vanilla Recurrent Neural Network - Sequence Generator


###Outline of the data:
    we’ll be building a no frills RNN that accepts a binary sequence X and uses it to predict a binary sequence Y.
    The sequences are constructed as follows:
       Input Sequence(X):
           At time step t, X(t) has a 50% chance of being 1 (and a 50% chance of being 0). E.g., X might be [1, 0, 0, 1, 1, 1 … ].
       Output Sequence(Y):
            At time step t:
            Y(t) has a base 50% chance of being 1 (and a 50% base chance to be 0).
            The chance of Y(t) being 1 is increased by 50% (i.e., to 100%) if X(t−3) is 1.
            The chance of Y(t) being 1 is decreased by 25% (i.e., to 25%) if X(t−8) is 1.
            For example, if both X(t-3) and X(t-8) are 1, the chance of Y(t) being 1 is 50%(base) + 50% - 25% = 75%
       Thus there are two dependencies in the data, one at t-3(three steps back) and one at t-8(eight steps back)

###Model Architecture:
    At time step t, for t ∈ {0,1,...,n} the model accepts a (one-hot) binary X(t) vector and a previous state vector, 
    S(t-1).
    
    The modele will produces a state vector S(t) and a predicted probability distribution vector P(t) ∈ R2.

    Formally, the model is :
    S(t) = tanh(W * (X(t) @ S(t−1))+ b(s))
    P(t) = Softmax(U * S(t) + b(p))
    
    where @ represents vector concatenation, X(t) ∈ R2 (two dimensions) is a one-hot binary vector
    X(t)    ∈  2
    S(t-1)  ∈  d
    W       ∈  d×(2+d), 
    b(s)    ∈  d, 
    U       ∈  2×d, 
    b(p)    ∈  2
    
    d is the size of state vector (Here we set d = 4),
    at the step 0, S(-1) is intialized as a vector of zeros
    

###How wide should our Tensorflow graph be?
    how wide should our graph be? 
    How many time steps of input should our graph accept at once?
    
    We can then execute our graph for each time step, feeding in the state returned from the previous execution into the current execution. 
    This would work for a model that was already trained, but there’s a problem with using this approach for training: 
    the gradients computed during backpropagation are graph-bound. We would only be able to backpropagate errors to the current timestep;
    we could not backpropagate the error to time step t-1. 
    This means our network will not be able to learn how to store long-term dependencies (such as the two in our data) in its state.
    
    Solution:  “truncate” our backpropagation by backpropagating errors a maximum of nsteps.
    
    Details:
    Alternatively, we might make our graph as wide as our data sequence. This often works, except that in this case,
    we have an arbitrarily long input sequence, so we have to stop somewhere. Let’s say we make our graph accept sequences of length 10,000. 
    This solves the problem of graph-bound gradients, and the errors from time step 9999 are propagated all the way back to time step 0. 
    Unfortunately, such backpropagation is not only (often prohibitively) expensive, but also ineffective, 
    
    due to the vanishing / exploding gradient problem: 
    it turns out that backpropagating errors over too many time steps often causes them to vanish (become insignificantly small) 
    or explode (become overwhelmingly large). 

###Tuncate
    1. A natural interpretation:
        A natural interpretation of backpropagating errors a maximum of n
         steps means that we backpropagate every possible error n
        steps. That is, if we have a sequence of length 49, and choose n=7
        , we would backpropagate 42 of the errors the full 7 steps.
    
    2. Tensorflow approach:
        Tensorflow’s approach is to limit the graph to being n units wide. 
        See Tensorflow’s writeup on Truncated Backpropagation (“[Truncated backpropagation] 
        is easy to implement by feeding inputs of length [n] at a time and doing backward pass after each iteration.”). 
        This means that we would take our sequence of length 49, break it up into 7 sub-sequences of length 7 that 
        we feed into the graph in 7 separate computations, and that only the errors from the 7th input in each graph are backpropagated the full 7 steps.
        
#Use Tensor to represent width
    
    
    
