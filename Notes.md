# Recurrent Neural Networks  
(https://www.youtube.com/watch?v=_aCuOwF1ZjU)  

- Similar structure to a traditional NN
  - RNN are __not__ feed forward
- Output of a layer is fed into the next layer as an input
  - Use t1 to predict t2 and so on
- Stacking can be achieved to increase functionality
- Trained via Backpropagation
  - Potential issues can arise with Vanishing Gradients
    - Using increasing amounts of information to train the network
  - __Gating__ can help with this issue
- Forecasting is best achieved with RNNs

# Time Series Forecasting with RNNs in Python  
(https://www.youtube.com/watch?v=S8tpSG6Q2H0)  

- Can work with non-stationary data
- Scaling data is important
