# CatZero-rust
The goal is to make implement alphazero inspired framework for which a set of rules and constraints can be the input.
The reason I didn't call it AlphaZero is that I don't know if google would appreciate it.

This project will probably end up in the dust, so excuses to any (if any) who comes across this project.
 
## What it will do:
1. Provide rules (somehow)
2. Play along these rules and learn.
3. Provide an API to play against

## In progress

## Todo
* Find out how to create a deep resnet in rust
  * What about leaf maybe?
* Implement a monte carlo tree search algorithm
  * from $s_{root}$ all the way to a leaf
* Train the neural network with the game
## Finished


## How AlphaZero works (globally)
How alpha zero works with a resnet split into two ends: Policy and Value networks.
The Policy network tells us what is the probability of a good move.
The Value network tells us wheter the state is winning, draw or losing.

1. Run a monte carlo tree simulation.
    * From state $s_{root}$ take all moves.
    * Simulate (alphazero did 800 simulations)
    * For each child state $s$ select a move
      * low visit count
      * high move probability (neural network)
      * high value (neural network)
      * With dirichlet noise $Dir(\alpha)$
    * Until childnode
    * Propagate back
2. Play the selected move
3. Go back to 1. untill the game is over
4. Train the neuralnetwork:
    * Train the policy for each state encountered
      * The move that mcts did compared to the move the network predicted (what if the game was lost?)
      * Compress the look ahead into the neural network
    * Train the values for each state encountered
      * The predicted outcome versus the actuall outcome
5. Go back to step 1 until satisfied with the strength of the algorithm.


## About the networks
A small description about the neural network
### Input
The input is represented as multiple planes and $T$ past states
for example, the input for chess is represented as follows:

| Description     | planes  |
| ----------------|---------|
| Player 1 pieces | 6 planes|
| Player 2 pieces | 6 planes|
| Repetitions?    | 2 planes|
| Colour (whose turn) | 1 plane |
| Total moves count | 1 plane |
| Player 1 castling (kingside, queenside) | 2 planes |
| Player 2 castling (kingside, queenside) | 2 planes |
| no progress count | 1 planes |

### Output
Because the neural network cannot change in size, all the different move destinations have to be encoded in the output.
Just like the input, it is encoded in planes.

Illegal moves are masked out of course.



## Other things
* Reminder to be able to save and load the neural network so we can continue from where we quit.
* Might be nice to collect a whole bunch of samples to train. So play multiple times with the same network and then improve? Test this out.