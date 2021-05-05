318634110
308281435
*****
Comments:
Description of evaluation function:
This function evaluates a state using a weighted sum of three features:
1. The number of empty tiles on the current board.
2. The number of possible pairs that can be merged on the board- adjacent same numbered pairs are counted
    (diagonal pairs aren't counted since they can't be merged) and a higher weight in the sum is given to
    larger numbers pairs.
3. The value of the max numbered tile on the board.
The higher the score of these features the better the state.
The most weight is given to the number of empty tiles on the board- the game ends when the board is full
then the top priority is to keep as large an amount of empty spaces.
The second most weight is given to the amount of possible pairs that we can merge- making pairs is the only
way to increase score and to clear tiles from the board so we would like to prefer moves that have the most
future potential to increase score and to clear tiles from the board.
In order to get the most potential from future merging of pairs, we give a preference to higher value pairs.
(The pairs weight might seem small but the sum generated from pairs is quite large)
The past feature has the least weight and it's mostly a smaller preference for merging the biggest tile
rather then merging smaller tiles.
The specific numbers chosen for the weights were carefully optimized during multiple runs to maximize score
and the highest tile value.